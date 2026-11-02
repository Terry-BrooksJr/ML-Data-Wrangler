"""Manual-triage case files: what led to the decision, reviewer evidence, re-runs.

When a review ends in manual_triage, open_case records how it got there, so
whoever picks it up (a reviewer, then the arbiter) starts from the engine's
work instead of from scratch:
- the route: the model abstained, a code check downgraded it, the models
  disagreed, or the engine kept failing;
- each model's answer (first, confirmation, the arbiter's preference and
  its reasoning), the guard overrides and the open questions;
- the reasoning trail: each answer's thinking and raw reply. It can be long,
  so it goes to S3 (MODEL_BUCKET, under triage-cases/) when storage is set
  up, else on the case itself.

A reviewer then adds evidence (evidence.py: a Jira RSS export or an Aha
note). Only the extracted fields are kept, never the file, and evidence is
add-only. rerun_case sends the ticket, the case and the evidence to the
arbiter model; its answer is saved as a new review (rerun_of) that can be
marked correct for training_export, and a case it resolves feeds the
week's lessons learned (feedback.suggest_lessons_summary).

Cases live in their own collection (TRIAGE_CASES_COLLECTION, default
triage_cases), one per manual-triage review.
"""

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Iterable, List, Optional

import bson
from bson import ObjectId
from loguru import logger
from pymongo import ReturnDocument
from pymongo.collection import Collection
from pymongo.errors import DuplicateKeyError

from . import classify, decisions
from .feedback import week_of
from .identity import current_user
from .payload import build_ticket_payload, support_comment_times, wrap_untrusted

OPEN, RESOLVED = "open", "resolved"
# How a review came to be manual triage.
ABSTAINED = "abstained"
GUARD_OVERRIDE = "guard_override"
DISAGREEMENT = "disagreement"
ENGINE_ERROR = "engine_error"
ROUTE_LABELS = {
    ABSTAINED: "the model declined to classify",
    GUARD_OVERRIDE: "a code check downgraded the model's answer",
    DISAGREEMENT: "the models disagreed",
    ENGINE_ERROR: "the engine failed repeatedly",
}
EVIDENCE_KINDS = ("jira", "aha")
TRAIL_PREFIX = "triage-cases/"
# Above this, the trail goes to S3; without S3 its thinking is cut to fit.
MAX_INLINE_TRAIL_BYTES = 512 * 1024
# Extracted evidence, per entry and per case. All of a case's evidence goes
# to the arbiter on every re-run, so the case total keeps that request well
# inside a model's input limit (roughly 100K tokens of evidence).
MAX_EVIDENCE_BYTES = 256 * 1024
MAX_CASE_EVIDENCE_BYTES = 384 * 1024
MAX_EVIDENCE_ITEMS = 25
# MongoDB refuses a document over 16 MiB; past this, a re-run's trail is
# left out of the case rather than kept inline.
MAX_CASE_BYTES = 12 * 1024 * 1024
# A model's thinking shown to the arbiter on a re-run, per answer.
THINKING_SHOWN_CHARS = 6000


class RerunUnavailable(RuntimeError):
    """The case can't be re-run now: no arbiter, no new evidence, or one is running."""


class EvidenceRejected(ValueError):
    """The evidence can't be added: a duplicate, too large, or the case is full."""


def get_cases_collection(database) -> Collection:
    """The triage_cases collection; the name is read here so a saved setting applies at once."""
    return database[os.environ.get("TRIAGE_CASES_COLLECTION", "triage_cases")]


def ensure_indexes(cases: Collection) -> None:
    cases.create_index("review_id", unique=True)
    cases.create_index("reruns.review_id")  # cases_for_reviews
    cases.create_index([("zendesk_id", 1), ("created_at", -1)])


# --- opening a case ---------------------------------------------------------------


def route_of(result: dict) -> str:
    if result.get("engine_error") or result.get("model_version") == "none":
        return ENGINE_ERROR
    arbitration = result.get("arbitration") or {}
    if arbitration and result.get("classification") == "manual_triage":
        return DISAGREEMENT
    # A check may have downgraded the first answer before a confirmation.
    overrides = [
        *(result.get("guard_overrides") or []),
        *((result.get("confirmation_of") or {}).get("guard_overrides") or []),
    ]
    if any(o.startswith("Downgraded from") for o in overrides):
        return GUARD_OVERRIDE
    return ABSTAINED


def _answer(role: str, answer: dict) -> dict:
    return {
        "role": role,
        "model": answer.get("model_version"),
        "classification": answer.get("classification"),
        "confidence": answer.get("confidence"),
        "signals_matched": list(answer.get("signals_matched") or []),
        "reasoning": answer.get("reasoning") or "",
        "open_questions": list(answer.get("open_questions") or []),
    }


def answers_of(result: dict) -> List[dict]:
    """Each answer that led to the result, oldest first."""
    first = result.get("confirmation_of")
    answers = []
    if first:
        answers.append(_answer("first", first))
        # After a disagreement (classify._disagreement) the result is a
        # synthetic manual triage; the confirmation's own answer is kept
        # under arbitration.confirmation. Its other fields are unchanged.
        confirmation = {
            **result,
            **((result.get("arbitration") or {}).get("confirmation") or {}),
        }
        answers.append(_answer("confirmation", confirmation))
    else:
        answers.append(_answer("first", result))
    return answers


def _trail_sources(result: dict) -> List[tuple]:
    """(role, model, audit) for each call behind the result."""
    first = result.get("confirmation_of")
    sources = [("first", first), ("confirmation", result)] if first else [("first", result)]
    found = [(role, a.get("model_version"), a.get("audit") or {}) for role, a in sources]
    arbiter = (result.get("arbitration") or {}).get("model")
    for call, audit in (result.get("arbiter_audits") or {}).items():
        found.append((f"arbiter_{call}", arbiter, audit or {}))
    return found


def _trail(result: dict) -> dict:
    """The thinking and raw reply behind each call, keyed by role: first,
    confirmation, and arbiter_judge / arbiter_ruling after a disagreement."""
    trail = {}
    for role, model, audit in _trail_sources(result):
        if audit.get("thinking") or audit.get("raw_response"):
            trail[role] = {
                "model": model,
                "thinking": list(audit.get("thinking") or []),
                "raw_response": audit.get("raw_response"),
            }
    return trail


def _replies_kept(where: dict) -> set:
    """The roles whose raw reply the stored trail holds in full."""
    if where.get("stored") == "s3":
        return set(where.get("roles") or [])
    if where.get("stored") == "inline":
        return {r for r, e in (where.get("trail") or {}).items() if e.get("raw_response")}
    return set()


def drop_kept_replies(tickets: Collection, zendesk_id: str, review: dict, where: dict) -> None:
    """Removes the raw replies the case trail now holds from the ticket's
    review, so they're kept once, with the case."""
    kept = _replies_kept(where)
    paths = []
    if review.get("confirmation_of"):
        if "first" in kept:
            paths.append("confirmation_of.audit.raw_response")
        if "confirmation" in kept:
            paths.append("audit.raw_response")
    elif "first" in kept:
        paths.append("audit.raw_response")
    # One path per update: mongomock applies only one positional path of
    # several, and there are at most two.
    for path in paths:
        tickets.update_one(
            {"zendesk_id": zendesk_id, "agent_reviews.review_id": review["review_id"]},
            {"$unset": {f"agent_reviews.$.{path}": ""}},
        )


def _storage():
    """(config, client) for S3 trail storage, or None when it isn't set up."""
    from core import model_storage

    try:
        cfg = model_storage.config()
    except model_storage.StorageNotConfigured as e:
        logger.warning(f"Triage trails stay in MongoDB: {e}")
        return None
    if cfg is None:
        return None
    return cfg, model_storage.make_client(cfg)


def store_trail(name: str, trail: dict, inline_room: Optional[int] = None) -> dict:
    """Saves a reasoning trail and returns where it went.

    S3 when storage is set up: {"stored": "s3", "bucket", "key", "sha256",
    "bytes", "roles"}. Otherwise inline: {"stored": "inline", "trail": ...},
    its thinking cut so the whole stays under MAX_INLINE_TRAIL_BYTES, or
    under inline_room when that is smaller; with no room at all it is left
    out ({"stored": "dropped"}). `name` must be unique to the trail.
    """
    if not trail:
        return {"stored": None}
    body = json.dumps(trail, ensure_ascii=False, default=str).encode("utf-8")
    try:
        storage = _storage()
    except Exception as e:  # a storage problem must never lose the case
        logger.warning(f"Triage trail storage unavailable ({e}); keeping it in MongoDB.")
        storage = None
    if storage is not None:
        cfg, client = storage
        key = f"{TRAIL_PREFIX}{name}.json"
        try:
            client.put_object(
                Bucket=cfg.bucket,
                Key=key,
                Body=body,
                ContentType="application/json",
            )
            return {
                "stored": "s3",
                "bucket": cfg.bucket,
                "key": key,
                "sha256": hashlib.sha256(body).hexdigest(),
                "bytes": len(body),
                "roles": sorted(r for r, e in trail.items() if e.get("raw_response")),
            }
        except Exception as e:
            logger.warning(f"Couldn't upload triage trail {key} ({e}); keeping it in MongoDB.")
    limit = MAX_INLINE_TRAIL_BYTES if inline_room is None else min(MAX_INLINE_TRAIL_BYTES, inline_room)
    if limit < 4096:
        logger.warning(f"Triage trail {name} left out: its case is near MongoDB's size limit.")
        return {"stored": "dropped"}
    return {"stored": "inline", "trail": _fit_inline(trail, limit)}


def _fit_inline(trail: dict, limit: int = MAX_INLINE_TRAIL_BYTES) -> dict:
    trail = json.loads(json.dumps(trail, default=str))
    while len(json.dumps(trail).encode("utf-8")) > limit:
        longest = max(
            (
                (len(text), role, i)
                for role, entry in trail.items()
                for i, text in enumerate(entry.get("thinking") or [])
            ),
            default=None,
        )
        if longest is None or longest[0] < 1000:
            for entry in trail.values():
                entry["raw_response"] = None
            break
        _, role, i = longest
        text = trail[role]["thinking"][i]
        trail[role]["thinking"][i] = text[: len(text) // 2] + "\n[cut to fit]"
    return trail


def load_trail(case: dict) -> dict:
    """The case's reasoning trail, fetched from S3 when it was stored there.

    Returns {} when there is none or it can't be fetched.
    """
    where = case.get("trail") or {}
    if where.get("stored") == "inline":
        return where.get("trail") or {}
    if where.get("stored") != "s3":
        return {}
    try:
        storage = _storage()
        if storage is None:
            return {}
        _, client = storage
        body = client.get_object(Bucket=where["bucket"], Key=where["key"])["Body"].read()
        if where.get("sha256") and hashlib.sha256(body).hexdigest() != where["sha256"]:
            logger.warning(f"Triage trail {where['key']} doesn't match its hash; ignored.")
            return {}
        return json.loads(body)
    except Exception as e:
        logger.warning(f"Couldn't fetch triage trail {where.get('key')}: {e}")
        return {}


def open_case(
    cases: Collection,
    zendesk_id: str,
    review: dict,
    result: dict,
    tickets: Optional[Collection] = None,
) -> Optional[ObjectId]:
    """Records a manual-triage review's case file. A second call for the
    same review changes nothing.

    Args:
        review: The agent_reviews entry as saved (its review_id).
        result: The engine's result, with its audits' thinking still in it
            (or the review itself, when recovering a missing case).
        tickets: When given, the raw replies the case trail now holds are
            removed from the ticket's review (drop_kept_replies).

    Returns:
        The case's _id, or None if the review isn't manual triage.
    """
    if review.get("classification") != "manual_triage":
        return None
    ensure_indexes(cases)
    case_id = ObjectId()
    arbitration = result.get("arbitration") or None
    if arbitration:
        arbitration = {
            key: arbitration.get(key)
            for key in ("model", "preferred", "preferred_reasoning", "ruling", "error")
        }
    open_questions = list(result.get("open_questions") or [])
    for answer in answers_of(result):
        open_questions += [q for q in answer["open_questions"] if q not in open_questions]
    now = datetime.now(timezone.utc)
    doc = {
        "_id": case_id,
        "zendesk_id": zendesk_id,
        "review_id": review["review_id"],
        "created_at": now,
        "status": OPEN,
        "route": route_of(result),
        "prompt_version": (review.get("audit") or {}).get("prompt_version"),
        "input_sha256": (review.get("input") or {}).get("payload_sha256"),
        "answers": answers_of(result),
        "arbitration": arbitration,
        "guard_overrides": list(result.get("guard_overrides") or []),
        "engine_error": review.get("engine_error"),
        "open_questions": open_questions,
        "trail": store_trail(str(case_id), _trail(result)),
        "evidence": [],
        "evidence_bytes": 0,
        "reruns": [],
    }
    try:
        cases.insert_one(doc)
    except DuplicateKeyError:
        existing = cases.find_one({"review_id": review["review_id"]}, {"_id": 1})
        return existing["_id"] if existing else None
    logger.info(f"{zendesk_id}: manual-triage case {case_id} opened ({doc['route']}).")
    if tickets is not None:
        drop_kept_replies(tickets, zendesk_id, review, doc["trail"])
    return case_id


def needs_case(review: dict) -> bool:
    """A manual-triage review of the engine's own, not a re-run."""
    return review.get("classification") == "manual_triage" and review.get("rerun_of") is None


def recover_missing_cases(cases: Collection, rows: Iterable[dict], found: dict) -> int:
    """Opens the cases that saving a review should have opened but didn't
    (persistence._open_case logs and carries on), from the reviews
    themselves. Their trail holds what the review kept: no thinking.

    Args:
        rows: {"zendesk_id", "review"} rows, as the Triage tab loads them.
        found: {review_id: case} already found (cases_for_reviews).

    Returns:
        How many were opened.
    """
    opened = 0
    for row in rows:
        review = row["review"]
        if needs_case(review) and review.get("review_id") not in found:
            try:
                if open_case(cases, row["zendesk_id"], review, review):
                    opened += 1
            except Exception as e:
                logger.warning(f"{row['zendesk_id']}: still couldn't open its manual-triage case: {e}")
    if opened:
        logger.info(f"Opened {opened} manual-triage case(s) that were missing.")
    return opened


def cases_for_reviews(cases: Collection, review_ids: Iterable) -> dict:
    """{review_id: case} for the reviews that have one (original or re-run)."""
    ids = [i for i in review_ids if i is not None]
    if not ids:
        return {}
    found = {}
    for case in cases.find(
        {"$or": [{"review_id": {"$in": ids}}, {"reruns.review_id": {"$in": ids}}]},
        {"trail.trail": 0},
    ):
        found[case["review_id"]] = case
        for rerun in case.get("reruns") or []:
            found[rerun["review_id"]] = case
    return found


# --- evidence ---------------------------------------------------------------------


def add_evidence(
    cases: Collection,
    case_id,
    kind: str,
    data: dict,
    source: str,
    content_sha256: str,
    added_by: Optional[str] = None,
) -> dict:
    """Adds a reviewer's extracted evidence to a case. Add-only: nothing
    here edits or removes an entry.

    Args:
        kind: "jira" or "aha".
        data: The fields evidence.py extracted (never the file).
        source: The file name, or "pasted".
        content_sha256: The original content's hash, so the same export
            can't be added twice.

    Returns:
        The stored entry.

    Raises:
        EvidenceRejected: a duplicate, too large (alone, or with the case's
            other evidence: MAX_CASE_EVIDENCE_BYTES), or the case is full.
        LookupError: no such case.
    """
    if kind not in EVIDENCE_KINDS:
        raise ValueError(f"kind must be one of {EVIDENCE_KINDS}")
    size = len(bson.encode({"data": data}))
    if size > MAX_EVIDENCE_BYTES:
        raise EvidenceRejected(
            f"The extracted {kind} data is {size // 1024} KB; the limit is "
            f"{MAX_EVIDENCE_BYTES // 1024} KB. Export fewer issues or a shorter history."
        )
    entry = {
        "evidence_id": ObjectId(),
        "kind": kind,
        "source": source,
        "sha256": content_sha256,
        "added_by": added_by or current_user(),
        "added_at": datetime.now(timezone.utc),
        "data": data,
    }
    room = MAX_CASE_EVIDENCE_BYTES - size
    updated = cases.find_one_and_update(
        {
            "_id": ObjectId(case_id),
            "evidence.sha256": {"$ne": content_sha256},
            f"evidence.{MAX_EVIDENCE_ITEMS - 1}": {"$exists": False},
            # A running total, so two reviewers adding at once can't pass it.
            "$or": [
                {"evidence_bytes": {"$lte": room}},
                {"evidence_bytes": {"$exists": False}},
            ],
        },
        {"$push": {"evidence": entry}, "$inc": {"evidence_bytes": size}},
        return_document=ReturnDocument.AFTER,
    )
    if updated is not None:
        return entry
    case = cases.find_one({"_id": ObjectId(case_id)}, {"evidence.sha256": 1, "evidence_bytes": 1})
    if case is None:
        raise LookupError(f"No triage case {case_id}")
    if any(e.get("sha256") == content_sha256 for e in case.get("evidence") or []):
        raise EvidenceRejected("This evidence was already added to the case.")
    if len(case.get("evidence") or []) >= MAX_EVIDENCE_ITEMS:
        raise EvidenceRejected(f"The case already holds {MAX_EVIDENCE_ITEMS} pieces of evidence.")
    raise EvidenceRejected(
        f"The case's evidence would total {(case.get('evidence_bytes', 0) + size) // 1024} KB; "
        f"the limit is {MAX_CASE_EVIDENCE_BYTES // 1024} KB, so a re-run fits the model's input. "
        "Add a shorter export."
    )


def unused_evidence(case: dict) -> List[dict]:
    """Evidence added since the last re-run (all of it before the first)."""
    used = {i for r in case.get("reruns") or [] for i in r.get("evidence_ids") or []}
    return [e for e in case.get("evidence") or [] if e["evidence_id"] not in used]


# --- re-running -------------------------------------------------------------------


def _wrap_strings(value):
    """Wraps every string in a value as untrusted text, keeping its shape."""
    if isinstance(value, str):
        return wrap_untrusted(value)
    if isinstance(value, list):
        return [_wrap_strings(v) for v in value]
    if isinstance(value, dict):
        return {k: _wrap_strings(v) for k, v in value.items()}
    return value


def _iso(value):
    return value.isoformat() if isinstance(value, datetime) else value


def rerun_fields(case: dict, trail: Optional[dict] = None) -> dict:
    """The triage_case and reviewer_evidence fields a re-run payload adds.

    Model names are left out, so no answer is weighed by who gave it. Every
    model- or reviewer-supplied string is wrapped as untrusted text: a
    reasoning or a Jira comment can quote the customer.
    """
    trail = trail or {}
    answers = []
    for answer in case.get("answers") or []:
        shown = {key: answer.get(key) for key in ("role", "classification", "confidence")}
        # Model-written: a reply can put any text in its trigger list.
        shown["signals_matched"] = [wrap_untrusted(s) for s in answer.get("signals_matched") or []]
        shown["reasoning"] = wrap_untrusted(answer.get("reasoning"))
        thinking = "\n\n".join((trail.get(answer["role"]) or {}).get("thinking") or [])
        if thinking:
            shown["working_notes"] = wrap_untrusted(thinking[:THINKING_SHOWN_CHARS])
        answers.append(shown)
    arbitration = case.get("arbitration") or {}
    triage_case = {
        "why_manual_triage": ROUTE_LABELS.get(case.get("route"), case.get("route")),
        "answers": answers,
        "guard_overrides": [wrap_untrusted(o) for o in case.get("guard_overrides") or []],
        "open_questions": [wrap_untrusted(q) for q in case.get("open_questions") or []],
    }
    if arbitration.get("preferred"):
        triage_case["arbiter_preferred_on_reasoning"] = {
            "classification": arbitration["preferred"],
            "reasoning": wrap_untrusted(arbitration.get("preferred_reasoning")),
            "then_ruled_on_ticket": arbitration.get("ruling"),
        }
    for rerun in case.get("reruns") or []:
        triage_case.setdefault("earlier_reruns", []).append(
            {
                "classification": rerun.get("classification"),
                "confidence": rerun.get("confidence"),
                "open_questions": [wrap_untrusted(q) for q in rerun.get("open_questions") or []],
            }
        )
    evidence = [
        {
            "kind": e["kind"],
            "added_at": _iso(e.get("added_at")),
            "data": _wrap_strings(e.get("data") or {}),
        }
        for e in case.get("evidence") or []
    ]
    return {"triage_case": triage_case, "reviewer_evidence": evidence}


def _claim(cases: Collection, case_id, worker: str) -> Optional[dict]:
    now = datetime.now(timezone.utc)
    from .persistence import claim_ttl

    return cases.find_one_and_update(
        {
            "_id": ObjectId(case_id),
            "$or": [
                {"rerun_claim": {"$exists": False}},
                {"rerun_claim": None},
                {"rerun_claim.expires_at": {"$lt": now}},
            ],
        },
        {"$set": {"rerun_claim": {"worker": worker, "expires_at": now + claim_ttl()}}},
        return_document=ReturnDocument.AFTER,
    )


def rerun_problem(case: dict, arbiter: Optional[str] = None) -> Optional[str]:
    """Why the case can't be re-run now, or None.

    Args:
        arbiter: The arbiter model to use; defaults to the one in Settings.
    """
    if not (arbiter or classify.arbiter_model()):
        return "No arbiter model is set: add one in Settings to re-run with evidence."
    if not case.get("evidence"):
        return "Add Jira or Aha evidence before re-running."
    if not unused_evidence(case):
        return "No evidence has been added since the last re-run."
    return None


def _rerun_update(rerun: dict) -> dict:
    """The case update recording a re-run, releasing the claim.

    A non-manual-triage answer resolves the case; a manual_triage one
    (re)opens it, its open questions becoming the case's current ones.
    """
    update = {"$push": {"reruns": rerun}, "$unset": {"rerun_claim": ""}}
    if rerun["classification"] != "manual_triage":
        update["$set"] = {
            "status": RESOLVED,
            "resolved_at": rerun["at"],
            "resolved_by_review": rerun["review_id"],
            "week_of": week_of(rerun["at"]),
        }
    else:
        update["$set"] = {"status": OPEN}
        if rerun.get("open_questions"):
            update["$set"]["open_questions"] = rerun["open_questions"]
        update["$unset"].update({"resolved_at": "", "resolved_by_review": "", "week_of": ""})
    return update


def _adopt_interrupted(tickets: Collection, cases: Collection, case: dict, worker: str) -> dict:
    """Records re-run reviews already on the ticket that the case doesn't
    list: a re-run stopped between saving its review and updating the case.
    Run while holding the claim, so a retry finishes that re-run instead of
    paying for it again.

    Returns:
        The case as it now stands.
    """
    listed = {r["review_id"] for r in case.get("reruns") or []}
    ticket = tickets.find_one({"zendesk_id": case["zendesk_id"]}, {"agent_reviews": 1}) or {}
    for review in ticket.get("agent_reviews") or []:
        if review.get("case_id") != case["_id"] or review["review_id"] in listed:
            continue
        rerun = {
            "review_id": review["review_id"],
            "at": review.get("reviewed_at"),
            "by": review.get("requested_by"),
            "model": review.get("model_version"),
            "classification": review.get("classification"),
            "confidence": review.get("confidence"),
            "signals_matched": review.get("signals_matched") or [],
            "open_questions": review.get("open_questions") or [],
            "evidence_ids": review.get("evidence_ids") or [],
            "trail": {"stored": None, "note": "recorded after an interrupted re-run"},
        }
        update = _rerun_update(rerun)
        update["$set"]["rerun_claim"] = case["rerun_claim"]  # keep holding it
        del update["$unset"]["rerun_claim"]
        cases.update_one({"_id": case["_id"], "rerun_claim.worker": worker}, update)
        logger.warning(f"{case['zendesk_id']}: recorded interrupted re-run {review['review_id']}.")
    return cases.find_one({"_id": case["_id"]})


def rerun_case(
    tickets: Collection,
    cases: Collection,
    case_id,
    requested_by: Optional[str] = None,
    config: Optional[classify.EngineConfig] = None,
) -> dict:
    """Classifies the case's ticket again on the arbiter model, with the
    case and its evidence, and saves the answer as a new review.

    The new review (rerun_of: the case's review) doesn't touch the ticket's
    queue state. A non-manual-triage answer resolves the case; another
    manual_triage leaves it (or puts it back) open for more evidence. A paid
    model call.

    The review is saved first, then the case, which only the claim's holder
    can update. A re-run that lost its claim takes its review back off the
    ticket; one that stopped in between is finished by the next attempt
    (_adopt_interrupted) without another call.

    Raises:
        RerunUnavailable: no arbiter, no new evidence, a re-run is already
            running, or this one lost its claim.
        LookupError: the case or its ticket is gone.
        anthropic.APIError or classify.EngineError: the call failed.
        decisions.DecisionModelNotAllowed: the arbiter is a decision model.
    """
    from .persistence import (
        _fit_review,
        _review_entry,
        default_worker_id,
    )

    config = config or classify.EngineConfig.from_env()
    arbiter = config.arbiter or classify.arbiter_model()
    worker = default_worker_id()
    case = cases.find_one({"_id": ObjectId(case_id)})
    if case is None:
        raise LookupError(f"No triage case {case_id}")
    problem = rerun_problem(case, arbiter)
    if problem:
        raise RerunUnavailable(problem)
    case = _claim(cases, case_id, worker)
    if case is None:
        raise RerunUnavailable("This case is already being re-run.")
    release = {"$unset": {"rerun_claim": ""}}
    try:
        case = _adopt_interrupted(tickets, cases, case, worker)
        # Another re-run may have used the evidence since the first check.
        problem = rerun_problem(case, arbiter)
        if problem:
            raise RerunUnavailable(problem)
        ticket = tickets.find_one({"zendesk_id": case["zendesk_id"]})
        if ticket is None:
            raise LookupError(f"Ticket {case['zendesk_id']} is no longer in MongoDB.")
        if decisions.is_decision_model(classify.get_client(), arbiter):
            raise decisions.DecisionModelNotAllowed(
                f"{arbiter} is a decision model: it can't write the reasoning a re-run needs."
            )
        as_of = datetime.now(timezone.utc)
        evidence_ids = [e["evidence_id"] for e in case.get("evidence") or []]
        payload = build_ticket_payload(ticket, as_of, rerun_fields(case, load_trail(case)))
        result = classify._classify_with(
            arbiter,
            payload,
            reengage_count=len(ticket.get("reengage_attempts") or []),
            assignee=ticket.get("assignee"),
            max_tokens=config.max_tokens,
            effort=config.effort,
            support_comment_times=support_comment_times(ticket),
        )
        result["ticket_id"] = ticket["zendesk_id"]
        entry = _review_entry(result, payload=payload, as_of=as_of)
        entry.update(
            {
                "rerun_of": case["review_id"],
                "case_id": case["_id"],
                "evidence_ids": evidence_ids,
                "requested_by": requested_by or current_user(),
            }
        )
        trail = store_trail(
            f"{case['_id']}-rerun-{entry['review_id']}",
            _trail(result),
            inline_room=MAX_CASE_BYTES - len(bson.encode(case)),
        )
        if "first" in _replies_kept(trail) and entry.get("audit"):
            entry["audit"].pop("raw_response", None)  # kept once, in the trail
        rerun = {
            "review_id": entry["review_id"],
            "at": as_of,
            "by": entry["requested_by"],
            "model": arbiter,
            "classification": result["classification"],
            "confidence": result["confidence"],
            "signals_matched": result.get("signals_matched") or [],
            "open_questions": result.get("open_questions") or [],
            "evidence_ids": evidence_ids,
            "trail": trail,
        }
        _fit_review(tickets, ticket["zendesk_id"], entry, None)
        tickets.update_one(
            {"zendesk_id": ticket["zendesk_id"]}, {"$push": {"agent_reviews": entry}}
        )
        saved = cases.update_one(
            {"_id": case["_id"], "rerun_claim.worker": worker}, _rerun_update(rerun)
        )
        if not saved.modified_count and not cases.find_one(
            {"_id": case["_id"], "reruns.review_id": entry["review_id"]}, {"_id": 1}
        ):
            # Our claim expired and another re-run took the case over.
            tickets.update_one(
                {"zendesk_id": ticket["zendesk_id"]},
                {"$pull": {"agent_reviews": {"review_id": entry["review_id"]}}},
            )
            raise RerunUnavailable(
                "Another re-run took this case over while the model answered; this answer was discarded."
            )
    except BaseException:
        cases.update_one({"_id": case["_id"], "rerun_claim.worker": worker}, release)
        raise
    logger.info(
        f"{case['zendesk_id']}: re-run with evidence on {arbiter}: "
        f"{result['classification']} ({result['confidence']})"
    )
    return result


def resolved_cases_for_week(cases: Collection, week: datetime) -> List[dict]:
    """The cases a re-run resolved in that week, oldest first, without their trails."""
    return list(
        cases.find({"status": RESOLVED, "week_of": week_of(week)}, {"trail.trail": 0}).sort(
            "resolved_at", 1
        )
    )
