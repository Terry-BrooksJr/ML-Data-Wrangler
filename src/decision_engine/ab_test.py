"""A/B tests: the same tickets classified by two engine configurations.

A run samples tickets, classifies each one with configuration A (usually
the current engine) and configuration B (the one being considered), and
stores both answers in ab_results. A reviewer then reads the two answers
side by side, blind (each ticket shows them in a random order, as "Output
1" and "Output 2"), and records which is better. The summary reports the
preferences, how often the two agree, and what each cost in tokens.

Until a run is finalized nothing here writes to the tickets collection: an
A/B run never adds an agent review, claims a ticket or changes
awaiting_review. Finalizing a fully judged run adds each verdict to its
ticket's ab_test_verdicts and removes the run's bulky working data.

Answers are saved as they arrive and reused where possible, so a model is
asked about the same ticket content and configuration only once (within
REUSE_WINDOW); a failing variant is retried at most MAX_ATTEMPTS times. Each result keeps
a snapshot of its ticket taken when the run was created, so both variants,
and the reviewer, see the same ticket however it changes afterwards.

    python -m decision_engine.ab_test create --size 30 --b-primary claude-opus-5.5
    python -m decision_engine.ab_test run RUN_ID      # paid model calls
    python -m decision_engine.ab_test report RUN_ID
    python -m decision_engine.ab_test export RUN_ID results.csv   # or .json
    python -m decision_engine.ab_test finalize RUN_ID   # verdicts onto the tickets
"""

import argparse
import csv
import hashlib
import io
import json
import os
import pathlib
import random
import socket
import sys
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import bson
from loguru import logger
from pymongo import ReturnDocument
from pymongo.collection import Collection
from pymongo.errors import DuplicateKeyError

from . import decisions, pricing
from .classify import (
    classify_ticket,
    EFFORT_LEVELS,
    EngineConfig,
    is_fatal,
    worst_case_seconds,
)
from .guards import CLASSIFICATIONS
from .payload import build_ticket_payload, support_comment_times
from .prompts import PROMPT_VERSION

VARIANTS = ("A", "B")
# What a reviewer can record for one ticket, and how the summary names it.
PREFERENCES = {
    "A": "A better",
    "B": "B better",
    "tie": "Both equally good",
    "both_bad": "Both wrong",
}
# The parts of a classification kept for review; the reply's raw text and
# the system prompt aren't needed to compare two answers.
_KEPT_FIELDS = (
    "classification",
    "confidence",
    "signals_matched",
    "reengage_attempts",
    "reasoning",
    "internal_note",
    "draft_message",
    "leadership_recipient",
    "tag",
    "guard_overrides",
    # Set when the variant's models disagreed (classify.arbitrate).
    "arbitration",
    "model_version",
    # Set on answers from decision models: a classification and confidence,
    # with each category's probability, and no text.
    "decision_only",
    "probabilities",
)
# The ticket fields kept beside the payload, for the reviewer. The transcript
# isn't copied: the reviewer reads it from the payload the models were sent.
_SNAPSHOT_FIELDS = ("zendesk_id", "subject", "status")
# A claim (and an in-flight lock) older than claim_ttl() is treated as
# abandoned: its worker stopped. It is renewed before each variant's calls,
# so it must outlast one variant's worst case at the configured timeout
# (classify.worst_case_seconds: primary, secondary, confirmation and the
# arbiter's two calls, each with its retries), plus a margin.
CLAIM_MARGIN = timedelta(minutes=5)


def claim_ttl() -> timedelta:
    """How long a result's claim lasts, read from the timeout setting when used."""
    return timedelta(seconds=worst_case_seconds()) + CLAIM_MARGIN
# MongoDB refuses documents over 16 MiB; a result whose ticket is this big
# is left out of the run (and reported) rather than failing the whole run.
MAX_RESULT_BYTES = 15 * 1024 * 1024
# Run statuses. "creating" is only ever seen if creation was interrupted;
# "finalized" means the verdicts were copied to the tickets and the bulky
# working data removed (finalize).
CREATING, PENDING, RUNNING, PARTIAL, COMPLETE, FINALIZED = (
    "creating",
    "pending",
    "running",
    "partial",
    "complete",
    "finalized",
)
# A variant that failed this many times on a ticket isn't tried again.
MAX_ATTEMPTS = 3
# A saved answer is reused for the same ticket content and configuration
# only if it was made for a moment this close: the engine judges elapsed
# time ("no reply in 48 hours"), so an older answer may no longer hold.
REUSE_WINDOW = timedelta(hours=24)
# How much of a failed call's raw reply is kept for debugging.
RAW_REPLY_LIMIT = 20000
# The tickets field a finalized run's verdicts are added to.
VERDICTS_FIELD = "ab_test_verdicts"
# Fields finalize removes from each result: the ticket text and model input
# (copies of what is in the tickets collection), raw replies and claims.
# The answers, verdicts and content fingerprints stay, for reports and reuse.
_SLIMMED_FIELDS = (
    "payload",
    "ticket.subject",
    "ticket.transcript",  # in results saved before the transcript was dropped
    "ticket.assignee",
    "assignee",
    "claim",
    "started",
    *(f"outputs.{v}.raw_response" for v in ("A", "B")),
)


class Stopped(Exception):
    """The run was asked to stop; what finished is saved and it can resume."""


class PromptChanged(ValueError):
    """The system prompt changed after the run started classifying."""


class VerdictConflict(ValueError):
    """Someone else saved a verdict on the ticket since it was loaded."""


class RunFinalized(ValueError):
    """The run is finalized: its verdicts are on the tickets and can't change."""


class NotReadyToFinalize(ValueError):
    """The run still has tickets to classify or answers without a verdict."""


def collection_name_problem(
    tickets_name: str, runs_name: str, results_name: str
) -> Optional[str]:
    """Why these collection names can't be used together, or None.

    A/B documents written into the tickets collection, or runs and results
    sharing one collection, would corrupt both.
    """
    if runs_name == results_name:
        return f"The A/B runs and A/B results collections are both named {runs_name!r}; give them different names."
    for label, name in (("runs", runs_name), ("results", results_name)):
        if name == tickets_name:
            return f"The A/B {label} collection is named {name!r}, the same as the tickets collection; give it a different name."
    return None


def collection_names() -> Tuple[str, str, str]:
    """(tickets, ab_tests, ab_results) collection names, from the settings."""
    return (
        os.environ.get("MONGO_COLLECTION") or "tickets",
        os.environ.get("AB_TEST_COLLECTION") or "ab_tests",
        os.environ.get("AB_RESULT_COLLECTION") or "ab_results",
    )


def get_collections(database) -> Tuple[Collection, Collection]:
    """(ab_tests, ab_results) in the given database.

    The names are read here, not at import, so a saved setting applies at once.

    Raises:
        ValueError: The names collide with each other or with the tickets
            collection (collection_name_problem).
    """
    tickets_name, runs_name, results_name = collection_names()
    problem = collection_name_problem(tickets_name, runs_name, results_name)
    if problem:
        raise ValueError(problem)
    runs = database[runs_name]
    results = database[results_name]
    results.create_index([("run_id", 1), ("ticket_id", 1)], unique=True)
    results.create_index([("content_hash", 1), ("as_of", 1)])
    return runs, results


def _duplicates(ids: Iterable[str]) -> List[str]:
    return sorted(i for i, n in Counter(ids).items() if n > 1)


def _duplicate_error(duplicates: List[str]) -> ValueError:
    shown = ", ".join(duplicates[:10]) + (" …" if len(duplicates) > 10 else "")
    return ValueError(
        f"The tickets collection has more than one document for ticket(s) {shown}. "
        "Remove the duplicates (and give zendesk_id a unique index) before A/B testing."
    )


def sample_ticket_ids(
    tickets: Collection,
    size: int,
    seed: Optional[int] = None,
    awaiting_only: bool = False,
) -> List[str]:
    """A random sample of ticket IDs that have a transcript to classify.

    Args:
        size: How many to pick; fewer if fewer tickets qualify.
        seed: Makes the sample repeatable.
        awaiting_only: Only tickets currently awaiting review.

    Raises:
        ValueError: A ticket ID appears on more than one document.
    """
    query = {"transcript.0": {"$exists": True}}
    if awaiting_only:
        query["awaiting_review"] = True
    ids = sorted(
        str(doc["zendesk_id"]) for doc in tickets.find(query, {"zendesk_id": 1})
    )
    duplicates = _duplicates(ids)
    if duplicates:
        raise _duplicate_error(duplicates)
    return random.Random(seed).sample(ids, min(size, len(ids)))


# Stands in for as_of when fingerprinting a ticket, so the fingerprint
# depends on the ticket's content and not on when the run was created.
_FINGERPRINT_AS_OF = datetime(2000, 1, 1, tzinfo=timezone.utc)


def _snapshot(ticket: dict, as_of: datetime) -> dict:
    """What a result keeps of its ticket: the exact payload both variants
    are sent (which the reviewer reads the transcript from), the subject and
    status, and a fingerprint of the ticket's content for reusing answers
    (content_hash)."""
    reengage_count = len(ticket.get("reengage_attempts") or [])
    assignee = ticket.get("assignee")
    fingerprint = json.dumps(
        [
            build_ticket_payload(ticket, as_of=_FINGERPRINT_AS_OF),
            reengage_count,
            assignee,
        ],
        default=str,
    )
    return {
        "payload": build_ticket_payload(ticket, as_of=as_of),
        "reengage_count": reengage_count,
        "assignee": assignee,
        "support_comment_times": support_comment_times(ticket),
        "as_of": as_of,
        "content_hash": hashlib.sha256(fingerprint.encode()).hexdigest(),
        "ticket": {key: ticket.get(key) for key in _SNAPSHOT_FIELDS},
    }


def answer_key(config: EngineConfig) -> str:
    """Identifies what an answer depends on besides the ticket: the
    configuration and the prompts. Equal keys and content hashes mean the
    same question asked of the same models."""
    return hashlib.sha256(
        json.dumps([config.to_dict(), _versions()], sort_keys=True).encode()
    ).hexdigest()[:16]


def create_run(
    tickets: Collection,
    runs: Collection,
    results: Collection,
    a: EngineConfig,
    b: EngineConfig,
    ticket_ids: Iterable[str],
    name: str = "",
    created_by: str = "",
    seed: Optional[int] = None,
    reuse_answers: bool = True,
    prices: Optional[Dict[str, pricing.Price]] = None,
) -> str:
    """Records a new run and one pending result per ticket; classifies nothing.

    With reuse_answers, classifying the run copies an answer saved in any
    run for the same ticket content and configuration within REUSE_WINDOW,
    instead of calling the model again.

    prices (model -> Price) are saved with the run and price any call whose
    reply didn't report its charge (pricing.calls_cost); give today's
    (pricing.live_prices), corrected where needed.

    Each result keeps a snapshot of its ticket, including the payload built
    once (as of the run's creation), so A and B are sent exactly the same
    input and the reviewer reads what they were sent, even if the ticket
    changes later. Each ticket gets a random display order, so a reviewer
    can't tell which side is A.

    If saving the results fails part way, the run and its results are
    removed, so no partial sample is left behind. A ticket whose result
    would exceed MongoDB's document limit is left out and listed in the
    run's "skipped".

    Returns:
        The run's ID.

    Raises:
        ValueError: A and B are the same configuration, there are no
            tickets, a ticket is missing, or a ticket ID is duplicated.
    """
    if a == b:
        raise ValueError("A and B are the same configuration; change one of them.")
    ticket_ids = [str(t) for t in ticket_ids]
    if not ticket_ids:
        raise ValueError("No tickets to test.")
    duplicates = _duplicates(ticket_ids)
    if duplicates:
        raise ValueError(f"Ticket(s) listed more than once: {', '.join(duplicates)}.")
    found: Dict[str, List[dict]] = {}
    for ticket in tickets.find({"zendesk_id": {"$in": ticket_ids}}):
        found.setdefault(str(ticket["zendesk_id"]), []).append(ticket)
    duplicates = sorted(i for i, docs in found.items() if len(docs) > 1)
    if duplicates:
        raise _duplicate_error(duplicates)
    missing = [i for i in ticket_ids if i not in found]
    if missing:
        raise ValueError(f"Ticket(s) not found: {', '.join(missing[:10])}.")

    run_id = uuid.uuid4().hex[:12]
    shuffle = random.Random(seed)
    created_at = datetime.now(timezone.utc)
    docs, skipped = [], []
    for ticket_id in ticket_ids:
        doc = {
            "run_id": run_id,
            "ticket_id": ticket_id,
            "order": shuffle.sample(VARIANTS, 2),
            "outputs": {},
            "preference": None,
            "preference_version": 0,
            "claim": None,
            **_snapshot(found[ticket_id][0], created_at),
        }
        size = len(bson.encode(doc))
        if size > MAX_RESULT_BYTES:
            skipped.append(
                {
                    "ticket_id": ticket_id,
                    "reason": f"too large to store ({size / 1024 / 1024:.1f} MiB)",
                }
            )
        else:
            docs.append(doc)
    if skipped:
        logger.warning(
            f"A/B {run_id}: left out {len(skipped)} ticket(s) too large to store: "
            + ", ".join(s["ticket_id"] for s in skipped)
        )
    if not docs:
        raise ValueError("Every sampled ticket is too large to store; nothing to test.")
    runs.insert_one(
        {
            "_id": run_id,
            "name": name or f"{a.primary} vs {b.primary}",
            "created_at": created_at,
            "created_by": created_by,
            "variants": {"A": a.to_dict(), "B": b.to_dict()},
            **_versions(),
            "ticket_count": len(docs),
            "skipped": skipped,
            "reuse_answers": reuse_answers,
            "prices": {m: p.to_dict() for m, p in (prices or {}).items()},
            "status": CREATING,
        }
    )
    try:
        results.insert_many(docs)
        runs.update_one({"_id": run_id}, {"$set": {"status": PENDING}})
    except BaseException:
        results.delete_many({"run_id": run_id})
        runs.delete_one({"_id": run_id})
        raise
    return run_id


def _versions() -> Dict[str, str]:
    """The prompts a run's answers depend on: the engine's system prompt, and
    the question decision models are asked (decisions.py)."""
    return {
        "prompt_version": PROMPT_VERSION,
        "decision_instructions_version": decisions.INSTRUCTIONS_VERSION,
    }


def configs(run: dict) -> Dict[str, EngineConfig]:
    return {v: EngineConfig.from_dict(run["variants"][v]) for v in VARIANTS}


def models_of(*configs: EngineConfig) -> List[str]:
    """Every model the configurations can call, each once."""
    models = []
    for config in configs:
        for model in (
            config.primary,
            config.secondary,
            config.confirmation,
            config.arbiter,
        ):
            if model and model not in models:
                models.append(model)
    return models


def run_prices(run: dict) -> Dict[str, pricing.Price]:
    """The prices a run was created with (see create_run)."""
    return {
        model: pricing.Price.from_dict(values)
        for model, values in (run.get("prices") or {}).items()
    }


def _cost(calls: List[dict], prices: Dict[str, pricing.Price]) -> dict:
    """Tokens, time and dollars across every model call one classification made."""
    return {
        "usage": _sum_usage(call.get("usage") for call in calls),
        "latency_ms": sum(call.get("latency_ms") or 0 for call in calls),
        "model_calls": len(calls),
        **pricing.calls_cost(calls, prices),
    }


def _single_call(result_or_error, audit: dict) -> List[dict]:
    """The one call behind a result or error that has no model_calls list."""
    return [
        {
            "model": getattr(result_or_error, "model", None)
            or (
                result_or_error.get("model_version")
                if isinstance(result_or_error, dict)
                else None
            ),
            "usage": audit.get("usage"),
            "latency_ms": audit.get("latency_ms"),
            "billed_usd": audit.get("billed_cost_usd"),
        }
    ]


def _output(result: dict, prices: Optional[Dict[str, pricing.Price]] = None) -> dict:
    """What an A/B result keeps of a classification.

    Usage, latency and cost cover every model call behind the answer: a
    failed primary call, the secondary that replaced it, and a confirmation.
    """
    kept = {key: result.get(key) for key in _KEPT_FIELDS}
    audit = result.get("audit") or {}
    calls = result.get("model_calls") or _single_call(result, audit)
    kept.update(_cost(calls, prices or {}))
    confirmed = result.get("confirmation_of")
    if confirmed:
        kept["confirmed_from"] = {
            key: confirmed.get(key)
            for key in ("model_version", "classification", "confidence")
        }
    return kept


def _error_output(
    error: Exception, prices: Optional[Dict[str, pricing.Price]] = None
) -> dict:
    """A failed classification: the error, its cost, and the raw reply (or
    the API's error body) for debugging. The raw reply can quote ticket
    text; finalize removes it."""
    audit = getattr(error, "audit", None) or {}
    calls = getattr(error, "model_calls", None) or _single_call(error, audit)
    raw = audit.get("raw_response")
    if raw is None and getattr(error, "body", None) is not None:
        raw = json.dumps(error.body, default=str)
    return {
        "error": f"{type(error).__name__}: {error}"[:1000],
        "raw_response": raw[:RAW_REPLY_LIMIT] if raw else None,
        "stop_reason": audit.get("stop_reason"),
        "request_id": audit.get("request_id") or getattr(error, "request_id", None),
        **_cost(calls, prices or {}),
    }


def _sum_usage(usages: Iterable[dict]) -> dict:
    total: Counter = Counter()
    for usage in usages:
        for key, value in (usage or {}).items():
            total[key] += value or 0
    return dict(total)


def pending_variants(doc: dict, retry_interrupted: bool = False) -> List[str]:
    """The variants still to classify: never tried, or failed fewer than
    MAX_ATTEMPTS times. A variant whose call was interrupted (interrupted)
    isn't retried unless retry_interrupted: it may already have been charged."""
    outputs = doc.get("outputs") or {}
    return [
        v
        for v in VARIANTS
        if (
            v not in outputs
            or ("error" in outputs[v] and _attempts(outputs[v]) < MAX_ATTEMPTS)
        )
        and (retry_interrupted or v not in interrupted(doc))
    ]


def interrupted(doc: dict) -> List[str]:
    """The variants whose model call started but whose answer was never saved:
    the worker stopped mid-call (crashed, killed) and its claim has expired,
    so the call may have been billed. They wait for a deliberate retry."""
    started = doc.get("started") or {}
    claim = doc.get("claim") or {}
    live = claim.get("at") is not None and _aware(claim["at"]) >= _now() - claim_ttl()
    return [
        v
        for v in VARIANTS
        if v in started
        and not (live and claim.get("worker") == started[v].get("worker"))
    ]


def gave_up(doc: dict) -> List[str]:
    """The variants that failed MAX_ATTEMPTS times on this ticket."""
    outputs = doc.get("outputs") or {}
    return [
        v
        for v in VARIANTS
        if v in outputs
        and "error" in outputs[v]
        and _attempts(outputs[v]) >= MAX_ATTEMPTS
    ]


def _attempts(output: dict) -> int:
    return output.get("attempts") or 1


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _aware(moment: datetime) -> datetime:
    """moment with UTC attached; Mongo returns naive UTC datetimes."""
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def _claim(results: Collection, doc_id, worker: str) -> Optional[dict]:
    """Takes a result for this worker, so two workers never pay for the same
    ticket; returns the claimed document, or None if another worker has it."""
    now = _now()
    return results.find_one_and_update(
        {
            "_id": doc_id,
            "$or": [
                {"claim": None},
                {"claim.at": {"$lt": now - claim_ttl()}},
                {"claim.worker": worker},
            ],
        },
        {"$set": {"claim": {"worker": worker, "at": now}}},
        return_document=ReturnDocument.AFTER,
    )


def _renew(results: Collection, doc_id, worker: str) -> bool:
    """Extends this worker's claim before a model call, so it outlasts the
    call (claim_ttl() bounds one variant's calls); False if it was lost."""
    return (
        results.update_one(
            {"_id": doc_id, "claim.worker": worker},
            {"$set": {"claim.at": _now()}},
        ).matched_count
        > 0
    )


def _locks(results: Collection) -> Collection:
    """In-flight answers, one per ticket content and answer key, shared by
    every run, so two tests never pay for the same answer at once."""
    return results.database[f"{results.name}_locks"]


def _lock_answer(results: Collection, doc: dict, key: str, worker: str) -> bool:
    """Takes the in-flight lock for this ticket content and answer key; False
    if another worker (in any run) is getting that answer right now."""
    lock_id = f"{doc['content_hash']}:{key}"
    locks = _locks(results)
    now = _now()
    try:
        locks.insert_one({"_id": lock_id, "worker": worker, "at": now})
        return True
    except DuplicateKeyError:
        # Taken over only if its worker stopped (the lock outlived claim_ttl()).
        return (
            locks.update_one(
                {
                    "_id": lock_id,
                    "$or": [{"worker": worker}, {"at": {"$lt": now - claim_ttl()}}],
                },
                {"$set": {"worker": worker, "at": now}},
            ).matched_count
            > 0
        )


def _unlock_answer(results: Collection, doc: dict, key: str, worker: str) -> None:
    _locks(results).delete_one(
        {"_id": f"{doc['content_hash']}:{key}", "worker": worker}
    )


def _finish(runs: Collection, results: Collection, run_id: str) -> Optional[str]:
    """Records the run's status from what is saved: complete when nothing is
    left to classify (every ticket has both answers, or a variant gave up
    after MAX_ATTEMPTS failures) and no call was interrupted. A finalized
    run is left alone. Returns the status set, or None."""
    unfinished = sum(
        1
        for doc in results.find({"run_id": run_id})
        if pending_variants(doc) or interrupted(doc)
    )
    status = PARTIAL if unfinished else COMPLETE
    update = {"status": status}
    if status == COMPLETE:
        update["completed_at"] = _now()
    outcome = runs.update_one(
        {"_id": run_id, "status": {"$ne": FINALIZED}}, {"$set": update}
    )
    return status if outcome.matched_count else None


class _Refused(Exception):
    """run_ab_test couldn't start; carries the error to raise."""

    def __init__(self, error: Exception):
        super().__init__(str(error))
        self.error = error


def _start(runs: Collection, results: Collection, run_id: str) -> dict:
    """Marks the run running for this app's prompts, atomically.

    Workers on one run must use the same prompts. The run takes this app's
    prompt versions only if it has them already, or if no answer was saved
    yet and no other worker is running it; the versions and the running
    status are set together, so two app versions can't both start.

    Raises:
        _Refused: no such run, unfinished creation, finalized, or a
            different prompt (PromptChanged).
    """
    current = _versions()
    if results.count_documents({"run_id": run_id, "outputs": {"$ne": {}}}, limit=1):
        # Runs from before outputs_saved existed.
        runs.update_one(
            {"_id": run_id, "outputs_saved": {"$exists": False}},
            {"$set": {"outputs_saved": True}},
        )
    run = runs.find_one_and_update(
        {
            "_id": run_id,
            "status": {"$nin": [CREATING, FINALIZED]},
            "$or": [
                current,
                {"outputs_saved": {"$ne": True}, "status": {"$ne": RUNNING}},
            ],
        },
        {"$set": {**current, "status": RUNNING}},
        return_document=ReturnDocument.AFTER,
    )
    if run is not None:
        return run
    run = runs.find_one({"_id": run_id})
    if run is None:
        raise _Refused(ValueError(f"No A/B run {run_id}."))
    if run.get("status") == CREATING:
        raise _Refused(
            ValueError(f"A/B run {run_id} wasn't created completely; create a new run.")
        )
    if run.get("status") == FINALIZED:
        raise _Refused(
            RunFinalized(
                f"A/B run {run_id} is finalized; there is nothing to classify."
            )
        )
    was = ", ".join(
        f"{run.get(key)} now {version}"
        for key, version in current.items()
        if run.get(key) != version
    )
    raise _Refused(
        PromptChanged(
            f"A/B run {run_id} is being classified, or was classified, with a different "
            f"prompt ({was}). Answers from two prompts can't be compared; create a new run, "
            "or classify it with the app version that started it."
        )
    )


def run_ab_test(
    run_id: str,
    runs: Collection,
    results: Collection,
    classify: Callable[..., dict] = classify_ticket,
    on_progress: Optional[Callable[[int, int], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    worker: Optional[str] = None,
    retry_interrupted: bool = False,
) -> dict:
    """Classifies every pending ticket of a run with both configurations.

    Each ticket's saved payload is sent to both, so they see the same input.
    Resumable: an output is saved as soon as it arrives and isn't asked for
    again, but a failed one is retried by the next run (up to MAX_ATTEMPTS),
    and every attempt is kept in attempt_history. Errors that will hit every
    ticket (bad key, out of credit) stop the run, as in a review batch, after
    saving what they cost. Each ticket is claimed while it is classified,
    with the claim renewed before each call, so two workers on one run never
    classify the same ticket; with reuse, an in-flight lock per ticket
    content and answer key does the same across runs. A call interrupted by
    a crash isn't repeated unless retry_interrupted. However the call ends,
    the run is left "complete" or "partial", never "running".

    Args:
        on_progress: Called with (done, total) after each ticket.
        should_stop: Checked between tickets; True stops the run (Stopped).
        worker: Identifies this worker's claims; defaults to a fresh ID.
        retry_interrupted: Also retry calls a crashed worker left unfinished.

    Returns:
        The run's summary (summarize).

    Raises:
        ValueError: No such run, or its creation didn't finish.
        RunFinalized: The run is finalized.
        PromptChanged: The run uses a different system prompt.
    """
    worker = worker or f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"
    started = False
    try:
        try:
            run = _start(runs, results, run_id)
        except _Refused as refused:
            raise refused.error from None
        started = True
        docs = list(results.find({"run_id": run_id}).sort("ticket_id", 1))
        context = _RunContext(
            run_id=run_id,
            runs=runs,
            results=results,
            configs=configs(run),
            versions=_versions(),
            classify=classify,
            worker=worker,
            reuse=run.get("reuse_answers", False),
            prices=run_prices(run),
            retry_interrupted=retry_interrupted,
        )
        for done, doc in enumerate(docs, start=1):
            if should_stop and should_stop():
                raise Stopped(f"Stopped after {done - 1} of {len(docs)} tickets.")
            if pending_variants(doc, retry_interrupted):
                claimed = _claim(results, doc["_id"], worker)
                if claimed is None:
                    logger.info(
                        f"A/B {run_id}: ticket {doc['ticket_id']} is being classified by another worker."
                    )
                else:
                    try:
                        _classify_pending(claimed, context)
                    finally:
                        results.update_one(
                            {"_id": doc["_id"], "claim.worker": worker},
                            {"$set": {"claim": None}},
                        )
            if on_progress:
                on_progress(done, len(docs))
    except BaseException as error:
        # A refusal changed nothing; anything else may have left the run
        # marked running (even a failed start: its update may have applied).
        if started or not isinstance(error, (ValueError, RunFinalized, PromptChanged)):
            _finish(runs, results, run_id)
        raise
    _finish(runs, results, run_id)
    return summarize(
        runs.find_one({"_id": run_id}), list(results.find({"run_id": run_id}))
    )


@dataclass
class _RunContext:
    run_id: str
    runs: Collection
    results: Collection
    configs: Dict[str, EngineConfig]
    versions: Dict[str, str]
    classify: Callable[..., dict]
    worker: str
    reuse: bool
    prices: Dict[str, pricing.Price]
    retry_interrupted: bool


def _attempt_record(output: dict, attempt: int) -> dict:
    """What attempt_history keeps of one model attempt: its outcome and cost."""
    record = {
        key: output.get(key)
        for key in (
            "classification",
            "confidence",
            "error",
            "fatal",
            "raw_response",
            "request_id",
            "usage",
            "latency_ms",
            "model_calls",
            "cost_usd",
            "cost_basis",
        )
        if key in output
    }
    record["attempt"] = attempt
    record["at"] = _now()
    return record


def _classify_pending(doc: dict, ctx: "_RunContext") -> None:
    """Classifies a claimed ticket's pending variants and saves each output.

    For each variant: renew the claim, copy a reusable answer if there is one
    (taking the in-flight lock first, so only one run asks), otherwise mark
    the call started and ask the model. The output replaces outputs.<v>, and
    each model attempt (failed ones included) is added to
    attempt_history.<v>, so spend and debugging data are never lost. A fatal
    error is saved to the history (not as an attempt) and then raised.
    """
    for variant in pending_variants(doc, ctx.retry_interrupted):
        config = ctx.configs[variant]
        key = answer_key(config)
        previous = (doc.get("outputs") or {}).get(variant)
        attempt = _attempts(previous) + 1 if previous else 1
        if not _renew(ctx.results, doc["_id"], ctx.worker):
            logger.warning(
                f"A/B {ctx.run_id}: lost the claim on ticket {doc['ticket_id']}; stopping it."
            )
            return
        locked = False
        if ctx.reuse and doc.get("content_hash"):
            if not _lock_answer(ctx.results, doc, key, ctx.worker):
                logger.info(
                    f"A/B {ctx.run_id}: another test is getting ticket {doc['ticket_id']}'s "
                    f"{variant} answer; it will be reused next time."
                )
                continue
            locked = True
        try:
            reused = _reusable(ctx.results, doc, key) if ctx.reuse else None
            if reused is not None:
                output, record = reused, None
            else:
                output, record = _ask(doc, variant, key, attempt, ctx)
            output["attempts"] = attempt
            output["answered_at"] = _now()
            _check_versions(ctx)
            ctx.runs.update_one(
                {"_id": ctx.run_id, "outputs_saved": {"$ne": True}},
                {"$set": {"outputs_saved": True}},
            )
            update = {
                "$set": {f"outputs.{variant}": output},
                "$unset": {f"started.{variant}": ""},
            }
            if record is not None:
                update["$push"] = {f"attempt_history.{variant}": record}
            saved = ctx.results.update_one(
                {"_id": doc["_id"], "claim.worker": ctx.worker}, update
            )
            if saved.matched_count == 0:
                # The claim expired and another worker took the ticket over.
                logger.warning(
                    f"A/B {ctx.run_id}: lost the claim on ticket {doc['ticket_id']}; "
                    "its output wasn't saved."
                )
                return
        finally:
            if locked:
                _unlock_answer(ctx.results, doc, key, ctx.worker)


def _ask(doc: dict, variant: str, key: str, attempt: int, ctx: "_RunContext"):
    """Calls the model for one variant; returns (output, attempt record).

    The call is marked started first, so a crash mid-call is recognised as
    interrupted rather than silently paid for twice. A fatal error's cost is
    saved to attempt_history (it isn't counted as an attempt) before it is
    raised.
    """
    ctx.results.update_one(
        {"_id": doc["_id"], "claim.worker": ctx.worker},
        {"$set": {f"started.{variant}": {"worker": ctx.worker, "at": _now()}}},
    )
    try:
        output = _output(
            ctx.classify(
                doc["payload"],
                reengage_count=doc.get("reengage_count", 0),
                assignee=doc.get("assignee"),
                # None for a result snapshotted before it was kept: unchecked.
                support_comment_times=doc.get("support_comment_times"),
                config=ctx.configs[variant],
                allow_decision_models=True,
            ),
            ctx.prices,
        )
        output["answer_key"] = key
    except Exception as e:
        output = _error_output(e, ctx.prices)
        if is_fatal(e):
            output["fatal"] = True
            ctx.results.update_one(
                {"_id": doc["_id"], "claim.worker": ctx.worker},
                {
                    "$push": {
                        f"attempt_history.{variant}": _attempt_record(output, attempt)
                    },
                    "$unset": {f"started.{variant}": ""},
                },
            )
            raise
        logger.warning(
            f"A/B {ctx.run_id}: ticket {doc['ticket_id']} failed on {variant} "
            f"(attempt {attempt} of {MAX_ATTEMPTS}): {e}"
        )
    return output, _attempt_record(output, attempt)


def _check_versions(ctx: "_RunContext") -> None:
    """Refuses to save an answer if the run now uses other prompts (another
    app version took it over)."""
    run = ctx.runs.find_one({"_id": ctx.run_id}, {key: 1 for key in ctx.versions})
    if run is None or any(run.get(k) != v for k, v in ctx.versions.items()):
        raise PromptChanged(
            f"A/B run {ctx.run_id} now uses a different prompt; this answer wasn't saved."
        )


def _reusable(results: Collection, doc: dict, key: str) -> Optional[dict]:
    """A successful answer saved for the same ticket content and answer key,
    for a moment within REUSE_WINDOW of this ticket's, or None."""
    as_of = doc.get("as_of")
    if not doc.get("content_hash") or as_of is None:
        return None
    for other in results.find(
        {
            "_id": {"$ne": doc["_id"]},
            "content_hash": doc["content_hash"],
            "as_of": {"$gte": as_of - REUSE_WINDOW, "$lte": as_of + REUSE_WINDOW},
            "$or": [{f"outputs.{v}.answer_key": key} for v in VARIANTS],
        }
    ):
        for v in VARIANTS:
            output = (other.get("outputs") or {}).get(v) or {}
            if output.get("answer_key") == key and "error" not in output:
                copied = {
                    k: value
                    for k, value in output.items()
                    if k not in ("reused_from", "attempts", "answered_at")
                }
                copied["reused_from"] = {
                    "run_id": other["run_id"],
                    "ticket_id": other["ticket_id"],
                    "variant": v,
                }
                return copied
    return None


def estimate_calls(
    runs: Collection, results: Collection, run_id: str, retry_interrupted: bool = False
) -> dict:
    """What classifying the run now would involve, before paying for it.

    Returns:
        {"answers": answers still needed, "reusable": how many of those a
        saved answer covers (no model call), "calls": the rest, each at
        least one paid call (more when a fallback or confirmation model is
        used), "interrupted": calls left waiting after a crash, which are
        only retried on request}.
    """
    run = runs.find_one({"_id": run_id}) or {}
    variant_configs = configs(run) if run.get("variants") else {}
    answers = reusable = waiting = 0
    for doc in results.find({"run_id": run_id}):
        waiting += len(interrupted(doc))
        for variant in pending_variants(doc, retry_interrupted):
            answers += 1
            if (
                run.get("reuse_answers")
                and variant in variant_configs
                and _reusable(results, doc, answer_key(variant_configs[variant]))
            ):
                reusable += 1
    return {
        "answers": answers,
        "reusable": reusable,
        "calls": answers - reusable,
        "interrupted": waiting,
    }


def describe_estimate(estimate: dict) -> str:
    """The estimate as a sentence for a confirmation."""
    if not estimate["answers"]:
        return "Nothing is left to classify."
    text = f"{estimate['answers']} answer(s) to get"
    if estimate["reusable"]:
        text += f", {estimate['reusable']} of them copied from earlier tests at no cost"
    calls = estimate["calls"]
    text += (
        f": at least {calls} paid model call(s), more if a fallback or "
        "confirmation model is needed."
        if calls
        else ": no paid model calls."
    )
    return text


def record_preference(
    results: Collection,
    run_id: str,
    ticket_id: str,
    choice: Optional[str],
    expected_version: int,
    reviewer: str = "",
    note: str = "",
) -> int:
    """Saves a reviewer's verdict on one ticket; None clears it.

    Args:
        expected_version: The ticket's preference_version when the reviewer
            loaded it. The save is refused if the verdict changed since.

    Returns:
        The new preference_version.

    Raises:
        VerdictConflict: Someone else saved a verdict on it since it was loaded.
        RunFinalized: The run is finalized; its verdicts can't change.
        ValueError: An unknown choice, or the ticket isn't in the run.
    """
    if choice is not None and choice not in PREFERENCES:
        raise ValueError(
            f"choice must be one of {', '.join(PREFERENCES)}, got {choice!r}."
        )
    preference = (
        None
        if choice is None
        else {
            "choice": choice,
            "reviewer": reviewer,
            "note": note,
            "at": datetime.now(timezone.utc),
        }
    )
    # Results saved before versions existed have no preference_version: 0.
    version = expected_version if expected_version else {"$in": [0, None]}
    outcome = results.update_one(
        {
            "run_id": run_id,
            "ticket_id": str(ticket_id),
            "preference_version": version,
            "finalized": {"$ne": True},
        },
        {"$set": {"preference": preference}, "$inc": {"preference_version": 1}},
    )
    if outcome.matched_count == 0:
        current = results.find_one({"run_id": run_id, "ticket_id": str(ticket_id)})
        if current is None:
            raise ValueError(f"Ticket {ticket_id} isn't in A/B run {run_id}.")
        if current.get("finalized"):
            raise RunFinalized(
                f"A/B run {run_id} is finalized: its verdicts are saved on the tickets and can't change."
            )
        by = ((current.get("preference") or {}).get("reviewer")) or "someone else"
        raise VerdictConflict(
            f"The verdict on ticket {ticket_id} was changed by {by} after you opened it. "
            "Reload the run to see it before saving yours."
        )
    return expected_version + 1


def blind_choice(doc: dict, side: int) -> str:
    """The variant shown as "Output {side}" (1 or 2) for this ticket."""
    return doc["order"][side - 1]


def list_runs(runs: Collection, limit: int = 50) -> List[dict]:
    """The newest runs first."""
    return list(runs.find().sort("created_at", -1).limit(limit))


def answered(doc: dict) -> bool:
    """True when both variants classified the ticket without an error."""
    outputs = doc.get("outputs") or {}
    return all(v in outputs and "error" not in outputs[v] for v in VARIANTS)


def differs(doc: dict) -> bool:
    """True when both variants answered and chose different classifications."""
    return (
        answered(doc)
        and doc["outputs"]["A"]["classification"]
        != doc["outputs"]["B"]["classification"]
    )


def summarize(run: dict, docs: List[dict]) -> dict:
    """Preferences, agreement and cost for one run.

    Returns:
        {"run": run, "tickets": int, "classified": int (both variants done),
         "agreement": int (same classification), "variants": {variant:
         {"done", "errors", "classifications", "mean_confidence",
         "confirmed", "classification_only", "usage", "mean_latency_ms",
         "attempts" (model attempts this run made), "cost_usd" (every
         answer, reused ones at their original cost), "spent_usd" (every
         attempt this run paid for, failed ones included), "cost_unknown"
         and "spent_unknown" (how many of those had no known cost; the
         amounts are then subtotals of the known ones), "reused",
         "interrupted"}},
         "preferences":
         Counter, "reviewed": int, "disagreement_preferences": Counter}
    """
    variants = {}
    for v in VARIANTS:
        outputs = [d["outputs"][v] for d in docs if v in (d.get("outputs") or {})]
        good = [o for o in outputs if "error" not in o]
        # Failed answers took time too.
        latencies = [
            o["latency_ms"] for o in outputs if o.get("latency_ms") is not None
        ]
        attempts = [record for d in docs for record in attempt_records(d, v)]
        reused = [o for o in outputs if o.get("reused_from")]
        spent = [r.get("cost_usd") for r in attempts]
        reused_costs = [o.get("cost_usd") for o in reused]
        variants[v] = {
            "done": len(outputs),
            "errors": len(outputs) - len(good),
            "classifications": Counter(o["classification"] for o in good),
            "mean_confidence": (
                sum(o["confidence"] for o in good) / len(good) if good else None
            ),
            "confirmed": sum(1 for o in good if o.get("confirmed_from")),
            "classification_only": sum(1 for o in good if o.get("decision_only")),
            "usage": _sum_usage(
                [r.get("usage") for r in attempts] + [o.get("usage") for o in reused]
            ),
            "mean_latency_ms": sum(latencies) / len(latencies) if latencies else None,
            "attempts": len(attempts),
            "spent_usd": sum(c for c in spent if c is not None),
            "spent_unknown": sum(1 for c in spent if c is None),
            "cost_usd": sum(c for c in spent + reused_costs if c is not None),
            "cost_unknown": sum(1 for c in spent + reused_costs if c is None),
            "reused": len(reused),
            "interrupted": sum(1 for d in docs if v in interrupted(d)),
        }
    both = [d for d in docs if answered(d)]
    agree = [d for d in both if not differs(d)]
    reviewed = [d for d in docs if d.get("preference")]
    return {
        "run": run,
        "tickets": len(docs),
        "classified": len(both),
        "agreement": len(agree),
        "variants": variants,
        "preferences": Counter(d["preference"]["choice"] for d in reviewed),
        "reviewed": len(reviewed),
        "disagreement_preferences": Counter(
            d["preference"]["choice"] for d in reviewed if differs(d)
        ),
    }


def attempt_records(doc: dict, variant: str) -> List[dict]:
    """Every model attempt this result made for the variant (failed and fatal
    ones included); a reused answer made none. Results saved before
    attempt_history existed count their output as one attempt."""
    history = (doc.get("attempt_history") or {}).get(variant)
    if history is not None:
        return history
    output = (doc.get("outputs") or {}).get(variant)
    return [output] if output and not output.get("reused_from") else []


def _known_cost(amount: float, unknown: int) -> str:
    """A cost subtotal, marked when some costs are unknown."""
    if unknown and not amount:
        return f"unknown ({unknown})"
    return pricing.format_usd(amount) + (f" + {unknown} unknown" if unknown else "")


def _percent(part: int, whole: int) -> str:
    return f"{part / whole:.0%}" if whole else "-"


def format_summary(summary: dict) -> str:
    """The summary as text, with the variants named (not blind)."""
    run = summary["run"]
    cfg = configs(run)
    lines = [
        f"A/B run {run['_id']}: {run.get('name', '')}",
        f"  A: {cfg['A'].describe()}",
        f"  B: {cfg['B'].describe()}",
        "",
        f"Tickets: {summary['tickets']}; classified by both: {summary['classified']}",
        f"Same classification: {summary['agreement']} of {summary['classified']} "
        f"({_percent(summary['agreement'], summary['classified'])})",
        "",
        f"Reviewed: {summary['reviewed']} of {summary['tickets']}",
    ]
    for choice, label in PREFERENCES.items():
        count = summary["preferences"][choice]
        lines.append(
            f"  {label:<20} {count:>4}  {_percent(count, summary['reviewed'])}"
        )
    disagreed = sum(summary["disagreement_preferences"].values())
    if disagreed:
        lines.append(f"  Where they disagreed ({disagreed} reviewed):")
        for choice, label in PREFERENCES.items():
            count = summary["disagreement_preferences"][choice]
            lines.append(f"    {label:<18} {count:>4}  {_percent(count, disagreed)}")
    lines.append("")
    header = "".ljust(24) + "".join(v.rjust(14) for v in VARIANTS)
    lines.append(header)

    def row(label: str, value: Callable[[dict], str]) -> None:
        lines.append(
            label.ljust(24)
            + "".join(value(summary["variants"][v]).rjust(14) for v in VARIANTS)
        )

    row("Errors", lambda s: str(s["errors"]))
    row(
        "Mean confidence",
        lambda s: (
            f"{s['mean_confidence']:.0f}" if s["mean_confidence"] is not None else "-"
        ),
    )
    row("Confirmed", lambda s: str(s["confirmed"]))
    row("Classification only", lambda s: str(s["classification_only"]))
    row("Cost", lambda s: _known_cost(s["cost_usd"], s["cost_unknown"]))
    row(
        "Cost per answer",
        lambda s: (
            pricing.format_usd(s["cost_usd"] / s["done"])
            if s["done"] and not s["cost_unknown"]
            else "-"
        ),
    )
    row("Reused answers", lambda s: str(s["reused"]))
    row("Model attempts", lambda s: str(s["attempts"]))
    row("Spent in this test", lambda s: _known_cost(s["spent_usd"], s["spent_unknown"]))
    row("Interrupted calls", lambda s: str(s["interrupted"]))
    row(
        "Mean latency (s)",
        lambda s: (
            f"{s['mean_latency_ms'] / 1000:.1f}"
            if s["mean_latency_ms"] is not None
            else "-"
        ),
    )
    for key, label in (
        ("input_tokens", "Input tokens"),
        ("cache_read_input_tokens", "Cache-read tokens"),
        ("cache_creation_input_tokens", "Cache-write tokens"),
        ("output_tokens", "Output tokens"),
    ):
        row(label, lambda s, key=key: f"{s['usage'].get(key, 0):,}")
    for c in CLASSIFICATIONS:
        row(c, lambda s, c=c: str(s["classifications"][c]))
    return "\n".join(lines)


# --- finishing a run -------------------------------------------------------------


def ready_to_finalize(run: dict, docs: List[dict]) -> bool:
    """True when nothing is left to classify, no call is waiting after an
    interruption, and every ticket both variants answered has a verdict.
    Tickets a variant gave up on can't be judged; a run where every ticket
    gave up has nothing to judge and is ready too."""
    if run.get("status") in (FINALIZED, CREATING) or not docs:
        return False
    if any(pending_variants(d) or interrupted(d) for d in docs):
        return False
    return all(d.get("preference") for d in docs if answered(d))


def _verdict_entry(run: dict, doc: dict, finalized_at: datetime) -> dict:
    """What a ticket keeps of an A/B verdict: the run, both configurations
    and their answers, and the reviewer's choice."""
    preference = doc["preference"]
    choice = preference["choice"]
    return {
        "run_id": run["_id"],
        "run_name": run.get("name"),
        "choice": choice,
        "preferred_config": run["variants"][choice] if choice in VARIANTS else None,
        "reviewer": preference.get("reviewer"),
        "note": preference.get("note"),
        "decided_at": preference.get("at"),
        "finalized_at": finalized_at,
        "prompt_version": run.get("prompt_version"),
        "variants": {
            v: {
                "config": run["variants"][v],
                **{
                    key: doc["outputs"][v].get(key)
                    for key in (
                        "classification",
                        "confidence",
                        "model_version",
                        "decision_only",
                        "cost_usd",
                    )
                },
            }
            for v in VARIANTS
        },
    }


def finalize(
    tickets: Collection, runs: Collection, results: Collection, run_id: str
) -> dict:
    """Saves a judged run's verdicts on the tickets and removes the working data.

    Each verdict is added to its ticket's ab_test_verdicts (once per run, so
    finalizing again adds nothing). Then each result loses the ticket text,
    model input, raw replies and claim (_SLIMMED_FIELDS); its answers,
    verdict and fingerprint stay, so reports, exports and answer reuse keep
    working. The run is marked finalized with its summary, and its verdicts
    can't change after that. Safe to repeat if it was interrupted.

    Returns:
        {"tickets_updated": int, "tickets_missing": [ticket IDs]}.

    Raises:
        NotReadyToFinalize: something is left to classify or judge.
    """
    run = runs.find_one({"_id": run_id})
    if run is None:
        raise ValueError(f"No A/B run {run_id}.")
    docs = list(results.find({"run_id": run_id}))
    if run.get("status") != FINALIZED and not ready_to_finalize(run, docs):
        raise NotReadyToFinalize(
            f"A/B run {run_id} isn't finished: classify every ticket and give each answered "
            "ticket a verdict first."
        )
    finalized_at = run.get("finalized_at") or datetime.now(timezone.utc)
    # Lock the verdicts, then copy them as they are now: one saved between
    # the check above and the lock is copied too.
    results.update_many({"run_id": run_id}, {"$set": {"finalized": True}})
    docs = list(results.find({"run_id": run_id}))
    if run.get("status") != FINALIZED and not ready_to_finalize(run, docs):
        # A verdict was cleared just before the lock: unlock and refuse.
        results.update_many({"run_id": run_id}, {"$set": {"finalized": False}})
        raise NotReadyToFinalize(
            f"A/B run {run_id} isn't finished: a verdict was cleared while finalizing."
        )
    updated, missing = 0, []
    for doc in docs:
        if not doc.get("preference"):
            continue
        outcome = tickets.update_one(
            {
                "zendesk_id": doc["ticket_id"],
                f"{VERDICTS_FIELD}.run_id": {"$ne": run_id},
            },
            {"$push": {VERDICTS_FIELD: _verdict_entry(run, doc, finalized_at)}},
        )
        if outcome.modified_count:
            updated += 1
        elif tickets.count_documents({"zendesk_id": doc["ticket_id"]}, limit=1) == 0:
            missing.append(doc["ticket_id"])
    if missing:
        logger.warning(
            f"A/B {run_id}: {len(missing)} ticket(s) are no longer in the tickets collection; "
            "their verdicts stay in the A/B results only."
        )
    results.update_many(
        {"run_id": run_id}, {"$unset": {field: "" for field in _SLIMMED_FIELDS}}
    )
    # Raw replies in the attempt history can quote ticket text too.
    for doc in docs:
        history = doc.get("attempt_history") or {}
        if any("raw_response" in r for records in history.values() for r in records):
            results.update_one(
                {"_id": doc["_id"]},
                {
                    "$set": {
                        "attempt_history": {
                            v: [
                                {k: x for k, x in r.items() if k != "raw_response"}
                                for r in records
                            ]
                            for v, records in history.items()
                        }
                    }
                },
            )
    summary = summarize(run, list(results.find({"run_id": run_id})))
    runs.update_one(
        {"_id": run_id},
        {
            "$set": {
                "status": FINALIZED,
                "finalized_at": finalized_at,
                "summary": _plain_summary(summary),
            }
        },
    )
    return {"tickets_updated": updated, "tickets_missing": missing}


def _plain_summary(summary: dict) -> dict:
    """The summary without the run, with Counters as plain dicts (for storage and JSON)."""
    plain = {k: v for k, v in summary.items() if k != "run"}
    for key in ("preferences", "disagreement_preferences"):
        plain[key] = dict(plain[key])
    plain["variants"] = {
        v: {**stats, "classifications": dict(stats["classifications"])}
        for v, stats in summary["variants"].items()
    }
    return plain


# --- export --------------------------------------------------------------------------

# A cell starting like this is a formula to a spreadsheet; customer-written
# text gets a leading apostrophe (as bertopic_runs.spreadsheet_safe does).
_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")
_OUTPUT_COLUMNS = (
    "classification",
    "confidence",
    "decision_only",
    "model_version",
    "error",
    "attempts",
    "reused",
    "model_calls",
    "input_tokens",
    "output_tokens",
    "latency_ms",
    "cost_usd",
    "cost_basis",
)


def _cell(value) -> str:
    text = "" if value is None else str(value)
    return "'" + text if text.startswith(_FORMULA_PREFIXES) else text


def export_rows(run: dict, docs: List[dict]) -> List[dict]:
    """One row per ticket with both answers named (not blind) and the verdict."""
    rows = []
    for doc in sorted(docs, key=lambda d: d["ticket_id"]):
        preference = doc.get("preference") or {}
        row = {
            "ticket_id": doc["ticket_id"],
            "subject": (doc.get("ticket") or {}).get("subject"),
            "same_classification": answered(doc) and not differs(doc),
            "verdict": preference.get("choice"),
            "reviewer": preference.get("reviewer"),
            "note": preference.get("note"),
            "decided_at": preference.get("at"),
        }
        for v in VARIANTS:
            output = (doc.get("outputs") or {}).get(v) or {}
            usage = output.get("usage") or {}
            spent = [r.get("cost_usd") for r in attempt_records(doc, v)]
            row[f"{v}_spent_usd"] = sum(c for c in spent if c is not None)
            row[f"{v}_spent_unknown"] = sum(1 for c in spent if c is None)
            values = {
                **output,
                "reused": bool(output.get("reused_from")),
                "input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
            }
            for column in _OUTPUT_COLUMNS:
                row[f"{v}_{column}"] = values.get(column)
        rows.append(row)
    return rows


def export_run(run: dict, docs: List[dict], path) -> pathlib.Path:
    """Writes a run to path: .csv for one row per ticket, .json for everything
    kept (run, summary, and each result without its model input).

    The file is written beside path and then moved into place, so a failed
    export never leaves a half-written file.
    """
    path = pathlib.Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        rows = export_rows(run, docs)
        buffer = io.StringIO()
        writer = csv.DictWriter(
            buffer, fieldnames=list(rows[0]) if rows else ["ticket_id"]
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _cell(v) for k, v in row.items()})
        text = buffer.getvalue()
    elif suffix == ".json":
        text = json.dumps(
            {
                "run": run,
                "summary": _plain_summary(summarize(run, docs)),
                "results": [
                    {
                        k: v
                        for k, v in doc.items()
                        if k not in ("_id", "payload", "claim")
                    }
                    for doc in sorted(docs, key=lambda d: d["ticket_id"])
                ],
            },
            indent=2,
            default=str,
        )
    else:
        raise ValueError(f"Export to a .csv or .json file, not {path.name!r}.")
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


# --- command line --------------------------------------------------------------


def _config_args(parser: argparse.ArgumentParser, variant: str) -> None:
    lower = variant.lower()
    default = "the current engine" if variant == "A" else "A's"
    parser.add_argument(
        f"--{lower}-primary", help=f"{variant}'s primary model (default: {default})"
    )
    parser.add_argument(
        f"--{lower}-secondary", help=f"{variant}'s secondary model; '' for none"
    )
    parser.add_argument(
        f"--{lower}-confirmation", help=f"{variant}'s confirmation model; '' for none"
    )
    parser.add_argument(
        f"--{lower}-arbiter", help=f"{variant}'s arbiter model; '' for none"
    )
    parser.add_argument(f"--{lower}-max-tokens", type=int)
    parser.add_argument(
        f"--{lower}-effort",
        choices=(*EFFORT_LEVELS, "none"),
        help="'none' sends no effort (the model's default), even if A sets one",
    )


def _config_from(
    args: argparse.Namespace, variant: str, base: EngineConfig
) -> EngineConfig:
    values = base.to_dict()
    for field in (
        "primary",
        "secondary",
        "confirmation",
        "arbiter",
        "max_tokens",
        "effort",
    ):
        given = getattr(args, f"{variant.lower()}_{field}")
        if given is not None:
            values[field] = None if (field == "effort" and given == "none") else given
    return EngineConfig.from_dict(values)


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m decision_engine.ab_test")
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create", help="sample tickets and record a run")
    create.add_argument("--size", type=int, default=30)
    create.add_argument("--seed", type=int)
    create.add_argument("--name", default="")
    create.add_argument("--awaiting-only", action="store_true")
    create.add_argument(
        "--no-reuse",
        action="store_true",
        help="call the models even where a matching saved answer exists",
    )
    _config_args(create, "A")
    _config_args(create, "B")
    for name in ("run", "report", "finalize"):
        command = commands.add_parser(name)
        command.add_argument("run_id")
        if name == "run":
            command.add_argument(
                "--retry-interrupted",
                action="store_true",
                help="also retry calls a crashed worker left unfinished (they may "
                "already have been charged)",
            )
    export = commands.add_parser("export", help="write a run to a .csv or .json file")
    export.add_argument("run_id")
    export.add_argument("path")
    args = parser.parse_args(argv)

    from .persistence import get_database, get_tickets_collection

    runs, results = get_collections(get_database())
    tickets = get_tickets_collection()
    if args.command == "create":
        a = _config_from(args, "A", EngineConfig.from_env())
        b = _config_from(args, "B", a)
        ids = sample_ticket_ids(tickets, args.size, args.seed, args.awaiting_only)
        from .classify import get_client

        prices = pricing.live_prices(get_client(), models_of(a, b))
        for model in models_of(a, b):
            print(
                f"  {model}: "
                + (
                    f"${prices[model].input}/M in, ${prices[model].output}/M out"
                    if model in prices
                    else "no listed price (cost only from billed charges)"
                )
            )
        run_id = create_run(
            tickets,
            runs,
            results,
            a,
            b,
            ids,
            args.name,
            seed=args.seed,
            reuse_answers=not args.no_reuse,
            prices=prices,
        )
        print(f"Created A/B run {run_id} with {len(ids)} tickets.")
        print(f"  A: {a.describe()}\n  B: {b.describe()}")
        print(f"Classify them with: python -m decision_engine.ab_test run {run_id}")
        return 0
    if args.command == "run":
        estimate = estimate_calls(runs, results, args.run_id, args.retry_interrupted)
        print(describe_estimate(estimate))
        if estimate["interrupted"] and not args.retry_interrupted:
            print(
                f"{estimate['interrupted']} call(s) were interrupted and may already have "
                "been charged; add --retry-interrupted to retry them."
            )
        summary = run_ab_test(
            args.run_id,
            runs,
            results,
            on_progress=lambda done, total: print(f"{done}/{total}", end="\r"),
            retry_interrupted=args.retry_interrupted,
        )
        print(format_summary(summary))
        return 0
    run = runs.find_one({"_id": args.run_id})
    if run is None:
        print(f"No A/B run {args.run_id}.")
        return 1
    docs = list(results.find({"run_id": args.run_id}))
    if args.command == "finalize":
        done = finalize(tickets, runs, results, args.run_id)
        print(
            f"Saved verdicts on {done['tickets_updated']} ticket(s); removed the run's working data."
        )
        return 0
    if args.command == "export":
        print(f"Wrote {export_run(run, docs, args.path)}")
        return 0
    print(format_summary(summarize(run, docs)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
