import os
import socket
import uuid
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Callable, Iterable, List, Optional

from bson import ObjectId
from loguru import logger
from pymongo import MongoClient, ReturnDocument
from pymongo.collection import Collection
from pymongo.database import Database

from .classify import classify_ticket, EngineError, get_client, is_fatal
from .payload import build_ticket_payload

# An engine error (cut-off or invalid reply, API error) leaves the ticket in
# the queue; after this many failed attempts it is routed to manual triage.
MAX_ENGINE_ATTEMPTS = 3
# A claim older than this is treated as abandoned (its worker crashed).
CLAIM_TTL = timedelta(minutes=15)


@lru_cache(maxsize=1)
def get_database() -> Database:
    """Connects on first use so importing this module needs no Mongo access.

    The database defaults to support_db (MONGO_DB) until the real name is
    confirmed.
    """
    mongo = MongoClient(os.environ["MONGO_URI"])
    return mongo[os.environ.get("MONGO_DB", "support_db")]


def reset_clients() -> None:
    """Drops the cached Mongo database and model client.

    Call after changing MONGO_URI, MONGO_DB or the Cheaper Inference settings;
    the next get_database() / get_client() builds them from the new values.
    Jobs already running keep the clients they hold.
    """
    get_database.cache_clear()
    get_client.cache_clear()


def get_tickets_collection() -> Collection:
    return get_database()[os.environ.get("MONGO_COLLECTION", "tickets")]


def default_worker_id() -> str:
    return f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"


def claim_ticket(tickets: Collection, zendesk_id: str, worker: str) -> Optional[dict]:
    """Marks an awaiting ticket as being reviewed by `worker`.

    Two workers can't both claim a ticket, so only one pays for the model
    call. A claim past its expiry can be taken over.

    Returns:
        The claimed document, or None if it is missing, not awaiting review,
        or claimed by another worker.
    """
    now = datetime.now(timezone.utc)
    return tickets.find_one_and_update(
        {
            "zendesk_id": zendesk_id,
            "awaiting_review": True,
            "$or": [
                {"review_claim": {"$exists": False}},
                {"review_claim": None},
                {"review_claim.expires_at": {"$lt": now}},
            ],
        },
        {
            "$set": {
                "review_claim": {
                    "worker": worker,
                    "claimed_at": now,
                    "expires_at": now + CLAIM_TTL,
                }
            }
        },
        return_document=ReturnDocument.AFTER,
    )


def _release_claim(
    tickets: Collection, zendesk_id: str, worker: str, update: Optional[dict] = None
) -> None:
    update = update or {}
    update.setdefault("$unset", {})["review_claim"] = ""
    tickets.update_one(
        {"zendesk_id": zendesk_id, "review_claim.worker": worker}, update
    )


def _review_entry(result: dict, engine_error: Optional[str] = None) -> dict:
    return {
        "review_id": ObjectId(),  # outcomes[].review_id references this
        "reviewed_at": datetime.now(timezone.utc),
        "model_version": result["model_version"],
        "classification": result["classification"],
        "confidence": result["confidence"],
        "signals_matched": result.get("signals_matched", []),
        "reasoning": result.get("reasoning", ""),
        "tag_applied": result.get("tag"),
        "internal_note": result.get("internal_note", ""),
        "draft_message": result.get("draft_message"),
        "leadership_recipient": result.get("leadership_recipient"),
        # Set when the confirmation model replaced the first answer.
        "confirmation_of": result.get("confirmation_of"),
        # Set when a code check overrode the model (guards.apply_guards).
        "guard_overrides": result.get("guard_overrides", []),
        # Set when the engine gave up after repeated failures; distinguishes
        # this from a deliberate manual_triage abstention.
        "engine_error": engine_error,
        "audit": result.get("audit"),
    }


def _manual_triage_fallback(ticket_doc: dict, error: Exception, attempts: int) -> dict:
    return {
        "ticket_id": ticket_doc["zendesk_id"],
        "classification": "manual_triage",
        "confidence": 0,
        "signals_matched": [],
        "reasoning": f"Engine failed {attempts} times; last error: {error}",
        "tag": "manual_triage",
        "internal_note": "Auto-routed: the decision engine could not classify this ticket.",
        "draft_message": None,
        "leadership_recipient": None,
        "model_version": getattr(error, "model", None) or "none",
        "audit": getattr(error, "audit", None),
    }


def process_one_ticket(
    zendesk_id: str, tickets: Collection = None, worker: Optional[str] = None
):
    """
    Claims one awaiting-review ticket, classifies it, and appends the result
    to agent_reviews. Does NOT call Zendesk — tagging/noting the real ticket
    is Module 3 (write-back), kept strictly separate so dry-run mode stays
    meaningful and the read-only/write-back boundary isn't blurred.

    An engine failure leaves the ticket awaiting review, with `review_attempts`
    and `last_engine_error` recorded, and re-raises. On the
    MAX_ENGINE_ATTEMPTS-th failure a manual_triage review marked with
    `engine_error` is stored instead, and returned. Errors that will hit
    every ticket (401/402) don't count as an attempt.

    The review is saved only if the ticket's assignee is still the one it was
    classified for: a leadership escalation's recipients follow the assignee.
    If ownership changed during the model call, the review is discarded and
    the ticket stays awaiting review, to be classified again for its new owner.

    Returns:
        The result dict, or None if the ticket is missing, not awaiting
        review, claimed by another worker, or reassigned while it was classified.
    """
    tickets = tickets if tickets is not None else get_tickets_collection()
    worker = worker or default_worker_id()
    ticket_doc = claim_ticket(tickets, zendesk_id, worker)
    if not ticket_doc:
        return None

    try:
        payload = build_ticket_payload(ticket_doc)
        result = classify_ticket(
            payload,
            reengage_count=len(ticket_doc.get("reengage_attempts") or []),
            assignee=ticket_doc.get("assignee"),
        )
    except Exception as e:
        if is_fatal(e):
            _release_claim(tickets, zendesk_id, worker)
            raise
        attempts = (ticket_doc.get("review_attempts") or 0) + 1
        error_record = {
            "at": datetime.now(timezone.utc),
            "type": type(e).__name__,
            "message": str(e)[:1000],
            "model": getattr(e, "model", None),
            "audit": getattr(e, "audit", None),
        }
        if attempts < MAX_ENGINE_ATTEMPTS:
            logger.warning(
                f"{zendesk_id}: engine error (attempt {attempts}/{MAX_ENGINE_ATTEMPTS}): {e}"
            )
            _release_claim(
                tickets,
                zendesk_id,
                worker,
                {
                    "$set": {
                        "review_attempts": attempts,
                        "last_engine_error": error_record,
                    }
                },
            )
            raise
        logger.error(
            f"{zendesk_id}: engine failed {attempts} times; routing to manual triage."
        )
        result = _manual_triage_fallback(ticket_doc, e, attempts)
        _save_review(
            tickets,
            zendesk_id,
            worker,
            _review_entry(result, engine_error=str(e)[:1000]),
            {"review_attempts": attempts, "last_engine_error": error_record},
        )
        result["engine_error"] = str(e)
        return result

    result["ticket_id"] = ticket_doc["zendesk_id"]  # don't trust the model's echo
    saved = _save_review(
        tickets,
        zendesk_id,
        worker,
        _review_entry(result),
        assignee=ticket_doc.get("assignee"),
    )
    if saved == REASSIGNED:
        return None
    return result


SAVED, CLAIM_LOST, REASSIGNED = "saved", "claim_lost", "reassigned"
_NOT_GIVEN = object()


def _save_review(
    tickets: Collection,
    zendesk_id: str,
    worker: str,
    entry: dict,
    extra_set: Optional[dict] = None,
    assignee=_NOT_GIVEN,
) -> str:
    """Appends entry to agent_reviews and clears awaiting_review.

    Matching on the claim means a worker whose claim expired and was taken
    over can't append a second entry. With assignee, the ticket must still
    have that assignee (None: none); if Zendesk reassigned it meanwhile, the
    review is discarded, the claim released, and the ticket stays awaiting
    review.

    Returns:
        SAVED, CLAIM_LOST or REASSIGNED.
    """
    match = {
        "zendesk_id": zendesk_id,
        "awaiting_review": True,
        "review_claim.worker": worker,
    }
    if assignee is not _NOT_GIVEN:
        match["assignee"] = assignee
    update = {
        "$push": {"agent_reviews": entry},
        "$set": {"awaiting_review": False, **(extra_set or {})},
        "$unset": {"review_claim": ""},
    }
    if not extra_set:
        update["$unset"].update({"review_attempts": "", "last_engine_error": ""})
    saved = tickets.update_one(match, update)
    if saved.modified_count:
        return SAVED
    if assignee is not _NOT_GIVEN and tickets.find_one(
        {"zendesk_id": zendesk_id, "review_claim.worker": worker}
    ):
        # Still ours, so the assignee is what changed.
        _release_claim(tickets, zendesk_id, worker)
        logger.warning(
            f"{zendesk_id}: reassigned while it was being classified; review discarded, "
            "left awaiting review for the new owner."
        )
        return REASSIGNED
    logger.warning(f"{zendesk_id}: claim lost before saving; review discarded.")
    return CLAIM_LOST


def _add_usage(totals: dict, audit: Optional[dict]) -> None:
    for key, value in ((audit or {}).get("usage") or {}).items():
        totals[key] = totals.get(key, 0) + (value or 0)


def _oldest_first(zendesk_id) -> tuple:
    """Sort key: Zendesk IDs are sequential, so a lower number is an older ticket."""
    text = str(zendesk_id)
    return (0, int(text), "") if text.isdigit() else (1, 0, text)


def awaiting_tickets(tickets: Collection) -> List[dict]:
    """Every ticket awaiting review, oldest first, as {"zendesk_id", "subject"}."""
    docs = tickets.find(
        {"awaiting_review": True}, {"_id": 0, "zendesk_id": 1, "subject": 1}
    )
    return sorted(docs, key=lambda doc: _oldest_first(doc["zendesk_id"]))


def awaiting_ids(tickets: Collection) -> List[str]:
    """Zendesk IDs of every ticket currently awaiting review, oldest first."""
    return [doc["zendesk_id"] for doc in awaiting_tickets(tickets)]


def run_review_batch(
    tickets: Collection = None,
    zendesk_ids: Optional[Iterable[str]] = None,
    worker: Optional[str] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    on_progress: Optional[Callable[[dict], None]] = None,
) -> dict:
    """Reviews awaiting tickets, continuing past per-ticket failures.

    Stops early on errors that will hit every ticket (401 bad key, 402 out
    of credit), or when `should_stop` returns True. The ticket in progress
    always finishes first, so its claim is saved or released cleanly.

    Args:
        zendesk_ids: The tickets to review. Pass the IDs the user confirmed,
            so tickets inserted or reopened after confirmation aren't billed
            without approval. Any that are no longer awaiting review are
            skipped. Defaults to every ticket awaiting review now.
        should_stop: Checked before each ticket; True cancels the rest.
        on_progress: Called after each ticket (on the batch's thread) with
            {"zendesk_id", "outcome", "review", "done", "total", "summary"}:
            outcome is classified, manual_fallback, skipped or errored;
            review is the stored review (None unless one was stored); done
            counts tickets finished so far, of total; summary is the running
            counts. Not called for a fatal stop or a cancel.

    Returns:
        Counts: classified, manual_fallback (gave up after repeated engine
        errors), skipped (claimed elsewhere or already reviewed), errored,
        cancelled (not started because should_stop fired), plus `stopped`
        (the fatal error message, or None) and token `usage`.
    """
    tickets = tickets if tickets is not None else get_tickets_collection()
    worker = worker or default_worker_id()
    summary = {
        "classified": 0,
        "manual_fallback": 0,
        "skipped": 0,
        "errored": 0,
        "cancelled": 0,
        "stopped": None,
        "usage": {},
    }
    # dict.fromkeys de-duplicates while keeping order, so no ID is billed twice.
    zendesk_ids = list(
        dict.fromkeys(zendesk_ids if zendesk_ids is not None else awaiting_ids(tickets))
    )
    def report(zendesk_id: str, outcome: str, review: Optional[dict], done: int) -> None:
        if on_progress is None:
            return
        try:
            on_progress(
                {
                    "zendesk_id": zendesk_id,
                    "outcome": outcome,
                    "review": review,
                    "done": done,
                    "total": len(zendesk_ids),
                    "summary": dict(summary),
                }
            )
        except Exception as e:  # a display problem must never stop a paid batch
            logger.warning(f"Batch progress callback failed: {e}")

    for index, zendesk_id in enumerate(zendesk_ids):
        if should_stop and should_stop():
            summary["cancelled"] = len(zendesk_ids) - index
            logger.warning(
                f"Batch cancelled; {summary['cancelled']} ticket(s) not started."
            )
            break
        try:
            result = process_one_ticket(zendesk_id, tickets, worker)
        except Exception as e:
            if is_fatal(e):
                summary["stopped"] = f"{type(e).__name__}: {e}"
                logger.error(
                    f"{zendesk_id}: {e} — stopping the batch; every ticket would fail the same way."
                )
                break
            summary["errored"] += 1
            logger.error(f"{zendesk_id}: engine error, left in the queue: {e}")
            report(zendesk_id, "errored", None, index + 1)
            continue
        if result is None:
            summary["skipped"] += 1
            logger.info(f"{zendesk_id}: already reviewed or claimed elsewhere, skipped")
            report(zendesk_id, "skipped", None, index + 1)
            continue
        _add_usage(summary["usage"], result.get("audit"))
        _add_usage(summary["usage"], (result.get("confirmation_of") or {}).get("audit"))
        outcome = "manual_fallback" if result.get("engine_error") else "classified"
        summary[outcome] += 1
        logger.info(
            f"{zendesk_id}: {result['classification']} ({result['confidence']})"
        )
        report(zendesk_id, outcome, result, index + 1)

    logger.info(
        f"Batch finished: {summary['classified']} classified, {summary['manual_fallback']} routed to manual "
        f"triage after engine errors, {summary['skipped']} skipped, {summary['errored']} errored, "
        f"{summary['cancelled']} cancelled. "
        f"Tokens: {summary['usage'] or 'none'}"
    )
    return summary


def format_batch_summary(summary: dict) -> str:
    lines = [
        f"Classified: {summary['classified']}",
        f"Routed to manual triage after repeated engine errors: {summary['manual_fallback']}",
        f"Skipped (already reviewed or claimed elsewhere): {summary['skipped']}",
        f"Engine errors (left in the queue to retry): {summary['errored']}",
    ]
    if summary.get("cancelled"):
        lines.append(f"Cancelled before starting: {summary['cancelled']}")
    usage = summary.get("usage") or {}
    if usage:
        lines.append(
            f"Tokens: {usage.get('input_tokens', 0)} in, {usage.get('output_tokens', 0)} out, "
            f"{usage.get('cache_read_input_tokens', 0)} cache read"
        )
    if summary.get("stopped"):
        lines.append(f"\nStopped early: {summary['stopped']}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(format_batch_summary(run_review_batch()))
