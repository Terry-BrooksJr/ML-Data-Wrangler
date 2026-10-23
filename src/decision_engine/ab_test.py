"""A/B tests: the same tickets classified by two engine configurations.

A run samples tickets, classifies each one with configuration A (usually
the current engine) and configuration B (the one being considered), and
stores both answers in ab_results. A reviewer then reads the two answers
side by side, blind (each ticket shows them in a random order, as "Output
1" and "Output 2"), and records which is better. The summary reports the
preferences, how often the two agree, and what each cost in tokens.

Nothing here writes to the tickets collection: an A/B run never adds an
agent review, claims a ticket or changes awaiting_review. Each result keeps
a snapshot of its ticket taken when the run was created, so both variants,
and the reviewer, see the same ticket however it changes afterwards.

    python -m decision_engine.ab_test create --size 30 --b-primary claude-opus-5.5
    python -m decision_engine.ab_test run RUN_ID      # paid model calls
    python -m decision_engine.ab_test report RUN_ID
"""

import argparse
import os
import random
import socket
import sys
import uuid
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from loguru import logger
from pymongo import ReturnDocument
from pymongo.collection import Collection

from .classify import classify_ticket, EFFORT_LEVELS, EngineConfig, is_fatal
from .guards import CLASSIFICATIONS
from .payload import build_ticket_payload
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
    "reasoning",
    "internal_note",
    "draft_message",
    "leadership_recipient",
    "tag",
    "guard_overrides",
    "model_version",
)
# The ticket fields kept for the reviewer; the model gets the saved payload.
_SNAPSHOT_FIELDS = ("zendesk_id", "subject", "status", "assignee", "transcript")
# A claim older than this is treated as abandoned (its worker crashed). Long
# enough for both variants of one ticket, confirmation calls included.
CLAIM_TTL = timedelta(minutes=30)
# Run statuses. "creating" is only ever seen if creation was interrupted.
CREATING, PENDING, RUNNING, PARTIAL, COMPLETE = (
    "creating",
    "pending",
    "running",
    "partial",
    "complete",
)


class Stopped(Exception):
    """The run was asked to stop; what finished is saved and it can resume."""


class PromptChanged(ValueError):
    """The system prompt changed after the run started classifying."""


class VerdictConflict(ValueError):
    """Someone else saved a verdict on the ticket since it was loaded."""


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


def _snapshot(ticket: dict, as_of: datetime) -> dict:
    """What a result keeps of its ticket: the exact payload both variants
    are sent, and the fields the reviewer reads."""
    return {
        "payload": build_ticket_payload(ticket, as_of=as_of),
        "reengage_count": len(ticket.get("reengage_attempts") or []),
        "assignee": ticket.get("assignee"),
        "ticket": {key: ticket.get(key) for key in _SNAPSHOT_FIELDS},
    }


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
) -> str:
    """Records a new run and one pending result per ticket; classifies nothing.

    Each result keeps a snapshot of its ticket, including the payload built
    once (as of the run's creation), so A and B are sent exactly the same
    input and the reviewer reads what they were sent, even if the ticket
    changes later. Each ticket gets a random display order, so a reviewer
    can't tell which side is A.

    If saving the results fails part way, the run and its results are
    removed, so no partial sample is left behind.

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
    runs.insert_one(
        {
            "_id": run_id,
            "name": name or f"{a.primary} vs {b.primary}",
            "created_at": created_at,
            "created_by": created_by,
            "variants": {"A": a.to_dict(), "B": b.to_dict()},
            "prompt_version": PROMPT_VERSION,
            "ticket_count": len(ticket_ids),
            "status": CREATING,
        }
    )
    try:
        results.insert_many(
            [
                {
                    "run_id": run_id,
                    "ticket_id": ticket_id,
                    "order": shuffle.sample(VARIANTS, 2),
                    "outputs": {},
                    "preference": None,
                    "preference_version": 0,
                    "claim": None,
                    **_snapshot(found[ticket_id][0], created_at),
                }
                for ticket_id in ticket_ids
            ]
        )
        runs.update_one({"_id": run_id}, {"$set": {"status": PENDING}})
    except BaseException:
        results.delete_many({"run_id": run_id})
        runs.delete_one({"_id": run_id})
        raise
    return run_id


def configs(run: dict) -> Dict[str, EngineConfig]:
    return {v: EngineConfig.from_dict(run["variants"][v]) for v in VARIANTS}


def _cost(calls: List[dict]) -> dict:
    """Tokens and time across every model call one classification made."""
    return {
        "usage": _sum_usage(call.get("usage") for call in calls),
        "latency_ms": sum(call.get("latency_ms") or 0 for call in calls),
        "model_calls": len(calls),
    }


def _output(result: dict) -> dict:
    """What an A/B result keeps of a classification.

    Usage and latency cover every model call behind the answer: a failed
    primary call, the secondary that replaced it, and a confirmation.
    """
    kept = {key: result.get(key) for key in _KEPT_FIELDS}
    audit = result.get("audit") or {}
    calls = result.get("model_calls") or [
        {"usage": audit.get("usage"), "latency_ms": audit.get("latency_ms")}
    ]
    kept.update(_cost(calls))
    confirmed = result.get("confirmation_of")
    if confirmed:
        kept["confirmed_from"] = {
            key: confirmed.get(key)
            for key in ("model_version", "classification", "confidence")
        }
    return kept


def _error_output(error: Exception) -> dict:
    audit = getattr(error, "audit", None) or {}
    calls = getattr(error, "model_calls", None) or [
        {"usage": audit.get("usage"), "latency_ms": audit.get("latency_ms")}
    ]
    return {"error": f"{type(error).__name__}: {error}"[:1000], **_cost(calls)}


def _sum_usage(usages: Iterable[dict]) -> dict:
    total: Counter = Counter()
    for usage in usages:
        for key, value in (usage or {}).items():
            total[key] += value or 0
    return dict(total)


def pending_variants(doc: dict) -> List[str]:
    """The variants still to classify: never tried, or tried and failed."""
    outputs = doc.get("outputs") or {}
    return [v for v in VARIANTS if v not in outputs or "error" in outputs[v]]


def _claim(results: Collection, doc_id, worker: str) -> Optional[dict]:
    """Takes a result for this worker, so two workers never pay for the same
    ticket; returns the claimed document, or None if another worker has it."""
    now = datetime.now(timezone.utc)
    return results.find_one_and_update(
        {
            "_id": doc_id,
            "$or": [
                {"claim": None},
                {"claim.at": {"$lt": now - CLAIM_TTL}},
                {"claim.worker": worker},
            ],
        },
        {"$set": {"claim": {"worker": worker, "at": now}}},
        return_document=ReturnDocument.AFTER,
    )


def _finish(runs: Collection, results: Collection, run_id: str) -> str:
    """Records the run's status from what is saved: complete only when every
    ticket has both answers."""
    unfinished = sum(
        1 for doc in results.find({"run_id": run_id}) if pending_variants(doc)
    )
    status = PARTIAL if unfinished else COMPLETE
    update = {"status": status}
    if status == COMPLETE:
        update["completed_at"] = datetime.now(timezone.utc)
    runs.update_one({"_id": run_id}, {"$set": update})
    return status


def run_ab_test(
    run_id: str,
    runs: Collection,
    results: Collection,
    classify: Callable[..., dict] = classify_ticket,
    on_progress: Optional[Callable[[int, int], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    worker: Optional[str] = None,
) -> dict:
    """Classifies every pending ticket of a run with both configurations.

    Each ticket's saved payload is sent to both, so they see the same input.
    Resumable: an output is saved as soon as it arrives and isn't asked for
    again, but a failed one is retried by the next run. Errors that will
    hit every ticket (bad key, out of credit) stop the run, as in a review
    batch. Each ticket is claimed while it is classified, so two workers on
    one run never classify the same ticket. However the call ends, the run
    is left "complete" or "partial", never "running".

    Args:
        on_progress: Called with (done, total) after each ticket.
        should_stop: Checked between tickets; True stops the run (Stopped).
        worker: Identifies this worker's claims; defaults to a fresh ID.

    Returns:
        The run's summary (summarize).

    Raises:
        ValueError: No such run, or its creation didn't finish.
        PromptChanged: Answers were saved under a different system prompt.
    """
    run = runs.find_one({"_id": run_id})
    if run is None:
        raise ValueError(f"No A/B run {run_id}.")
    if run.get("status") == CREATING:
        raise ValueError(
            f"A/B run {run_id} wasn't created completely; create a new run."
        )
    docs = list(results.find({"run_id": run_id}).sort("ticket_id", 1))
    if run.get("prompt_version") != PROMPT_VERSION:
        if any(doc.get("outputs") for doc in docs):
            raise PromptChanged(
                f"A/B run {run_id} was classified with prompt {run.get('prompt_version')}, but the "
                f"engine now uses {PROMPT_VERSION}. Answers from two prompts can't be compared; "
                "create a new run."
            )
        runs.update_one({"_id": run_id}, {"$set": {"prompt_version": PROMPT_VERSION}})
    variant_configs = configs(run)
    worker = worker or f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"
    runs.update_one({"_id": run_id}, {"$set": {"status": RUNNING}})
    try:
        for done, doc in enumerate(docs, start=1):
            if should_stop and should_stop():
                raise Stopped(f"Stopped after {done - 1} of {len(docs)} tickets.")
            if pending_variants(doc):
                claimed = _claim(results, doc["_id"], worker)
                if claimed is None:
                    logger.info(
                        f"A/B {run_id}: ticket {doc['ticket_id']} is being classified by another worker."
                    )
                else:
                    try:
                        _classify_pending(
                            claimed, run_id, variant_configs, results, classify, worker
                        )
                    finally:
                        results.update_one(
                            {"_id": doc["_id"], "claim.worker": worker},
                            {"$set": {"claim": None}},
                        )
            if on_progress:
                on_progress(done, len(docs))
    finally:
        _finish(runs, results, run_id)
    return summarize(
        runs.find_one({"_id": run_id}), list(results.find({"run_id": run_id}))
    )


def _classify_pending(
    doc: dict,
    run_id: str,
    variant_configs: Dict[str, EngineConfig],
    results: Collection,
    classify: Callable[..., dict],
    worker: str,
) -> None:
    """Classifies a claimed ticket's pending variants and saves each output."""
    for variant in pending_variants(doc):
        try:
            output = _output(
                classify(
                    doc["payload"],
                    reengage_count=doc.get("reengage_count", 0),
                    assignee=doc.get("assignee"),
                    config=variant_configs[variant],
                )
            )
        except Exception as e:
            if is_fatal(e):
                raise
            logger.warning(
                f"A/B {run_id}: ticket {doc['ticket_id']} failed on {variant}: {e}"
            )
            output = _error_output(e)
        saved = results.update_one(
            {"_id": doc["_id"], "claim.worker": worker},
            {"$set": {f"outputs.{variant}": output}},
        )
        if saved.matched_count == 0:
            # The claim expired and another worker took the ticket over.
            logger.warning(
                f"A/B {run_id}: lost the claim on ticket {doc['ticket_id']}; its output wasn't saved."
            )
            return


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
        {"run_id": run_id, "ticket_id": str(ticket_id), "preference_version": version},
        {"$set": {"preference": preference}, "$inc": {"preference_version": 1}},
    )
    if outcome.matched_count == 0:
        current = results.find_one({"run_id": run_id, "ticket_id": str(ticket_id)})
        if current is None:
            raise ValueError(f"Ticket {ticket_id} isn't in A/B run {run_id}.")
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
         "confirmed", "usage", "mean_latency_ms"}}, "preferences":
         Counter, "reviewed": int, "disagreement_preferences": Counter}
    """
    variants = {}
    for v in VARIANTS:
        outputs = [d["outputs"][v] for d in docs if v in (d.get("outputs") or {})]
        good = [o for o in outputs if "error" not in o]
        latencies = [o["latency_ms"] for o in good if o.get("latency_ms") is not None]
        variants[v] = {
            "done": len(outputs),
            "errors": len(outputs) - len(good),
            "classifications": Counter(o["classification"] for o in good),
            "mean_confidence": (
                sum(o["confidence"] for o in good) / len(good) if good else None
            ),
            "confirmed": sum(1 for o in good if o.get("confirmed_from")),
            "usage": _sum_usage(o.get("usage") for o in outputs),
            "mean_latency_ms": sum(latencies) / len(latencies) if latencies else None,
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
    parser.add_argument(f"--{lower}-max-tokens", type=int)
    parser.add_argument(f"--{lower}-effort", choices=EFFORT_LEVELS)


def _config_from(
    args: argparse.Namespace, variant: str, base: EngineConfig
) -> EngineConfig:
    values = base.to_dict()
    for field in ("primary", "secondary", "confirmation", "max_tokens", "effort"):
        given = getattr(args, f"{variant.lower()}_{field}")
        if given is not None:
            values[field] = given
    return EngineConfig.from_dict(values)


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m decision_engine.ab_test")
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create", help="sample tickets and record a run")
    create.add_argument("--size", type=int, default=30)
    create.add_argument("--seed", type=int)
    create.add_argument("--name", default="")
    create.add_argument("--awaiting-only", action="store_true")
    _config_args(create, "A")
    _config_args(create, "B")
    for name in ("run", "report"):
        command = commands.add_parser(name)
        command.add_argument("run_id")
    args = parser.parse_args(argv)

    from .persistence import get_database, get_tickets_collection

    runs, results = get_collections(get_database())
    tickets = get_tickets_collection()
    if args.command == "create":
        a = _config_from(args, "A", EngineConfig.from_env())
        b = _config_from(args, "B", a)
        ids = sample_ticket_ids(tickets, args.size, args.seed, args.awaiting_only)
        run_id = create_run(
            tickets, runs, results, a, b, ids, args.name, seed=args.seed
        )
        print(f"Created A/B run {run_id} with {len(ids)} tickets.")
        print(f"  A: {a.describe()}\n  B: {b.describe()}")
        print(f"Classify them with: python -m decision_engine.ab_test run {run_id}")
        return 0
    if args.command == "run":
        summary = run_ab_test(
            args.run_id,
            runs,
            results,
            on_progress=lambda done, total: print(f"{done}/{total}", end="\r"),
        )
        print(format_summary(summary))
        return 0
    run = runs.find_one({"_id": args.run_id})
    if run is None:
        print(f"No A/B run {args.run_id}.")
        return 1
    print(format_summary(summarize(run, list(results.find({"run_id": args.run_id})))))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
