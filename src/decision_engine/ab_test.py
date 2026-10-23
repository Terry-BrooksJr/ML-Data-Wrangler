"""A/B tests: the same tickets classified by two engine configurations.

A run samples tickets, classifies each one with configuration A (usually
the current engine) and configuration B (the one being considered), and
stores both answers in ab_results. A reviewer then reads the two answers
side by side, blind (each ticket shows them in a random order, as "Output
1" and "Output 2"), and records which is better. The summary reports the
preferences, how often the two agree, and what each cost in tokens.

Nothing here writes to the tickets collection: an A/B run never adds an
agent review, claims a ticket or changes awaiting_review.

    python -m decision_engine.ab_test create --size 30 --b-primary claude-opus-5.5
    python -m decision_engine.ab_test run RUN_ID      # paid model calls
    python -m decision_engine.ab_test report RUN_ID
"""

import argparse
import os
import random
import sys
import uuid
from collections import Counter
from datetime import datetime, timezone
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from loguru import logger
from pymongo.collection import Collection

from .classify import classify_ticket, EFFORT_LEVELS, EngineConfig
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


class Stopped(Exception):
    """The run was asked to stop; what finished is saved and it can resume."""


def get_collections(database) -> Tuple[Collection, Collection]:
    """(ab_tests, ab_results) in the given database.

    The names are read here, not at import, so a saved setting applies at once.
    """
    runs = database[os.environ.get("AB_TEST_COLLECTION", "ab_tests")]
    results = database[os.environ.get("AB_RESULT_COLLECTION", "ab_results")]
    results.create_index([("run_id", 1), ("ticket_id", 1)], unique=True)
    return runs, results


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
    """
    query = {"transcript.0": {"$exists": True}}
    if awaiting_only:
        query["awaiting_review"] = True
    ids = sorted(
        str(doc["zendesk_id"]) for doc in tickets.find(query, {"zendesk_id": 1})
    )
    return random.Random(seed).sample(ids, min(size, len(ids)))


def create_run(
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

    Each ticket gets a random display order, so a reviewer can't tell which
    side is A.

    Returns:
        The run's ID.

    Raises:
        ValueError: A and B are the same configuration, or there are no tickets.
    """
    if a == b:
        raise ValueError("A and B are the same configuration; change one of them.")
    ticket_ids = list(dict.fromkeys(str(t) for t in ticket_ids))
    if not ticket_ids:
        raise ValueError("No tickets to test.")
    run_id = uuid.uuid4().hex[:12]
    shuffle = random.Random(seed)
    runs.insert_one(
        {
            "_id": run_id,
            "name": name or f"{a.primary} vs {b.primary}",
            "created_at": datetime.now(timezone.utc),
            "created_by": created_by,
            "variants": {"A": a.to_dict(), "B": b.to_dict()},
            "prompt_version": PROMPT_VERSION,
            "ticket_count": len(ticket_ids),
            "status": "pending",
        }
    )
    results.insert_many(
        [
            {
                "run_id": run_id,
                "ticket_id": ticket_id,
                "order": shuffle.sample(VARIANTS, 2),
                "outputs": {},
                "preference": None,
            }
            for ticket_id in ticket_ids
        ]
    )
    return run_id


def configs(run: dict) -> Dict[str, EngineConfig]:
    return {v: EngineConfig.from_dict(run["variants"][v]) for v in VARIANTS}


def _output(result: dict) -> dict:
    """What an A/B result keeps of a classification."""
    kept = {key: result.get(key) for key in _KEPT_FIELDS}
    audit = result.get("audit") or {}
    kept["usage"] = audit.get("usage") or {}
    kept["latency_ms"] = audit.get("latency_ms")
    confirmed = result.get("confirmation_of")
    if confirmed:
        kept["confirmed_from"] = {
            key: confirmed.get(key)
            for key in ("model_version", "classification", "confidence")
        }
        # The confirmation replaced an answer that was paid for too.
        kept["usage"] = _sum_usage(
            [kept["usage"], (confirmed.get("audit") or {}).get("usage") or {}]
        )
    return kept


def _sum_usage(usages: Iterable[dict]) -> dict:
    total: Counter = Counter()
    for usage in usages:
        for key, value in (usage or {}).items():
            total[key] += value or 0
    return dict(total)


def run_ab_test(
    run_id: str,
    tickets: Collection,
    runs: Collection,
    results: Collection,
    classify: Callable[..., dict] = classify_ticket,
    on_progress: Optional[Callable[[int, int], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> dict:
    """Classifies every pending ticket of a run with both configurations.

    Resumable: a ticket's output for a variant is saved as soon as it
    arrives, and outputs already saved are skipped. A failed classification
    is saved as that variant's error and the run continues; errors that will
    hit every ticket (bad key, out of credit) stop it, as in a review batch.

    Args:
        on_progress: Called with (done, total) after each ticket.
        should_stop: Checked between tickets; True stops the run (Stopped).

    Returns:
        The run's summary (summarize).
    """
    from .persistence import is_fatal

    run = runs.find_one({"_id": run_id})
    if run is None:
        raise ValueError(f"No A/B run {run_id}.")
    variant_configs = configs(run)
    pending = list(results.find({"run_id": run_id}).sort("ticket_id", 1))
    runs.update_one({"_id": run_id}, {"$set": {"status": "running"}})
    for done, doc in enumerate(pending, start=1):
        if should_stop and should_stop():
            runs.update_one({"_id": run_id}, {"$set": {"status": "partial"}})
            raise Stopped(f"Stopped after {done - 1} of {len(pending)} tickets.")
        missing = [v for v in VARIANTS if v not in (doc.get("outputs") or {})]
        if missing:
            ticket = tickets.find_one({"zendesk_id": doc["ticket_id"]})
            for variant in missing:
                if ticket is None:
                    output = {"error": "Ticket not found."}
                else:
                    try:
                        output = _output(
                            classify(
                                build_ticket_payload(ticket),
                                reengage_count=len(
                                    ticket.get("reengage_attempts") or []
                                ),
                                assignee=ticket.get("assignee"),
                                config=variant_configs[variant],
                            )
                        )
                    except Exception as e:
                        if is_fatal(e):
                            runs.update_one(
                                {"_id": run_id}, {"$set": {"status": "partial"}}
                            )
                            raise
                        logger.warning(
                            f"A/B {run_id}: ticket {doc['ticket_id']} failed on {variant}: {e}"
                        )
                        output = {
                            "error": f"{type(e).__name__}: {e}"[:1000],
                            "usage": ((getattr(e, "audit", None) or {}).get("usage"))
                            or {},
                        }
                results.update_one(
                    {"_id": doc["_id"]}, {"$set": {f"outputs.{variant}": output}}
                )
        if on_progress:
            on_progress(done, len(pending))
    runs.update_one(
        {"_id": run_id},
        {"$set": {"status": "complete", "completed_at": datetime.now(timezone.utc)}},
    )
    return summarize(
        runs.find_one({"_id": run_id}), list(results.find({"run_id": run_id}))
    )


def record_preference(
    results: Collection,
    run_id: str,
    ticket_id: str,
    choice: Optional[str],
    reviewer: str = "",
    note: str = "",
) -> None:
    """Saves a reviewer's verdict on one ticket; None clears it."""
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
    outcome = results.update_one(
        {"run_id": run_id, "ticket_id": str(ticket_id)},
        {"$set": {"preference": preference}},
    )
    if outcome.matched_count == 0:
        raise ValueError(f"Ticket {ticket_id} isn't in A/B run {run_id}.")


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
        run_id = create_run(runs, results, a, b, ids, args.name, seed=args.seed)
        print(f"Created A/B run {run_id} with {len(ids)} tickets.")
        print(f"  A: {a.describe()}\n  B: {b.describe()}")
        print(f"Classify them with: python -m decision_engine.ab_test run {run_id}")
        return 0
    if args.command == "run":
        summary = run_ab_test(
            args.run_id,
            tickets,
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
