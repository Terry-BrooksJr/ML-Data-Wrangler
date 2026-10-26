"""Training data from human judgements: fine-tuning JSONL and few-shot examples.

Two kinds of judgement mark an engine answer as right:
- a review marked correct on the Decision Engine tab (its latest outcome is
  "good"), and
- an A/B verdict: the preferred side's answer ("A" or "B"), or side A's on a
  tie.
Each becomes an example: the ticket as the engine saw it, and that answer as
the target. A review marked misclassified (misclassification_log) only says
which category was right, not what a full answer would have been, so it
becomes a correction, written to the few-shot file only.

The ticket is rebuilt as it was when it was judged: comments and re-engage
attempts after that moment are left out, and "as_of" is that moment. An A/B
result that still holds its payload uses it as is.

Formats, chosen by the file's extension:
- .jsonl: one chat-format line per example ({"messages": [system, user,
  assistant]}), as fine-tuning APIs take it. The system message is the
  engine's current prompt. Where each example came from goes to a side file,
  <name>.meta.jsonl, line for line, because those APIs reject other keys.
- .md: numbered examples (ticket, then answer) for a few-shot prompt, then
  the corrections.

The files hold customer-written ticket text: they are created readable by
their owner only. Review them before sharing.

    python -m decision_engine.training_export examples.jsonl
    python -m decision_engine.training_export examples.md
"""

import argparse
import hashlib
import json
import os
import pathlib
import sys
from datetime import datetime, timezone
from typing import Iterable, List, Optional, Tuple

from loguru import logger
from pymongo.collection import Collection

from .payload import build_ticket_payload
from .prompts import PROMPT_VERSION, SYSTEM_INSTRUCTIONS

# The answer fields the engine returns (prompts.py, "Output format").
ANSWER_FIELDS = (
    "ticket_id",
    "classification",
    "confidence",
    "signals_matched",
    "reasoning",
    "tag",
    "internal_note",
    "draft_message",
    "leadership_recipient",
)


def _aware(moment) -> Optional[datetime]:
    if not isinstance(moment, datetime):
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def ticket_as_of(ticket: dict, as_of: datetime) -> dict:
    """The ticket as it was at as_of: later comments and re-engage attempts
    are left out (entries without a time are kept)."""

    def before(moment) -> bool:
        moment = _aware(moment)
        return moment is None or moment <= as_of

    past = dict(ticket)
    past["transcript"] = [
        entry
        for entry in ticket.get("transcript") or []
        if before(entry.get("timestamp"))
    ]
    past["reengage_attempts"] = [
        attempt
        for attempt in ticket.get("reengage_attempts") or []
        if before(attempt.get("drafted_at"))
    ]
    return past


def answer_of(source: dict, ticket_id: str) -> dict:
    """The target answer, in the engine's output format. A review stores its
    tag as tag_applied."""
    answer = {field: source.get(field) for field in ANSWER_FIELDS}
    answer["ticket_id"] = ticket_id
    answer["tag"] = (
        source.get("tag") or source.get("tag_applied") or answer["classification"]
    )
    answer["signals_matched"] = list(source.get("signals_matched") or [])
    return answer


def _complete(answer: dict) -> bool:
    """A full answer to learn from: a classification, a confidence and reasoning."""
    return bool(
        answer.get("classification")
        and isinstance(answer.get("confidence"), (int, float))
        and answer.get("reasoning")
    )


def _latest_outcomes(ticket: dict) -> dict:
    # Outcomes are only ever appended, so the last one per review is the latest.
    return {entry.get("review_id"): entry for entry in ticket.get("outcomes") or []}


def review_examples(tickets: Collection) -> Tuple[List[dict], int]:
    """Examples from reviews whose latest outcome is "good" (marked correct).

    Returns:
        (examples, how many correct reviews were left out as incomplete).
    """
    examples, skipped = [], 0
    for ticket in tickets.find({"outcomes.outcome": "good"}):
        latest = _latest_outcomes(ticket)
        for review in ticket.get("agent_reviews") or []:
            outcome = latest.get(review.get("review_id")) or {}
            if outcome.get("outcome") != "good":
                continue
            answer = answer_of(review, str(ticket["zendesk_id"]))
            as_of = _aware(review.get("reviewed_at"))
            if review.get("engine_error") or not _complete(answer) or as_of is None:
                skipped += 1
                continue
            examples.append(
                {
                    "source": "review",
                    "label": "marked correct",
                    "ticket_id": str(ticket["zendesk_id"]),
                    "as_of": as_of,
                    "payload": build_ticket_payload(ticket_as_of(ticket, as_of), as_of),
                    "answer": answer,
                    "model": review.get("model_version"),
                    "prompt_version": (review.get("audit") or {}).get("prompt_version"),
                    "judged_by": outcome.get("detection_method"),
                }
            )
    return examples, skipped


def ab_examples(
    tickets: Collection, ab_runs: Collection, ab_results: Collection
) -> Tuple[List[dict], int]:
    """Examples from A/B verdicts: the preferred side's answer, or side A's on
    a tie. "Both wrong", errors and classification-only answers (no text)
    are left out.

    Returns:
        (examples, how many verdicts were left out).
    """
    runs = {run["_id"]: run for run in ab_runs.find()}
    examples, skipped = [], 0
    for result in ab_results.find({"preference": {"$ne": None}}):
        choice = (result.get("preference") or {}).get("choice")
        side = {"A": "A", "B": "B", "tie": "A"}.get(choice)
        output = ((result.get("outputs") or {}).get(side) or {}) if side else {}
        answer = answer_of(output, str(result["ticket_id"]))
        as_of = _aware(result.get("as_of"))
        if (
            side is None
            or "error" in output
            or output.get("decision_only")
            or not _complete(answer)
            or as_of is None
        ):
            skipped += 1
            continue
        payload = result.get("payload")
        if not payload:
            # Finalized: the payload was removed, so rebuild it from the ticket.
            ticket = tickets.find_one({"zendesk_id": result["ticket_id"]})
            if ticket is None:
                skipped += 1
                continue
            payload = build_ticket_payload(ticket_as_of(ticket, as_of), as_of)
        run = runs.get(result.get("run_id")) or {}
        examples.append(
            {
                "source": "ab_test",
                "label": {"tie": "equally good (side A)"}.get(
                    choice, f"{choice} preferred"
                ),
                "ticket_id": str(result["ticket_id"]),
                "as_of": as_of,
                "payload": payload,
                "answer": answer,
                "model": output.get("model_version"),
                "prompt_version": run.get("prompt_version"),
                "judged_by": (result.get("preference") or {}).get("reviewer"),
                "run_id": result.get("run_id"),
                "note": (result.get("preference") or {}).get("note"),
            }
        )
    return examples, skipped


def corrections(log: Collection, reviewed_only: bool = True) -> List[dict]:
    """Reviews marked misclassified: the right category, what the engine said
    and the signal it misread. By default only entries marked reviewed in
    the Weekly Review."""
    query = {"reviewed": True} if reviewed_only else {}
    return [
        {
            "ticket_id": str(entry.get("ticket_id")),
            "agent_classification": entry.get("agent_classification"),
            "correct_classification": entry.get("correct_classification"),
            "misread_signal": entry.get("misread_signal") or "",
            "logged_at": entry.get("logged_at"),
        }
        for entry in log.find(query).sort("logged_at", 1)
    ]


def _dedupe(examples: Iterable[dict]) -> List[dict]:
    """One example per distinct input; an A/B verdict wins over a review."""
    chosen = {}
    for example in sorted(examples, key=lambda e: e["source"] != "ab_test"):
        key = hashlib.sha256(example["payload"].encode()).hexdigest()
        chosen.setdefault(key, example)
    return sorted(chosen.values(), key=lambda e: (e["ticket_id"], e["as_of"]))


def collect(
    tickets: Collection,
    log: Collection,
    ab_runs: Optional[Collection] = None,
    ab_results: Optional[Collection] = None,
    current_prompt_only: bool = False,
    reviewed_corrections_only: bool = True,
) -> dict:
    """Every example and correction, deduplicated.

    Args:
        current_prompt_only: Keep only answers made with today's prompt.

    Returns:
        {"examples", "corrections", "skipped": {"reviews", "ab_tests",
        "older_prompt"}}.
    """
    reviews, skipped_reviews = review_examples(tickets)
    tests, skipped_tests = (
        ab_examples(tickets, ab_runs, ab_results)
        if ab_runs is not None and ab_results is not None
        else ([], 0)
    )
    examples = _dedupe(reviews + tests)
    older = 0
    if current_prompt_only:
        kept = [e for e in examples if e.get("prompt_version") == PROMPT_VERSION]
        older, examples = len(examples) - len(kept), kept
    return {
        "examples": examples,
        "corrections": corrections(log, reviewed_corrections_only),
        "skipped": {
            "reviews": skipped_reviews,
            "ab_tests": skipped_tests,
            "older_prompt": older,
        },
    }


def _metadata(example: dict) -> dict:
    return {
        key: (value.isoformat() if isinstance(value, datetime) else value)
        for key, value in example.items()
        if key not in ("payload", "answer")
    }


def to_jsonl(examples: List[dict]) -> Tuple[str, str]:
    """(fine-tuning lines, metadata lines), one line per example in each."""
    data, meta = [], []
    for example in examples:
        data.append(
            json.dumps(
                {
                    "messages": [
                        {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                        {"role": "user", "content": example["payload"]},
                        {
                            "role": "assistant",
                            "content": json.dumps(example["answer"], indent=2),
                        },
                    ]
                },
                ensure_ascii=False,
            )
        )
        meta.append(json.dumps(_metadata(example), ensure_ascii=False, default=str))
    return "\n".join(data) + ("\n" if data else ""), "\n".join(meta) + (
        "\n" if meta else ""
    )


def to_few_shot(examples: List[dict], fixes: List[dict]) -> str:
    """Markdown: numbered examples (ticket, then answer), then corrections."""
    lines = [
        "# Triage examples",
        "",
        "Each example is a ticket as the engine was given it, and an answer a "
        "person judged right. Ticket text is customer-written.",
    ]
    for n, example in enumerate(examples, start=1):
        lines += [
            "",
            f"## Example {n}: ticket {example['ticket_id']} ({example['label']})",
            "",
            "Ticket:",
            "````",
            example["payload"],
            "````",
            "",
            "Answer:",
            "```json",
            json.dumps(example["answer"], indent=2, ensure_ascii=False),
            "```",
        ]
    if fixes:
        lines += ["", "# Corrected classifications", ""]
        for fix in fixes:
            line = (
                f"- Ticket {fix['ticket_id']}: should be {fix['correct_classification']}, "
                f"not {fix['agent_classification']}"
            )
            if fix["misread_signal"]:
                line += f". Misread: {fix['misread_signal']}"
            lines.append(line + ".")
    return "\n".join(lines) + "\n"


def _write_private(path: pathlib.Path, text: str) -> None:
    """Writes text readable by its owner only, via a temporary file, so a
    failed export never leaves a half-written file."""
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def export(collected: dict, path) -> List[pathlib.Path]:
    """Writes collect()'s result to path (.jsonl or .md); returns the files written."""
    path = pathlib.Path(path)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        data, meta = to_jsonl(collected["examples"])
        meta_path = path.with_name(path.stem + ".meta.jsonl")
        _write_private(path, data)
        _write_private(meta_path, meta)
        return [path, meta_path]
    if suffix in (".md", ".txt"):
        _write_private(
            path, to_few_shot(collected["examples"], collected["corrections"])
        )
        return [path]
    raise ValueError(f"Export to a .jsonl or .md file, not {path.name!r}.")


def describe(collected: dict, written: List[pathlib.Path]) -> str:
    """One paragraph on what was exported and what was left out."""
    skipped = collected["skipped"]
    left_out = [
        f"{count} {what}"
        for count, what in (
            (skipped["reviews"], "correct review(s) without a full answer"),
            (skipped["ab_tests"], "A/B verdict(s) with no full answer to learn from"),
            (skipped["older_prompt"], "example(s) made with an older prompt"),
        )
        if count
    ]
    text = (
        f"Exported {len(collected['examples'])} example(s)"
        + (
            f" and {len(collected['corrections'])} correction(s)"
            if written and written[0].suffix.lower() in (".md", ".txt")
            else ""
        )
        + f" to {', '.join(str(p) for p in written)}."
    )
    if left_out:
        text += " Left out: " + "; ".join(left_out) + "."
    return text


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m decision_engine.training_export")
    parser.add_argument(
        "path", help="examples.jsonl (fine-tuning) or examples.md (few-shot)"
    )
    parser.add_argument(
        "--current-prompt-only",
        action="store_true",
        help="leave out answers made with an older prompt",
    )
    parser.add_argument(
        "--all-corrections",
        action="store_true",
        help="include misclassifications not yet marked reviewed",
    )
    args = parser.parse_args(argv)

    from . import ab_test, feedback
    from .persistence import get_database, get_tickets_collection

    database = get_database()
    log, _ = feedback.get_feedback_collections(database)
    ab_runs, ab_results = ab_test.get_collections(database)
    collected = collect(
        get_tickets_collection(),
        log,
        ab_runs,
        ab_results,
        current_prompt_only=args.current_prompt_only,
        reviewed_corrections_only=not args.all_corrections,
    )
    written = export(collected, args.path)
    print(describe(collected, written))
    logger.info("The files hold customer ticket text; review them before sharing.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
