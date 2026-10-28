"""Training data from human judgements: fine-tuning JSONL and few-shot examples.

Two kinds of judgement mark an engine answer as right:
- a review marked correct on the Decision Engine tab (its latest outcome is
  "good"), unless it was also logged as misclassified, and
- an A/B verdict: the preferred side's answer ("A" or "B"), or side A's on a
  tie.
Each becomes an example: the ticket as the engine saw it, and that answer as
the target. A review marked misclassified only says which category was
right, not what a full answer would have been, so it becomes a correction,
written to the few-shot file only.

The input is exact where it was saved: a review keeps the payload it was
classified on (persistence.py, `input`), and an A/B result keeps its payload
until the test is finalized. Older reviews and finalized A/B results are
rebuilt from the ticket as of the judgement: comments and re-engage attempts
after that moment are left out, but fields with no history (subject, status,
assignee, priority, tags, Jira links) are as they are now. Each example's
metadata says which ("exact" or "rebuilt"); exact_inputs_only leaves the
rebuilt ones out.

Only answers made with the engine's current prompt are exported unless
include_older_prompts: the system message is today's prompt, and an answer
judged under an older one would teach a mismatched pair.

Formats, chosen by the file's extension:
- .jsonl: one chat-format line per example ({"messages": [system, user,
  assistant]}), as fine-tuning APIs take it. Where each example came from
  goes to a side file, <name>.meta.jsonl, line for line, because those APIs
  reject other keys. The two files are published together.
- .md: numbered examples (ticket, then answer) for a few-shot prompt, then
  the corrections.

Results are streamed to the files, so a large history doesn't have to fit in
memory. The files hold customer-written ticket text: they are created
readable by their owner only. Review them before sharing.

    python -m decision_engine.training_export examples.jsonl
    python -m decision_engine.training_export examples.md --include-older-prompts
"""

import argparse
import hashlib
import json
import os
import pathlib
import re
import sys
import tempfile
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Dict, Iterator, List, Optional, Tuple

from loguru import logger

try:  # serialises exports to the same file; not on Windows
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None
from pymongo.collection import Collection

from .payload import build_ticket_payload
from .prompts import PROMPT_VERSION, SYSTEM_INSTRUCTIONS

# The answer fields the engine returns (prompts.py, "Output format").
ANSWER_FIELDS = (
    "ticket_id",
    "classification",
    "confidence",
    "signals_matched",
    "reengage_attempts",
    "reasoning",
    "tag",
    "internal_note",
    "draft_message",
    "leadership_recipient",
)
FORMATS = {".jsonl": "jsonl", ".md": "markdown"}
EXACT, REBUILT = "exact", "rebuilt"


def _aware(moment) -> Optional[datetime]:
    if not isinstance(moment, datetime):
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def history_unknown(ticket: dict) -> bool:
    """True if a transcript entry or re-engage attempt has no time: whether
    it came before or after a judgement can't be told, so the ticket can't be
    rebuilt as of one."""
    return any(
        _aware(e.get("timestamp")) is None for e in ticket.get("transcript") or []
    ) or any(
        _aware(a.get("drafted_at")) is None
        for a in ticket.get("reengage_attempts") or []
    )


def ticket_as_of(ticket: dict, as_of: datetime) -> dict:
    """The ticket as it was at as_of, as far as its history allows: later
    comments and re-engage attempts are left out, and so are entries with no
    time (callers skip such tickets; see history_unknown)."""

    def before(moment) -> bool:
        moment = _aware(moment)
        return moment is not None and moment <= as_of

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
    # Answers saved before the field existed export it as the list it is now.
    answer["reengage_attempts"] = answer["reengage_attempts"] or []
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


def ab_candidates(
    tickets: Collection, ab_runs: Collection, ab_results: Collection, counts: dict
) -> Iterator[dict]:
    """Examples from A/B verdicts, one at a time: the preferred side's answer,
    or side A's on a tie. "Both wrong", errors and classification-only answers
    (no text) are counted in counts["ab_tests"] and left out. Only the runs
    the verdicts belong to are read."""
    runs: Dict[str, dict] = {}
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
            counts["ab_tests"] += 1
            continue
        run_id = result.get("run_id")
        if run_id not in runs:
            runs[run_id] = (
                ab_runs.find_one({"_id": run_id}, {"prompt_version": 1}) or {}
            )
        payload, kind = result.get("payload"), EXACT
        if not payload:
            # Finalized: the payload was removed, so rebuild it from the ticket.
            # Only from an unambiguous ticket: with duplicate documents for
            # the ID, which one the result was made from can't be told.
            matches = list(tickets.find({"zendesk_id": result["ticket_id"]}).limit(2))
            if len(matches) != 1:
                counts["ab_tests"] += 1
                continue
            ticket = matches[0]
            if history_unknown(ticket):
                counts["uncertain_history"] += 1
                continue
            payload, kind = (
                build_ticket_payload(ticket_as_of(ticket, as_of), as_of),
                REBUILT,
            )
        preference = result.get("preference") or {}
        yield {
            "source": "ab_test",
            "label": {"tie": "equally good (side A)"}.get(
                choice, f"{choice} preferred"
            ),
            "ticket_id": str(result["ticket_id"]),
            "as_of": as_of,
            "input": kind,
            "payload": payload,
            "answer": answer,
            "model": output.get("model_version"),
            "prompt_version": runs[run_id].get("prompt_version"),
            "judged_by": preference.get("reviewer"),
            "judged_at": preference.get("at"),
            "run_id": run_id,
            "note": preference.get("note"),
        }


def review_candidates(
    tickets: Collection, log: Collection, counts: dict
) -> Iterator[dict]:
    """Examples from reviews whose latest outcome is "good", one at a time.

    A review also logged as misclassified is left out, so the same answer is
    never exported both as right and as corrected. The log is checked for
    each review as it is exported, after its ticket was read, so a
    correction logged during the export is honoured (the outcome is written
    before the log entry; see feedback.log_misclassification). Incomplete answers are counted in counts["reviews"].
    """
    for ticket in tickets.find({"outcomes.outcome": "good"}):
        latest = _latest_outcomes(ticket)
        for review in ticket.get("agent_reviews") or []:
            outcome = latest.get(review.get("review_id")) or {}
            if outcome.get("outcome") != "good":
                continue
            if log.find_one({"review_id": review.get("review_id")}, {"_id": 1}):
                counts["contradicted"] += 1
                continue
            answer = answer_of(review, str(ticket["zendesk_id"]))
            saved = review.get("input") or {}
            # The moment the engine was given (saved since inputs were kept),
            # else the review time.
            as_of = _aware(saved.get("as_of")) or _aware(review.get("reviewed_at"))
            if review.get("engine_error") or not _complete(answer) or as_of is None:
                counts["reviews"] += 1
                continue
            payload, kind = saved.get("payload"), EXACT
            if not payload:
                if history_unknown(ticket):
                    counts["uncertain_history"] += 1
                    continue
                payload = build_ticket_payload(ticket_as_of(ticket, as_of), as_of)
                kind = REBUILT
            yield {
                "source": "review",
                "label": "marked correct",
                "ticket_id": str(ticket["zendesk_id"]),
                "as_of": as_of,
                "input": kind,
                "payload": payload,
                "answer": answer,
                "model": review.get("model_version"),
                "prompt_version": (review.get("audit") or {}).get("prompt_version"),
                # Outcomes saved before recorded_by existed have no judge.
                "judged_by": outcome.get("recorded_by"),
                "judged_how": outcome.get("detection_method"),
                "judged_at": outcome.get("detected_at"),
            }


def corrections(log: Collection, reviewed_only: bool = True) -> Iterator[dict]:
    """Reviews marked misclassified, oldest first: the right category, what
    the engine said and the signal it misread. By default only entries
    marked reviewed in the Weekly Review."""
    # Once per review: its latest entry (logs written before a review had a
    # single entry can hold several), and only then whether that is reviewed,
    # so an older reviewed entry never stands in for a newer unreviewed one.
    latest: Dict[object, dict] = {}
    for entry in log.find().sort("logged_at", 1):
        latest[entry.get("review_id") or entry["_id"]] = entry
    for entry in latest.values():
        if reviewed_only and not entry.get("reviewed"):
            continue
        yield {
            "ticket_id": str(entry.get("ticket_id")),
            "agent_classification": entry.get("agent_classification"),
            "correct_classification": entry.get("correct_classification"),
            "misread_signal": entry.get("misread_signal") or "",
        }


def _metadata(example: dict) -> dict:
    return {
        key: (value.isoformat() if isinstance(value, datetime) else value)
        for key, value in example.items()
        if key not in ("payload", "answer")
    }


def jsonl_line(example: dict) -> str:
    """One fine-tuning line: system (today's prompt), user, assistant."""
    return json.dumps(
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


def fence(content: str, minimum: int = 3) -> str:
    """A code fence longer than any run of backticks in content, so the
    content can't close it early."""
    longest = max((len(run) for run in re.findall(r"`+", content)), default=0)
    return "`" * max(minimum, longest + 1)


def markdown_example(n: int, example: dict) -> str:
    answer = json.dumps(example["answer"], indent=2, ensure_ascii=False)
    ticket_fence, answer_fence = fence(example["payload"]), fence(answer)
    return "\n".join(
        [
            "",
            f"## Example {n}: ticket {example['ticket_id']} ({example['label']})",
            "",
            "Ticket:",
            ticket_fence,
            example["payload"],
            ticket_fence,
            "",
            "Answer:",
            answer_fence + "json",
            answer,
            answer_fence,
            "",
        ]
    )


def markdown_correction(fix: dict) -> str:
    line = (
        f"- Ticket {fix['ticket_id']}: should be {fix['correct_classification']}, "
        f"not {fix['agent_classification']}"
    )
    if fix["misread_signal"]:
        line += f". Misread: {fix['misread_signal']}"
    return line + ".\n"


MARKDOWN_HEADER = (
    "# Triage examples\n\nEach example is a ticket as the engine was given it, and an "
    "answer a person judged right. Ticket text is customer-written.\n"
)


def _temp_beside(path: pathlib.Path):
    """A new, uniquely named file beside path, readable by its owner only.
    mkstemp creates it exclusively, so it never follows or reuses an
    existing file or link, and concurrent exports never share one."""
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    return os.fdopen(fd, "w", encoding="utf-8"), pathlib.Path(name)


def _journal_path(path: pathlib.Path) -> pathlib.Path:
    return path.with_name(f".{path.name}.publish.json")


def _write_journal(
    journal: pathlib.Path, plan: List[dict], committed: bool = False
) -> None:
    """Records a publish's plan, and once every file has moved, that it is
    committed. Written atomically, so it is always one or the other."""
    handle, tmp = _temp_beside(journal)
    with handle:
        json.dump({"committed": committed, "plan": plan}, handle)
    os.replace(tmp, journal)


def _finish(plan: List[dict]) -> None:
    """Completes a committed publish: every file is in place, so only its
    leftovers (backups of the replaced files, stray temporaries) go."""
    for step in plan:
        if step.get("backup"):
            pathlib.Path(step["backup"]).unlink(missing_ok=True)
        pathlib.Path(step["tmp"]).unlink(missing_ok=True)


def _roll_back(plan: List[dict]) -> None:
    """Puts each destination back as it was before the publish began."""
    for step in reversed(plan):
        tmp, dest = pathlib.Path(step["tmp"]), pathlib.Path(step["dest"])
        backup = step.get("backup")
        if backup and os.path.lexists(backup):
            os.replace(backup, dest)  # the previous file
        elif not backup and not tmp.exists() and os.path.lexists(dest):
            dest.unlink()  # a new file where there was none
        tmp.unlink(missing_ok=True)


def _recover(path: pathlib.Path) -> None:
    """Settles a publish a stopped export left unfinished, so the files
    always come from one export: a committed one (every file moved) is
    finished, anything earlier is rolled back to the previous pair."""
    journal = _journal_path(path)
    if not journal.exists():
        return
    record = json.loads(journal.read_text())
    if isinstance(record, list):  # journals from before "committed" existed
        record = {"committed": False, "plan": record}
    if record["committed"]:
        logger.warning(f"Finishing an export to {path} that stopped while tidying up.")
        _finish(record["plan"])
    else:
        logger.warning(f"Rolling back an unfinished export to {path}.")
        _roll_back(record["plan"])
    journal.unlink()


def _publish(
    path: pathlib.Path, pairs: List[Tuple[pathlib.Path, pathlib.Path]]
) -> None:
    """Moves each written temporary file onto its destination, all or none.

    The plan (each move, and the backup of the file it replaces) is
    journalled first and marked committed once every file has moved. If a
    move fails, everything is put back as it was. If the process stops, the
    next export to path settles it (_recover): rolled back before the
    commit, finished after it.
    """
    plan = [
        {
            "tmp": str(tmp),
            "dest": str(dest),
            "backup": (
                str(dest.with_name(f".{dest.name}.{uuid.uuid4().hex}.bak"))
                if os.path.lexists(dest)
                else None
            ),
        }
        for tmp, dest in pairs
    ]
    journal = _journal_path(path)
    _write_journal(journal, plan)
    try:
        for step in plan:
            if step["backup"]:
                os.replace(step["dest"], step["backup"])
            os.replace(step["tmp"], step["dest"])
    except BaseException:
        _roll_back(plan)
        journal.unlink(missing_ok=True)
        raise
    # Every file is in place: commit before deleting any backup, so a stop
    # from here on is finished, not rolled back with a backup missing.
    _write_journal(journal, plan, committed=True)
    _finish(plan)
    journal.unlink(missing_ok=True)


@contextmanager
def _exclusive(path: pathlib.Path):
    """Holds a lock beside path for the whole export, so two exports to the
    same file take turns instead of interleaving their files."""
    lock_path = path.with_name(f".{path.name}.lock")
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        if fcntl is not None:
            fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)  # closing releases the lock


def destination(path, fmt: Optional[str] = None) -> Tuple[pathlib.Path, str]:
    """The file to write and its format. fmt ("jsonl" or "markdown") wins over
    the extension, which is changed to match; without it the extension
    decides."""
    path = pathlib.Path(path)
    if fmt is None:
        fmt = FORMATS.get(path.suffix.lower())
        if fmt is None:
            raise ValueError(f"Export to a .jsonl or .md file, not {path.name!r}.")
    suffix = {v: k for k, v in FORMATS.items()}[fmt]
    if path.suffix.lower() != suffix:
        path = path.with_name(path.name + suffix)
    return path, fmt


def export_training_data(
    tickets: Collection,
    log: Collection,
    ab_runs: Optional[Collection],
    ab_results: Optional[Collection],
    path,
    fmt: Optional[str] = None,
    include_older_prompts: bool = False,
    exact_inputs_only: bool = False,
    all_corrections: bool = False,
) -> dict:
    """Streams every judged example (and, for Markdown, every correction) to
    path, then publishes the file(s) together.

    A/B verdicts are read first, so where a verdict and a review share an
    input, the verdict is kept. Filters apply before duplicates are dropped.

    Args:
        fmt: "jsonl" or "markdown"; default from path's extension.
        include_older_prompts: Also export answers made with an older prompt.
        exact_inputs_only: Leave out examples whose input had to be rebuilt.
        all_corrections: Include misclassifications not marked reviewed.

    Returns:
        {"files": [paths written], "examples", "corrections", "rebuilt",
        "skipped": {"reviews", "ab_tests", "older_prompt", "rebuilt_input",
        "duplicates", "contradicted", "uncertain_history"}}.
    """
    path, fmt = destination(path, fmt)
    with _exclusive(path):
        _recover(path)
        return _export_locked(
            tickets,
            log,
            ab_runs,
            ab_results,
            path,
            fmt,
            include_older_prompts,
            exact_inputs_only,
            all_corrections,
        )


def _export_locked(
    tickets,
    log,
    ab_runs,
    ab_results,
    path: pathlib.Path,
    fmt: str,
    include_older_prompts: bool,
    exact_inputs_only: bool,
    all_corrections: bool,
) -> dict:
    counts = dict.fromkeys(
        (
            "reviews",
            "ab_tests",
            "older_prompt",
            "rebuilt_input",
            "duplicates",
            "contradicted",
            "uncertain_history",
        ),
        0,
    )

    def examples() -> Iterator[dict]:
        if ab_runs is not None and ab_results is not None:
            yield from ab_candidates(tickets, ab_runs, ab_results, counts)
        yield from review_candidates(tickets, log, counts)

    data, data_tmp = _temp_beside(path)
    meta_path = path.with_name(path.name[: -len(".jsonl")] + ".meta.jsonl")
    meta, meta_tmp = (None, None)
    if fmt == "jsonl":
        meta, meta_tmp = _temp_beside(meta_path)
    written = rebuilt = fixes = 0
    seen = set()
    try:
        if fmt == "markdown":
            data.write(MARKDOWN_HEADER)
        for example in examples():
            if (
                not include_older_prompts
                and example.get("prompt_version") != PROMPT_VERSION
            ):
                counts["older_prompt"] += 1
                continue
            if exact_inputs_only and example["input"] != EXACT:
                counts["rebuilt_input"] += 1
                continue
            key = hashlib.sha256(example["payload"].encode()).hexdigest()
            if key in seen:
                counts["duplicates"] += 1
                continue
            seen.add(key)
            written += 1
            rebuilt += example["input"] == REBUILT
            if fmt == "jsonl":
                data.write(jsonl_line(example) + "\n")
                meta.write(
                    json.dumps(_metadata(example), ensure_ascii=False, default=str)
                    + "\n"
                )
            else:
                data.write(markdown_example(written, example))
        if fmt == "markdown":
            for fix in corrections(log, reviewed_only=not all_corrections):
                if not fixes:
                    data.write("\n# Corrected classifications\n\n")
                data.write(markdown_correction(fix))
                fixes += 1
        data.close()
        if meta is not None:
            meta.close()
        pairs = [(data_tmp, path)] + ([(meta_tmp, meta_path)] if meta_tmp else [])
        _publish(path, pairs)
    finally:
        for handle in (data, meta):
            if handle is not None and not handle.closed:
                handle.close()
        for tmp in (data_tmp, meta_tmp):
            if tmp is not None:
                tmp.unlink(missing_ok=True)
    return {
        "files": [path] + ([meta_path] if fmt == "jsonl" else []),
        "examples": written,
        "corrections": fixes,
        "rebuilt": rebuilt,
        "skipped": counts,
    }


def describe(summary: dict) -> str:
    """One paragraph on what was exported and what was left out."""
    skipped = summary["skipped"]
    left_out = [
        f"{count} {what}"
        for count, what in (
            (skipped["older_prompt"], "made with an older prompt"),
            (skipped["rebuilt_input"], "whose input couldn't be recovered exactly"),
            (
                skipped["contradicted"],
                "marked correct but also logged as misclassified",
            ),
            (
                skipped["uncertain_history"],
                "whose comments without a time made the input impossible to rebuild",
            ),
            (skipped["reviews"], "correct review(s) without a full answer"),
            (skipped["ab_tests"], "A/B verdict(s) with no full answer to learn from"),
            (skipped["duplicates"], "duplicate input(s)"),
        )
        if count
    ]
    files = ", ".join(str(p) for p in summary["files"])
    text = f"Exported {summary['examples']} example(s)"
    if summary["files"][0].suffix.lower() == ".md":
        text += f" and {summary['corrections']} correction(s)"
    text += f" to {files}."
    if summary["rebuilt"]:
        text += (
            f" {summary['rebuilt']} input(s) were rebuilt from the ticket (comments as of "
            "the judgement; subject, status and other fields as they are now)."
        )
    if left_out:
        text += " Left out: " + "; ".join(left_out) + "."
    return text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m decision_engine.training_export")
    parser.add_argument(
        "path", help="examples.jsonl (fine-tuning) or examples.md (few-shot)"
    )
    prompts = parser.add_mutually_exclusive_group()
    prompts.add_argument(
        "--current-prompt-only",
        action="store_true",
        help="only answers made with the current prompt (the default)",
    )
    prompts.add_argument(
        "--include-older-prompts",
        action="store_true",
        help="also export answers made with an older prompt (paired with today's)",
    )
    parser.add_argument(
        "--exact-inputs-only",
        action="store_true",
        help="leave out examples whose input had to be rebuilt from the ticket",
    )
    parser.add_argument(
        "--all-corrections",
        action="store_true",
        help="include misclassifications not yet marked reviewed",
    )
    return parser


def main(argv: List[str]) -> int:
    args = build_parser().parse_args(argv)

    try:  # the app's saved Settings, like the app itself (run from src/)
        from core import settings

        settings.load()
    except ImportError:
        pass
    from . import ab_test, feedback
    from .persistence import get_database, get_tickets_collection

    database = get_database()
    log, _ = feedback.get_feedback_collections(database)
    ab_runs, ab_results = ab_test.get_collections(database)
    summary = export_training_data(
        get_tickets_collection(),
        log,
        ab_runs,
        ab_results,
        args.path,
        include_older_prompts=args.include_older_prompts,
        exact_inputs_only=args.exact_inputs_only,
        all_corrections=args.all_corrections,
    )
    print(describe(summary))
    logger.info("The files hold customer ticket text; review them before sharing.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
