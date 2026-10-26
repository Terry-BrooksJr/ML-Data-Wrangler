import copy
import json
import os
from datetime import datetime, timedelta, timezone

import mongomock
import pytest
from bson import ObjectId
from decision_engine import ab_test, feedback, persistence, training_export
from decision_engine.classify import EngineConfig
from decision_engine.prompts import PROMPT_VERSION, SYSTEM_INSTRUCTIONS
from fixtures import CATEGORY_FIXTURES

NOW = datetime.now(timezone.utc)


def review(classification, **extra):
    entry = {
        "review_id": ObjectId(),
        "reviewed_at": NOW - timedelta(days=1),
        "model_version": "m",
        "classification": classification,
        "confidence": 90,
        "signals_matched": ["s"],
        "reasoning": "because",
        "tag_applied": classification,
        "internal_note": "note",
        "draft_message": None,
        "leadership_recipient": None,
        "audit": {"prompt_version": PROMPT_VERSION},
    }
    entry.update(extra)
    return entry


@pytest.fixture
def db():
    database = mongomock.MongoClient().support_db
    log, _ = feedback.get_feedback_collections(database)
    ab_runs, ab_results = ab_test.get_collections(database)

    # Marked correct, with no saved input: rebuilt, without the later comment.
    solved = copy.deepcopy(CATEGORY_FIXTURES["flag_solve"])
    good = review("flag_solve")
    solved["agent_reviews"] = [good]
    solved["transcript"].append(
        {
            "author": "Customer",
            "role": "customer",
            "body": "LATER COMMENT",
            "timestamp": NOW,
        }
    )
    # Marked correct with its exact input saved, but under an older prompt.
    older = copy.deepcopy(CATEGORY_FIXTURES["escalate_leadership"])
    old_review = review(
        "escalate_leadership",
        audit={"prompt_version": "old"},
        input={"as_of": NOW, "payload": "EXACT ```` PAYLOAD"},
    )
    older["agent_reviews"] = [old_review]
    # Marked correct, then logged as misclassified (contradictory).
    stalled = copy.deepcopy(CATEGORY_FIXTURES["flag_reengage"])
    bad = review("flag_solve")
    stalled["agent_reviews"] = [bad]
    escalated = copy.deepcopy(CATEGORY_FIXTURES["escalation_internal"])
    database.tickets.insert_many([solved, older, stalled, escalated])
    feedback.record_outcome(
        database.tickets,
        solved["zendesk_id"],
        good["review_id"],
        "good",
        recorded_by="lead@mac",
    )
    feedback.record_outcome(
        database.tickets, older["zendesk_id"], old_review["review_id"], "good"
    )
    feedback.record_outcome(
        database.tickets, stalled["zendesk_id"], bad["review_id"], "good"
    )
    log_id = feedback.log_misclassification(
        database.tickets,
        log,
        stalled["zendesk_id"],
        bad["review_id"],
        "flag_reengage",
        "no reply",
    )
    feedback.mark_misclassification_reviewed(log, log_id, "lead")
    # Reorder: the "good" outcome comes last, as if the "bad" write was lost.
    database.tickets.update_one(
        {"zendesk_id": stalled["zendesk_id"]},
        {"$push": {"outcomes": {"review_id": bad["review_id"], "outcome": "good"}}},
    )

    # A/B: B preferred (exact payload), both wrong.
    run_id = ab_test.create_run(
        database.tickets,
        ab_runs,
        ab_results,
        EngineConfig("a"),
        EngineConfig("b"),
        [escalated["zendesk_id"], stalled["zendesk_id"]],
    )
    answer = dict(review("escalation_internal"), tag="escalation_internal")
    for doc in ab_results.find({"run_id": run_id}):
        ab_results.update_one(
            {"_id": doc["_id"]},
            {"$set": {"outputs": {"A": answer, "B": dict(answer, reasoning="B's")}}},
        )
    first, second = sorted(d["ticket_id"] for d in ab_results.find())
    ab_test.record_preference(ab_results, run_id, first, "B", 0, "lead@mac")
    ab_test.record_preference(ab_results, run_id, second, "both_bad", 0, "lead@mac")
    return database, log, ab_runs, ab_results


def visible(directory):
    """Files a user would see: not the lock file kept beside an export."""
    return sorted(p.name for p in directory.iterdir() if not p.name.startswith("."))


def export(db, path, **kwargs):
    database, log, ab_runs, ab_results = db
    return training_export.export_training_data(
        database.tickets, log, ab_runs, ab_results, path, **kwargs
    )


def test_jsonl_export_and_its_paired_metadata(db, tmp_path):
    summary = export(db, tmp_path / "out.jsonl")
    assert summary["examples"] == 2  # the A/B verdict and the rebuilt review
    skipped = summary["skipped"]
    assert skipped["older_prompt"] == 1 and skipped["contradicted"] == 1
    assert skipped["ab_tests"] == 1  # both wrong

    data, meta = summary["files"]
    lines = [json.loads(l) for l in data.read_text().splitlines()]
    metas = [json.loads(l) for l in meta.read_text().splitlines()]
    assert len(lines) == len(metas) == 2 and set(lines[0]) == {"messages"}
    assert lines[0]["messages"][0]["content"] == SYSTEM_INSTRUCTIONS
    by_source = {m["source"]: (m, l) for m, l in zip(metas, lines)}
    ab_meta, ab_line = by_source["ab_test"]
    assert ab_meta["input"] == "exact" and ab_meta["judged_by"] == "lead@mac"
    assert json.loads(ab_line["messages"][2]["content"])["reasoning"] == "B's"
    review_meta, review_line = by_source["review"]
    assert review_meta["input"] == "rebuilt" and review_meta["judged_by"] == "lead@mac"
    rebuilt_input = review_line["messages"][1]["content"]
    assert "LATER COMMENT" not in rebuilt_input
    # ...but it holds the ticket as it was: its subject and earlier comments.
    ticket = CATEGORY_FIXTURES["flag_solve"]
    assert ticket["subject"] in rebuilt_input
    assert ticket["transcript"][0]["body"][:40] in rebuilt_input
    assert oct(os.stat(data).st_mode & 0o777) == "0o600"
    assert visible(tmp_path) == ["out.jsonl", "out.meta.jsonl"]


def test_options_include_older_prompts_and_drop_rebuilt_inputs(db, tmp_path):
    summary = export(db, tmp_path / "a.jsonl", include_older_prompts=True)
    assert summary["examples"] == 3
    summary = export(db, tmp_path / "b.jsonl", exact_inputs_only=True)
    assert summary["examples"] == 1 and summary["skipped"]["rebuilt_input"] == 1


def test_markdown_fences_hold_any_backticks_and_list_corrections(db, tmp_path):
    summary = export(db, tmp_path / "out", fmt="markdown", include_older_prompts=True)
    (md,) = summary["files"]
    assert md.name == "out.md"  # the chosen format sets the extension
    text = md.read_text()
    assert "`````\nEXACT ```` PAYLOAD\n`````" in text
    assert "should be flag_reengage, not flag_solve. Misread: no reply." in text


def test_a_failed_publish_keeps_the_previous_pair(db, tmp_path, monkeypatch):
    first = export(db, tmp_path / "out.jsonl")
    before = [p.read_text() for p in first["files"]]
    real_replace = os.replace
    calls = []

    def fail_on_metadata(src, dst):
        calls.append(dst)
        if str(dst).endswith("out.meta.jsonl") and str(src).endswith(".tmp"):
            raise OSError("disk full")
        return real_replace(src, dst)

    monkeypatch.setattr(training_export.os, "replace", fail_on_metadata)
    with pytest.raises(OSError):
        export(db, tmp_path / "out.jsonl", include_older_prompts=True)
    assert [p.read_text() for p in first["files"]] == before
    assert visible(tmp_path) == ["out.jsonl", "out.meta.jsonl"]


def test_reviews_save_the_exact_input_they_were_classified_on(monkeypatch):
    database = mongomock.MongoClient().support_db
    ticket = copy.deepcopy(CATEGORY_FIXTURES["flag_solve"])
    database.tickets.insert_one(ticket)
    seen = []

    def classify(payload, **kwargs):
        seen.append(payload)
        return dict(review("flag_solve"), model_version="m", audit={})

    monkeypatch.setattr(persistence, "classify_ticket", classify)
    moment = datetime(2026, 10, 1, 9, 30, tzinfo=timezone.utc)

    class Frozen(datetime):
        @classmethod
        def now(cls, tz=None):
            return moment

    monkeypatch.setattr(persistence, "datetime", Frozen)
    persistence.process_one_ticket(ticket["zendesk_id"], database.tickets, "w")
    saved = database.tickets.find_one()["agent_reviews"][0]["input"]
    assert saved["payload"] == seen[0]
    # The moment the input was built for, which the payload's as_of shows too.
    assert saved["as_of"].replace(tzinfo=timezone.utc) == moment
    assert '"as_of": "2026-10-01T09:30:00+00:00"' in seen[0]


def test_untimed_comments_and_duplicate_tickets_are_not_rebuilt(db, tmp_path):
    database, log, ab_runs, ab_results = db
    database.tickets.update_one(
        {"zendesk_id": CATEGORY_FIXTURES["flag_solve"]["zendesk_id"]},
        {"$push": {"transcript": {"author": "Agent", "body": "no time"}}},
    )
    # The A/B result is finalized (no payload) and its ticket is duplicated.
    ab_results.update_many({}, {"$unset": {"payload": ""}})
    duplicate = copy.deepcopy(CATEGORY_FIXTURES["escalation_internal"])
    database.tickets.insert_one(duplicate)
    summary = export(db, tmp_path / "out.jsonl")
    assert summary["examples"] == 0
    assert summary["skipped"]["uncertain_history"] == 1
    assert summary["skipped"]["ab_tests"] == 2  # both wrong + ambiguous ticket


def test_a_stopped_publish_is_rolled_back_by_the_next_export(db, tmp_path):
    first = export(db, tmp_path / "out.jsonl")
    before = [p.read_text() for p in first["files"]]
    data_path, meta_path = first["files"]
    # Replay a publish that stopped after the data file moved, before the
    # metadata did: what a killed process leaves behind.
    pairs = []
    for dest in (data_path, meta_path):
        handle, tmp = training_export._temp_beside(dest)
        with handle:
            handle.write("NEW\n")
        pairs.append((tmp, dest))
    plan = [
        {"tmp": str(tmp), "dest": str(dest), "backup": str(dest) + ".bak"}
        for tmp, dest in pairs
    ]
    training_export._write_journal(training_export._journal_path(data_path), plan)
    os.replace(data_path, plan[0]["backup"])
    os.replace(pairs[0][0], data_path)
    assert data_path.read_text() == "NEW\n"  # mismatched with its metadata

    training_export._recover(data_path)  # what the next export does first
    assert [p.read_text() for p in first["files"]] == before
    assert visible(tmp_path) == ["out.jsonl", "out.meta.jsonl"]
    assert not any(
        p.name.endswith((".tmp", ".bak", ".json")) for p in tmp_path.iterdir()
    )


def test_a_misclassification_is_logged_once_and_rolled_back_on_failure(monkeypatch):
    database = mongomock.MongoClient().support_db
    log, _ = feedback.get_feedback_collections(database)
    ticket = copy.deepcopy(CATEGORY_FIXTURES["flag_solve"])
    entry = review("flag_solve")
    ticket["agent_reviews"] = [entry]
    database.tickets.insert_one(ticket)
    zid, rid = ticket["zendesk_id"], entry["review_id"]
    first = feedback.log_misclassification(
        database.tickets, log, zid, rid, "manual_triage"
    )
    again = feedback.log_misclassification(
        database.tickets, log, zid, rid, "flag_reengage", "x"
    )
    assert first == again and log.count_documents({}) == 1
    assert log.find_one()["correct_classification"] == "flag_reengage"

    other = review("flag_solve")
    database.tickets.update_one(
        {"zendesk_id": zid}, {"$push": {"agent_reviews": other}}
    )

    def broken(*args, **kwargs):
        raise RuntimeError("write failed")

    monkeypatch.setattr(feedback, "record_outcome", broken)
    with pytest.raises(RuntimeError):
        feedback.log_misclassification(
            database.tickets, log, zid, other["review_id"], "manual_triage"
        )
    assert log.count_documents({}) == 1  # the new entry was removed again


def test_a_review_sheds_its_saved_input_to_fit_mongodb(monkeypatch):
    database = mongomock.MongoClient().support_db
    ticket = copy.deepcopy(CATEGORY_FIXTURES["flag_solve"])
    database.tickets.insert_one(ticket)
    monkeypatch.setattr(
        persistence,
        "classify_ticket",
        lambda payload, **k: dict(review("flag_solve"), model_version="m", audit={}),
    )
    size = len(__import__("bson").encode(database.tickets.find_one()))
    monkeypatch.setattr(persistence, "MAX_TICKET_BYTES", size + 900)
    persistence.process_one_ticket(ticket["zendesk_id"], database.tickets, "w")
    saved = database.tickets.find_one()["agent_reviews"][0]["input"]
    assert saved["payload"] is None and saved["payload_sha256"]

    monkeypatch.setattr(persistence, "MAX_TICKET_BYTES", 10)
    database.tickets.update_one(
        {"zendesk_id": ticket["zendesk_id"]}, {"$set": {"awaiting_review": True}}
    )
    with pytest.raises(persistence.ReviewTooLarge):
        persistence.process_one_ticket(ticket["zendesk_id"], database.tickets, "w")
    assert "review_claim" not in database.tickets.find_one()  # released


def test_the_cli_accepts_current_prompt_only():
    parser = training_export.build_parser()
    args = parser.parse_args(["x.jsonl", "--current-prompt-only"])
    assert args.current_prompt_only and not args.include_older_prompts
    with pytest.raises(SystemExit):  # the two can't be combined
        parser.parse_args(
            ["x.jsonl", "--current-prompt-only", "--include-older-prompts"]
        )
