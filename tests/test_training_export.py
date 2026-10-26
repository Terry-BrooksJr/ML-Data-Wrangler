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
    solved["transcript"].append({"author": "Agent", "body": "no time"})  # untimed
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
    assert "LATER COMMENT" not in review_line["messages"][1]["content"]
    assert oct(os.stat(data).st_mode & 0o777) == "0o600"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out.jsonl", "out.meta.jsonl"]


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
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out.jsonl", "out.meta.jsonl"]


def test_reviews_save_the_exact_input_they_were_classified_on(monkeypatch):
    database = mongomock.MongoClient().support_db
    ticket = copy.deepcopy(CATEGORY_FIXTURES["flag_solve"])
    database.tickets.insert_one(ticket)
    seen = []

    def classify(payload, **kwargs):
        seen.append(payload)
        return dict(review("flag_solve"), model_version="m", audit={})

    monkeypatch.setattr(persistence, "classify_ticket", classify)
    persistence.process_one_ticket(ticket["zendesk_id"], database.tickets, "w")
    saved = database.tickets.find_one()["agent_reviews"][0]["input"]
    assert saved["payload"] == seen[0] and saved["as_of"] is not None
