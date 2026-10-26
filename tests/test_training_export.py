import copy
import json
import os
from datetime import datetime, timedelta, timezone

import mongomock
from bson import ObjectId
from decision_engine import ab_test, feedback, training_export
from decision_engine.classify import EngineConfig
from decision_engine.prompts import SYSTEM_INSTRUCTIONS
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
        "audit": {"prompt_version": "p1"},
    }
    entry.update(extra)
    return entry


def test_judged_answers_become_examples_and_corrections(tmp_path):
    db = mongomock.MongoClient().support_db
    log, _ = feedback.get_feedback_collections(db)
    ab_runs, ab_results = ab_test.get_collections(db)

    # A review marked correct, on a ticket that got a comment after it.
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
    # A review marked misclassified (and reviewed in the Weekly Review).
    stalled = copy.deepcopy(CATEGORY_FIXTURES["flag_reengage"])
    bad = review("flag_solve")
    stalled["agent_reviews"] = [bad]
    escalated = copy.deepcopy(CATEGORY_FIXTURES["escalation_internal"])
    db.tickets.insert_many([solved, stalled, escalated])
    feedback.record_outcome(db.tickets, solved["zendesk_id"], good["review_id"], "good")
    log_id = feedback.log_misclassification(
        db.tickets,
        log,
        stalled["zendesk_id"],
        bad["review_id"],
        "flag_reengage",
        "no reply in 9 days",
    )
    feedback.mark_misclassification_reviewed(log, log_id, "lead")

    # An A/B test: B preferred on one ticket, both wrong on another.
    run_id = ab_test.create_run(
        db.tickets,
        ab_runs,
        ab_results,
        EngineConfig("a"),
        EngineConfig("b"),
        [CATEGORY_FIXTURES["escalation_internal"]["zendesk_id"], stalled["zendesk_id"]],
    )
    answer = dict(review("escalation_internal"), tag="escalation_internal")
    for doc in ab_results.find({"run_id": run_id}):
        ab_results.update_one(
            {"_id": doc["_id"]},
            {
                "$set": {
                    "outputs": {
                        "A": answer,
                        "B": dict(answer, reasoning="B's reasoning"),
                    }
                }
            },
        )
    first, second = sorted(d["ticket_id"] for d in ab_results.find())
    ab_test.record_preference(ab_results, run_id, first, "B", 0, "lead")
    ab_test.record_preference(ab_results, run_id, second, "both_bad", 0, "lead")

    collected = training_export.collect(db.tickets, log, ab_runs, ab_results)
    examples = collected["examples"]
    assert [e["source"] for e in examples].count("review") == 1
    assert [e["source"] for e in examples].count("ab_test") == 1
    assert collected["skipped"]["ab_tests"] == 1  # both wrong
    review_example = next(e for e in examples if e["source"] == "review")
    assert "LATER COMMENT" not in review_example["payload"]  # rebuilt as it was
    assert review_example["answer"]["tag"] == "flag_solve"
    ab_example = next(e for e in examples if e["source"] == "ab_test")
    assert ab_example["answer"]["reasoning"] == "B's reasoning"
    (fix,) = collected["corrections"]
    assert fix["correct_classification"] == "flag_reengage"

    written = training_export.export(collected, tmp_path / "out.jsonl")
    lines = [json.loads(l) for l in written[0].read_text().splitlines()]
    meta = [json.loads(l) for l in written[1].read_text().splitlines()]
    assert len(lines) == len(meta) == 2 and set(lines[0]) == {"messages"}
    roles = [m["role"] for m in lines[0]["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert lines[0]["messages"][0]["content"] == SYSTEM_INSTRUCTIONS
    json.loads(lines[0]["messages"][2]["content"])  # the target is valid JSON
    assert oct(os.stat(written[0]).st_mode & 0o777) == "0o600"

    (md,) = training_export.export(collected, tmp_path / "out.md")
    text = md.read_text()
    assert "## Example 2" in text and "should be flag_reengage, not flag_solve" in text
    assert "Misread: no reply in 9 days" in text
