import argparse
import copy
import json
from datetime import datetime, timedelta, timezone

import mongomock
import pytest
from decision_engine import ab_test
from decision_engine.classify import EngineConfig, EngineError
from fixtures import CATEGORY_FIXTURES
from test_decision_engine import status_error

A = EngineConfig("current-model", confirmation="checker")
B = EngineConfig("new-model", confirmation="checker", max_tokens=16000, effort="high")


@pytest.fixture
def db():
    database = mongomock.MongoClient().support_db
    database.tickets.insert_many(
        [copy.deepcopy(ticket) for ticket in CATEGORY_FIXTURES.values()]
    )
    return database


@pytest.fixture
def collections(db):
    return (db.tickets, *ab_test.get_collections(db))


def answer(classification, confidence=90, **extra):
    """A classify_ticket result, as the engine returns it."""
    result = {
        "classification": classification,
        "confidence": confidence,
        "signals_matched": ["s"],
        "reasoning": "r",
        "internal_note": "n",
        "draft_message": None,
        "leadership_recipient": None,
        "tag": classification,
        "guard_overrides": [],
        "audit": {"usage": {"input_tokens": 10, "output_tokens": 5}, "latency_ms": 200},
    }
    result.update(extra)
    return result


class FakeEngine:
    """Answers by model: B solves everything, A sends everything to manual triage."""

    def __init__(self, fail=None, during=None):
        self.calls = []
        self.fail = fail or {}  # (ticket subject, primary) -> exception
        self.during = during  # called before answering, to simulate other workers

    def __call__(
        self,
        payload,
        reengage_count=0,
        assignee=None,
        config=None,
        allow_decision_models=False,
    ):
        assert allow_decision_models  # A/B tests may compare decision models
        self.calls.append((payload, config))
        if self.during:
            self.during()
        for (subject, primary), error in self.fail.items():
            if subject in payload and primary == config.primary:
                raise error
        classification = (
            "flag_solve" if config.primary == "new-model" else "manual_triage"
        )
        return answer(classification, model_version=config.primary)


def new_run(collections, size=3, seed=1):
    tickets, runs, results = collections
    ids = ab_test.sample_ticket_ids(tickets, size, seed=seed)
    return ab_test.create_run(tickets, runs, results, A, B, ids, "test run", seed=seed)


def run(collections, run_id, **kwargs):
    _, runs, results = collections
    kwargs.setdefault("classify", FakeEngine())
    return ab_test.run_ab_test(run_id, runs, results, **kwargs)


def status(collections, run_id):
    return collections[1].find_one({"_id": run_id})["status"]


def ticket_ids(collections, run_id):
    return sorted(d["ticket_id"] for d in collections[2].find({"run_id": run_id}))


# --- collections ---------------------------------------------------------------


@pytest.mark.parametrize(
    "env, clash",
    [
        ({"AB_TEST_COLLECTION": "tickets"}, "same as the tickets"),
        (
            {"AB_RESULT_COLLECTION": "my_tickets", "MONGO_COLLECTION": "my_tickets"},
            "same as the tickets",
        ),
        ({"AB_TEST_COLLECTION": "ab", "AB_RESULT_COLLECTION": "ab"}, "both named"),
    ],
)
def test_collection_names_that_clash_are_refused(db, monkeypatch, env, clash):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=clash):
        ab_test.get_collections(db)
    assert db.tickets.count_documents({}) == len(CATEGORY_FIXTURES)


# --- creating ------------------------------------------------------------------


def test_sample_is_repeatable_and_can_be_limited_to_awaiting(db):
    db.tickets.update_one(
        {"zendesk_id": CATEGORY_FIXTURES["flag_solve"]["zendesk_id"]},
        {"$set": {"awaiting_review": False}},
    )
    first = ab_test.sample_ticket_ids(db.tickets, 3, seed=7)
    assert first == ab_test.sample_ticket_ids(db.tickets, 3, seed=7)
    awaiting = ab_test.sample_ticket_ids(db.tickets, 100, awaiting_only=True)
    assert CATEGORY_FIXTURES["flag_solve"]["zendesk_id"] not in awaiting
    assert len(ab_test.sample_ticket_ids(db.tickets, 100)) == len(awaiting) + 1


def test_duplicate_ticket_documents_are_a_clear_error(collections):
    tickets, runs, results = collections
    duplicate = copy.deepcopy(CATEGORY_FIXTURES["flag_solve"])
    tickets.insert_one(duplicate)
    with pytest.raises(ValueError, match="more than one document"):
        ab_test.sample_ticket_ids(tickets, 3)
    with pytest.raises(ValueError, match="more than one document"):
        ab_test.create_run(tickets, runs, results, A, B, [duplicate["zendesk_id"]])
    with pytest.raises(ValueError, match="listed more than once"):
        ab_test.create_run(tickets, runs, results, A, B, ["1", "1"])
    assert runs.count_documents({}) == 0


def test_create_run_records_variants_snapshots_and_a_mixed_blind_order(collections):
    tickets, runs, results = collections
    run_id = new_run(collections, size=5)
    run_doc = runs.find_one({"_id": run_id})
    assert run_doc["variants"]["B"]["effort"] == "high"
    assert run_doc["status"] == "pending" and run_doc["ticket_count"] == 5
    docs = list(results.find({"run_id": run_id}))
    assert len(docs) == 5
    # Seeded, so this exact mix of orders repeats: both orders occur.
    assert {tuple(d["order"]) for d in docs} == {("A", "B"), ("B", "A")}
    assert all(d["outputs"] == {} and d["preference"] is None for d in docs)
    doc = docs[0]
    assert doc["ticket"]["subject"] in doc["payload"]
    assert doc["ticket"]["zendesk_id"] == doc["ticket_id"]


def test_create_run_refuses_bad_input(collections):
    tickets, runs, results = collections
    same = EngineConfig.from_dict(A.to_dict())
    with pytest.raises(ValueError, match="same configuration"):
        ab_test.create_run(tickets, runs, results, A, same, ["1"])
    with pytest.raises(ValueError, match="No tickets"):
        ab_test.create_run(tickets, runs, results, A, B, [])
    with pytest.raises(ValueError, match="not found"):
        ab_test.create_run(tickets, runs, results, A, B, ["no-such-ticket"])
    assert runs.count_documents({}) == 0


def test_a_failed_result_insert_leaves_no_run(collections, monkeypatch):
    tickets, runs, results = collections

    def interrupted(docs, *args, **kwargs):
        results.insert_one(docs[0])  # part of the batch got in
        raise RuntimeError("connection lost")

    monkeypatch.setattr(results, "insert_many", interrupted)
    with pytest.raises(RuntimeError):
        new_run(collections)
    assert runs.count_documents({}) == 0 and results.count_documents({}) == 0


def test_a_run_whose_creation_was_interrupted_is_not_classified(collections):
    run_id = new_run(collections)
    collections[1].update_one({"_id": run_id}, {"$set": {"status": "creating"}})
    with pytest.raises(ValueError, match="wasn't created completely"):
        run(collections, run_id)


# --- running ---------------------------------------------------------------------


def test_run_classifies_each_ticket_with_both_configurations(collections):
    tickets, runs, results = collections
    before = list(tickets.find())
    run_id = new_run(collections)
    engine = FakeEngine()
    summary = run(collections, run_id, classify=engine)

    assert (
        sorted(c.primary for _, c in engine.calls)
        == ["current-model"] * 3 + ["new-model"] * 3
    )
    assert {c for _, c in engine.calls} == {A, B}
    doc = results.find_one({"run_id": run_id})
    assert doc["outputs"]["A"]["classification"] == "manual_triage"
    assert doc["outputs"]["B"]["usage"] == {"input_tokens": 10, "output_tokens": 5}
    assert "audit" not in doc["outputs"]["B"] and doc["claim"] is None
    assert status(collections, run_id) == "complete"
    assert summary["classified"] == 3 and summary["agreement"] == 0
    # An A/B run never changes the tickets themselves.
    assert list(tickets.find()) == before


def test_both_variants_get_the_snapshot_not_the_live_ticket(collections):
    tickets, runs, results = collections
    run_id = new_run(collections, size=1)
    ticket_id = ticket_ids(collections, run_id)[0]
    original = tickets.find_one({"zendesk_id": ticket_id})["subject"]
    tickets.update_one(
        {"zendesk_id": ticket_id}, {"$set": {"subject": "Changed after the run began"}}
    )

    engine = FakeEngine()
    run(collections, run_id, classify=engine)
    payloads = [payload for payload, _ in engine.calls]
    assert len(payloads) == 2 and payloads[0] == payloads[1]
    assert original in payloads[0] and "Changed after" not in payloads[0]
    assert results.find_one({"run_id": run_id})["ticket"]["subject"] == original


def test_a_failure_is_saved_and_retried_by_the_next_run(collections):
    tickets, runs, results = collections
    run_id = new_run(collections, size=len(CATEGORY_FIXTURES))
    subject = CATEGORY_FIXTURES["flag_solve"]["subject"]
    error = EngineError("bad reply", "new-model", {"usage": {"output_tokens": 9}})
    summary = run(
        collections, run_id, classify=FakeEngine(fail={(subject, "new-model"): error})
    )

    assert summary["variants"]["B"]["errors"] == 1
    assert summary["variants"]["A"]["errors"] == 0
    assert summary["classified"] == len(CATEGORY_FIXTURES) - 1
    assert status(collections, run_id) == "partial"
    failed = results.find_one({"outputs.B.error": {"$exists": True}})
    assert "bad reply" in failed["outputs"]["B"]["error"]
    assert failed["outputs"]["B"]["usage"] == {"output_tokens": 9}
    assert not ab_test.answered(failed) and not ab_test.differs(failed)

    retry = FakeEngine()
    run(collections, run_id, classify=retry)
    assert [c.primary for _, c in retry.calls] == ["new-model"]
    assert ab_test.answered(results.find_one({"_id": failed["_id"]}))
    assert status(collections, run_id) == "complete"


def test_a_fatal_error_stops_the_run_and_it_resumes(collections):
    tickets, runs, results = collections
    run_id = new_run(collections, size=3)
    second = ticket_ids(collections, run_id)[1]
    subject = tickets.find_one({"zendesk_id": second})["subject"]
    engine = FakeEngine(fail={(subject, "new-model"): status_error(401)})
    with pytest.raises(Exception, match="401"):
        run(collections, run_id, classify=engine)
    assert status(collections, run_id) == "partial"
    assert len(engine.calls) == 4  # ticket 1 both, ticket 2 A then B (fails)
    assert results.count_documents({"claim": {"$ne": None}}) == 0

    resumed = FakeEngine()
    run(collections, run_id, classify=resumed)
    # Only what was missing: ticket 2's B and ticket 3's A and B.
    assert len(resumed.calls) == 3
    assert status(collections, run_id) == "complete"


def test_stop_saves_what_finished(collections):
    run_id = new_run(collections, size=3)
    engine = FakeEngine()
    with pytest.raises(ab_test.Stopped):
        run(
            collections,
            run_id,
            classify=engine,
            should_stop=lambda: len(engine.calls) >= 2,
        )
    assert len(engine.calls) == 2
    assert (
        collections[2].count_documents(
            {"run_id": run_id, "outputs.B": {"$exists": True}}
        )
        == 1
    )
    assert status(collections, run_id) == "partial"


def test_an_unexpected_failure_never_leaves_the_run_running(collections):
    run_id = new_run(collections, size=3)

    def broken(done, total):
        raise RuntimeError("progress display failed")

    with pytest.raises(RuntimeError):
        run(collections, run_id, on_progress=broken)
    assert status(collections, run_id) == "partial"
    assert collections[2].count_documents({"claim": {"$ne": None}}) == 0


def test_a_ticket_claimed_by_another_worker_is_skipped(collections):
    _, runs, results = collections
    run_id = new_run(collections, size=2)
    first, second = ticket_ids(collections, run_id)
    now = datetime.now(timezone.utc)
    results.update_one(
        {"ticket_id": first}, {"$set": {"claim": {"worker": "other", "at": now}}}
    )
    results.update_one(
        {"ticket_id": second},
        {"$set": {"claim": {"worker": "crashed", "at": now - timedelta(hours=1)}}},
    )
    engine = FakeEngine()
    run(collections, run_id, classify=engine, worker="me")

    # The live claim is left alone; the abandoned one is taken over.
    assert len(engine.calls) == 2
    assert results.find_one({"ticket_id": first})["outputs"] == {}
    assert ab_test.answered(results.find_one({"ticket_id": second}))
    assert results.find_one({"ticket_id": first})["claim"]["worker"] == "other"
    assert status(collections, run_id) == "partial"


def test_an_output_is_not_saved_over_a_worker_that_took_the_ticket(collections):
    _, runs, results = collections
    run_id = new_run(collections, size=1)

    def taken_over():
        results.update_one(
            {"run_id": run_id},
            {"$set": {"claim": {"worker": "other", "at": datetime.now(timezone.utc)}}},
        )

    engine = FakeEngine(during=taken_over)
    run(collections, run_id, classify=engine, worker="me")
    assert len(engine.calls) == 1  # stopped at the first lost save
    doc = results.find_one({"run_id": run_id})
    assert doc["outputs"] == {} and doc["claim"]["worker"] == "other"


def test_a_changed_prompt_stops_a_resume_but_not_a_fresh_run(collections, monkeypatch):
    _, runs, results = collections
    started = new_run(collections, size=2)
    first = ticket_ids(collections, started)[0]
    results.update_one(
        {"ticket_id": first, "run_id": started},
        {"$set": {"outputs.A": ab_test._output(answer("flag_solve"))}},
    )
    fresh = new_run(collections, size=2, seed=2)
    monkeypatch.setattr(ab_test, "PROMPT_VERSION", "new-prompt")

    with pytest.raises(ab_test.PromptChanged, match="create a new run"):
        run(collections, started)
    run(collections, fresh)
    assert runs.find_one({"_id": fresh})["prompt_version"] == "new-prompt"
    assert status(collections, fresh) == "complete"


def test_cost_covers_every_model_call():
    calls = [
        {
            "model": "p",
            "usage": {"output_tokens": 4},
            "latency_ms": 100,
            "error": "EngineError",
        },
        {"model": "s", "usage": {"output_tokens": 6}, "latency_ms": 300, "error": None},
        {
            "model": "c",
            "usage": {"output_tokens": 10},
            "latency_ms": 900,
            "error": None,
        },
    ]
    output = ab_test._output(
        answer(
            "flag_solve",
            model_calls=calls,
            confirmation_of={
                "model_version": "s",
                "classification": "manual_triage",
                "confidence": 40,
            },
        )
    )
    assert output["confirmed_from"] == {
        "model_version": "s",
        "classification": "manual_triage",
        "confidence": 40,
    }
    assert output["usage"] == {"output_tokens": 20}
    assert output["latency_ms"] == 1300 and output["model_calls"] == 3

    error = EngineError("bad reply", "s")
    error.model_calls = calls[:2]
    failed = ab_test._error_output(error)
    assert failed["usage"] == {"output_tokens": 10} and failed["latency_ms"] == 400


# --- reviewing -----------------------------------------------------------------------


def test_preferences_are_tallied_overall_and_where_the_variants_disagree(collections):
    _, runs, results = collections
    run_id = new_run(collections, size=3)
    run(collections, run_id)
    ids = ticket_ids(collections, run_id)
    ab_test.record_preference(results, run_id, ids[0], "B", 0, "lead", "B solved it")
    ab_test.record_preference(results, run_id, ids[1], "tie", 0)
    version = ab_test.record_preference(results, run_id, ids[2], "A", 0)
    ab_test.record_preference(results, run_id, ids[2], None, version)  # cleared

    summary = ab_test.summarize(runs.find_one({"_id": run_id}), list(results.find()))
    assert summary["reviewed"] == 2
    assert summary["preferences"] == {"B": 1, "tie": 1}
    assert summary["disagreement_preferences"] == {"B": 1, "tie": 1}
    saved = results.find_one({"ticket_id": ids[0]})["preference"]
    assert saved["reviewer"] == "lead" and saved["note"] == "B solved it"

    text = ab_test.format_summary(summary)
    assert "A: primary current-model" in text and "B: primary new-model" in text
    assert "effort high" in text
    assert "Same classification: 0 of 3" in text


def test_a_second_reviewer_cannot_silently_replace_a_verdict(collections):
    _, runs, results = collections
    run_id = new_run(collections, size=1)
    ticket_id = ticket_ids(collections, run_id)[0]
    # Both reviewers loaded the ticket at version 0.
    ab_test.record_preference(results, run_id, ticket_id, "A", 0, "first", "A is right")
    with pytest.raises(ab_test.VerdictConflict, match="changed by first"):
        ab_test.record_preference(results, run_id, ticket_id, "B", 0, "second")
    saved = results.find_one({"ticket_id": ticket_id})
    assert saved["preference"]["note"] == "A is right"
    assert saved["preference_version"] == 1


def test_record_preference_rejects_unknown_choices_and_tickets(collections):
    _, runs, results = collections
    run_id = new_run(collections, size=1)
    ticket_id = results.find_one()["ticket_id"]
    with pytest.raises(ValueError, match="choice"):
        ab_test.record_preference(results, run_id, ticket_id, "C", 0)
    with pytest.raises(ValueError, match="isn't in"):
        ab_test.record_preference(results, run_id, "no-such-ticket", "A", 0)


def test_blind_choice_follows_the_ticket_order():
    doc = {"order": ["B", "A"]}
    assert ab_test.blind_choice(doc, 1) == "B" and ab_test.blind_choice(doc, 2) == "A"


# --- command line ------------------------------------------------------------------------


def test_b_starts_from_a_and_overrides_only_what_is_given():
    parser = argparse.ArgumentParser()
    ab_test._config_args(parser, "A")
    ab_test._config_args(parser, "B")
    args = parser.parse_args(["--b-primary", "new-model", "--b-confirmation", ""])
    a = ab_test._config_from(args, "A", A)
    b = ab_test._config_from(args, "B", a)
    assert a == A
    assert b == EngineConfig("new-model", confirmation=None)


def test_a_zero_token_limit_is_refused_not_replaced():
    parser = argparse.ArgumentParser()
    ab_test._config_args(parser, "B")
    args = parser.parse_args(["--b-max-tokens", "0"])
    with pytest.raises(ValueError, match="max_tokens"):
        ab_test._config_from(args, "B", A)


def test_decision_model_answers_are_kept_and_counted(collections):
    _, runs, results = collections
    run_id = new_run(collections, size=2)

    def engine(payload, config=None, **kwargs):
        if config.primary == "new-model":
            return answer(
                "flag_solve",
                confidence=82,
                decision_only=True,
                probabilities={"flag_solve": 0.82, "manual_triage": 0.18},
                internal_note=None,
                reasoning="Classification only.",
            )
        return answer("manual_triage")

    summary = run(collections, run_id, classify=engine)
    output = results.find_one({"run_id": run_id})["outputs"]["B"]
    assert output["decision_only"] is True
    assert output["probabilities"] == {"flag_solve": 0.82, "manual_triage": 0.18}
    assert summary["variants"]["B"]["classification_only"] == 2
    assert summary["variants"]["A"]["classification_only"] == 0
    assert "Classification only" in ab_test.format_summary(summary)


def test_a_changed_decision_question_stops_a_resume(collections, monkeypatch):
    _, runs, results = collections
    run_id = new_run(collections, size=1)
    results.update_one(
        {"run_id": run_id},
        {"$set": {"outputs.A": ab_test._output(answer("flag_solve"))}},
    )
    monkeypatch.setattr(ab_test.decisions, "INSTRUCTIONS_VERSION", "new-question")
    with pytest.raises(ab_test.PromptChanged, match="new-question"):
        run(collections, run_id)


# --- failures keep their raw reply; retries are capped ------------------------------


def test_a_failure_keeps_its_raw_reply(collections):
    _, runs, results = collections
    run_id = new_run(collections, size=1)
    error = EngineError(
        "bad reply",
        "new-model",
        {
            "raw_response": "not json " * 5000,
            "stop_reason": "end_turn",
            "request_id": "r1",
        },
    )
    engine = FakeEngine(fail={("", "new-model"): error})
    run(collections, run_id, classify=engine)
    output = results.find_one({"run_id": run_id})["outputs"]["B"]
    assert output["raw_response"].startswith("not json")
    assert len(output["raw_response"]) == ab_test.RAW_REPLY_LIMIT
    assert output["stop_reason"] == "end_turn" and output["request_id"] == "r1"


def test_an_api_error_keeps_its_error_body():
    import anthropic
    import httpx2

    request = httpx2.Request("POST", "https://test")
    error = anthropic.APIStatusError(
        "HTTP 400",
        response=httpx2.Response(400, request=request),
        body={"error": {"message": "context too long"}},
    )
    output = ab_test._error_output(error)
    assert output["error"].startswith("APIStatusError")
    assert json.loads(output["raw_response"]) == {
        "error": {"message": "context too long"}
    }


def test_a_failing_variant_is_retried_at_most_max_attempts_times(collections):
    _, runs, results = collections
    run_id = new_run(collections, size=1)
    error = EngineError("bad reply", "new-model")
    engine = FakeEngine(fail={("", "new-model"): error})
    for attempt in range(1, ab_test.MAX_ATTEMPTS + 2):
        run(collections, run_id, classify=engine)
    b_calls = [c for _, c in engine.calls if c.primary == "new-model"]
    assert len(b_calls) == ab_test.MAX_ATTEMPTS
    doc = results.find_one({"run_id": run_id})
    assert doc["outputs"]["B"]["attempts"] == ab_test.MAX_ATTEMPTS
    assert ab_test.gave_up(doc) == ["B"] and ab_test.pending_variants(doc) == []
    # Nothing left to try, so the run is complete (with the failure counted).
    assert status(collections, run_id) == "complete"


# --- reusing saved answers -------------------------------------------------------------


def test_a_second_run_reuses_saved_answers_instead_of_calling(collections):
    _, runs, results = collections
    first = new_run(collections, size=3, seed=1)
    run(collections, first)
    second = new_run(collections, size=3, seed=1)  # the same tickets
    engine = FakeEngine()
    run(collections, second, classify=engine)

    assert engine.calls == []
    doc = results.find_one({"run_id": second})
    assert doc["outputs"]["B"]["reused_from"]["run_id"] == first
    assert doc["outputs"]["B"]["classification"] == "flag_solve"
    assert status(collections, second) == "complete"


def test_reuse_needs_the_same_content_configuration_and_moment(collections):
    tickets, runs, results = collections
    first = new_run(collections, size=1)
    run(collections, first)
    ticket_id = ticket_ids(collections, first)[0]

    def calls_for_new_run(**kwargs):
        run_id = ab_test.create_run(
            tickets,
            runs,
            results,
            kwargs.pop("a", A),
            kwargs.pop("b", B),
            [ticket_id],
            **kwargs,
        )
        engine = FakeEngine()
        run(collections, run_id, classify=engine)
        return run_id, len(engine.calls)

    # Another configuration for B: only B is asked.
    other_b = EngineConfig("new-model", max_tokens=9000)
    assert calls_for_new_run(b=other_b)[1] == 1
    # Reuse switched off: both are asked.
    assert calls_for_new_run(reuse_answers=False)[1] == 2
    # The ticket changed: both are asked.
    tickets.update_one({"zendesk_id": ticket_id}, {"$set": {"status": "pending"}})
    run_id, calls = calls_for_new_run()
    assert calls == 2
    # Too long after the saved answers: both are asked.
    results.update_many(
        {"run_id": {"$ne": run_id}},
        {"$set": {"as_of": datetime.now(timezone.utc) - timedelta(days=3)}},
    )
    results.update_one(
        {"run_id": run_id},
        {"$set": {"as_of": datetime.now(timezone.utc) - timedelta(days=3)}},
    )
    assert calls_for_new_run()[1] == 2


# --- finalizing --------------------------------------------------------------------------


def judged_run(collections, size=2):
    _, runs, results = collections
    run_id = new_run(collections, size=size)
    run(collections, run_id)
    for ticket_id in ticket_ids(collections, run_id):
        ab_test.record_preference(results, run_id, ticket_id, "B", 0, "lead", "better")
    return run_id


def test_finalize_needs_every_answered_ticket_judged(collections):
    tickets, runs, results = collections
    run_id = new_run(collections, size=2)
    with pytest.raises(ab_test.NotReadyToFinalize):
        ab_test.finalize(tickets, runs, results, run_id)
    run(collections, run_id)
    first = ticket_ids(collections, run_id)[0]
    ab_test.record_preference(results, run_id, first, "A", 0)
    assert not ab_test.ready_to_finalize(
        runs.find_one({"_id": run_id}), list(results.find({"run_id": run_id}))
    )
    with pytest.raises(ab_test.NotReadyToFinalize):
        ab_test.finalize(tickets, runs, results, run_id)


def test_finalize_saves_verdicts_on_tickets_and_slims_the_results(collections):
    tickets, runs, results = collections
    run_id = judged_run(collections)
    outcome = ab_test.finalize(tickets, runs, results, run_id)
    assert outcome == {"tickets_updated": 2, "tickets_missing": []}

    for ticket_id in ticket_ids(collections, run_id):
        (verdict,) = tickets.find_one({"zendesk_id": ticket_id})["ab_test_verdicts"]
        assert verdict["run_id"] == run_id and verdict["choice"] == "B"
        assert verdict["preferred_config"]["primary"] == "new-model"
        assert verdict["reviewer"] == "lead" and verdict["note"] == "better"
        assert verdict["variants"]["A"]["classification"] == "manual_triage"
        assert verdict["variants"]["B"]["config"]["effort"] == "high"

    doc = results.find_one({"run_id": run_id})
    assert "payload" not in doc and "transcript" not in doc["ticket"]
    assert doc["ticket"]["subject"] and doc["content_hash"] and doc["finalized"]
    assert doc["outputs"]["B"]["classification"] == "flag_solve"  # kept for reports
    finalized = runs.find_one({"_id": run_id})
    assert finalized["status"] == "finalized"
    assert finalized["summary"]["preferences"] == {"B": 2}

    # Repeating it adds nothing; the verdicts and run are locked.
    assert ab_test.finalize(tickets, runs, results, run_id)["tickets_updated"] == 0
    assert all(
        len(t["ab_test_verdicts"]) == 1
        for t in tickets.find({"ab_test_verdicts": {"$exists": True}})
    )
    with pytest.raises(ab_test.RunFinalized):
        ab_test.record_preference(results, run_id, doc["ticket_id"], "A", 1)
    with pytest.raises(ab_test.RunFinalized):
        run(collections, run_id)


def test_finalized_answers_are_still_reused(collections):
    tickets, runs, results = collections
    first = judged_run(collections)
    ab_test.finalize(tickets, runs, results, first)
    second = new_run(collections, size=2)
    engine = FakeEngine()
    run(collections, second, classify=engine)
    assert engine.calls == []


def test_a_deleted_ticket_is_reported_not_fatal(collections):
    tickets, runs, results = collections
    run_id = judged_run(collections)
    gone = ticket_ids(collections, run_id)[0]
    tickets.delete_one({"zendesk_id": gone})
    outcome = ab_test.finalize(tickets, runs, results, run_id)
    assert outcome == {"tickets_updated": 1, "tickets_missing": [gone]}


# --- export ------------------------------------------------------------------------------


def test_export_csv_names_both_sides_and_is_spreadsheet_safe(collections, tmp_path):
    import csv

    _, runs, results = collections
    run_id = judged_run(collections)
    results.update_many(
        {"run_id": run_id}, {"$set": {"preference.note": "=HYPERLINK(1)"}}
    )
    path = ab_test.export_run(
        runs.find_one({"_id": run_id}),
        list(results.find({"run_id": run_id})),
        tmp_path / "out.csv",
    )
    rows = list(csv.DictReader(path.open()))
    assert len(rows) == 2
    assert rows[0]["A_classification"] == "manual_triage"
    assert rows[0]["B_classification"] == "flag_solve"
    assert rows[0]["verdict"] == "B" and rows[0]["note"] == "'=HYPERLINK(1)"
    assert rows[0]["B_output_tokens"] == "5"
    assert list(tmp_path.iterdir()) == [path]  # no temporary file left


def test_export_json_keeps_everything_but_the_model_input(collections, tmp_path):
    _, runs, results = collections
    run_id = judged_run(collections)
    path = ab_test.export_run(
        runs.find_one({"_id": run_id}),
        list(results.find({"run_id": run_id})),
        tmp_path / "out.json",
    )
    data = json.loads(path.read_text())
    assert data["run"]["_id"] == run_id
    assert data["summary"]["preferences"] == {"B": 2}
    assert len(data["results"]) == 2 and "payload" not in data["results"][0]
    assert data["results"][0]["ticket"]["transcript"]  # not finalized yet
    with pytest.raises(ValueError, match=".csv or .json"):
        ab_test.export_run(data["run"], [], tmp_path / "out.txt")


# --- cost --------------------------------------------------------------------------------


def priced_run(collections, size=2, **kwargs):
    from decision_engine import pricing

    tickets, runs, results = collections
    ids = ab_test.sample_ticket_ids(tickets, size, seed=1)
    prices = {
        "current-model": pricing.Price(input=1, output=2),
        "new-model": pricing.Price(input=10, output=20, source="entered"),
    }
    return ab_test.create_run(
        tickets, runs, results, A, B, ids, prices=prices, **kwargs
    )


def test_a_run_saves_its_prices_and_costs_each_answer(collections):
    _, runs, results = collections
    run_id = priced_run(collections)
    saved = runs.find_one({"_id": run_id})["prices"]
    assert saved["new-model"] == {
        "input": 10.0,
        "output": 20.0,
        "cache_read": None,
        "cache_write": None,
        "source": "entered",
    }
    summary = run(collections, run_id)
    doc = results.find_one({"run_id": run_id})
    # 10 input and 5 output tokens per answer.
    assert doc["outputs"]["A"]["cost_usd"] == pytest.approx((10 * 1 + 5 * 2) / 1e6)
    assert doc["outputs"]["B"]["cost_usd"] == pytest.approx((10 * 10 + 5 * 20) / 1e6)
    assert doc["outputs"]["B"]["cost_basis"] == "estimated"
    assert summary["variants"]["B"]["cost_usd"] == pytest.approx(2 * 200 / 1e6)
    assert summary["variants"]["B"]["spent_usd"] == summary["variants"]["B"]["cost_usd"]
    text = ab_test.format_summary(summary)
    assert "Cost" in text and "$0.0004" in text


def test_a_billed_charge_beats_the_estimate(collections):
    run_id = priced_run(collections, size=1)

    def engine(payload, config=None, **kwargs):
        return answer(
            "flag_solve",
            model_calls=[
                {
                    "model": config.primary,
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                    "latency_ms": 1,
                    "billed_usd": "0.0042",
                }
            ],
        )

    run(collections, run_id, classify=engine)
    output = collections[2].find_one({"run_id": run_id})["outputs"]["A"]
    assert (
        output["cost_usd"] == pytest.approx(0.0042) and output["cost_basis"] == "billed"
    )


def test_reused_answers_cost_nothing_in_the_new_test(collections):
    first = priced_run(collections)
    run(collections, first)
    second = priced_run(collections)
    summary = run(collections, second, classify=FakeEngine())
    stats = summary["variants"]["B"]
    assert stats["reused"] == 2 and stats["spent_usd"] == 0
    assert stats["cost_usd"] == pytest.approx(2 * 200 / 1e6)


def test_an_unpriced_model_is_counted_as_unknown(collections):
    _, runs, results = collections
    run_id = new_run(collections, size=1)  # no prices
    summary = run(collections, run_id)
    assert summary["variants"]["A"]["cost_unknown"] == 1
    assert results.find_one({"run_id": run_id})["outputs"]["A"]["cost_usd"] is None


def test_cost_reaches_exports_and_ticket_verdicts(collections, tmp_path):
    import csv

    tickets, runs, results = collections
    run_id = priced_run(collections)
    run(collections, run_id)
    for ticket_id in ticket_ids(collections, run_id):
        ab_test.record_preference(results, run_id, ticket_id, "A", 0)
    path = ab_test.export_run(
        runs.find_one({"_id": run_id}),
        list(results.find({"run_id": run_id})),
        tmp_path / "out.csv",
    )
    row = next(csv.DictReader(path.open()))
    assert float(row["B_cost_usd"]) == pytest.approx(200 / 1e6)
    assert row["B_cost_basis"] == "estimated"
    ab_test.finalize(tickets, runs, results, run_id)
    verdict = tickets.find_one({"zendesk_id": ticket_ids(collections, run_id)[0]})
    assert verdict["ab_test_verdicts"][0]["variants"]["B"]["cost_usd"] == pytest.approx(
        200 / 1e6
    )
