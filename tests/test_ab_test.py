import copy

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

    def __init__(self, fail=None):
        self.calls = []
        self.fail = fail or {}  # (ticket subject, primary) -> exception

    def __call__(self, payload, reengage_count=0, assignee=None, config=None):
        self.calls.append((payload, config))
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
    return ab_test.create_run(runs, results, A, B, ids, "test run", seed=seed)


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


def test_create_run_records_variants_and_a_blind_order(collections):
    tickets, runs, results = collections
    run_id = new_run(collections, size=5)
    run = runs.find_one({"_id": run_id})
    assert run["variants"]["B"]["effort"] == "high"
    assert run["status"] == "pending" and run["ticket_count"] == 5
    docs = list(results.find({"run_id": run_id}))
    assert len(docs) == 5
    assert all(sorted(d["order"]) == ["A", "B"] for d in docs)
    assert all(d["outputs"] == {} and d["preference"] is None for d in docs)


def test_create_run_rejects_the_same_configuration_twice(collections):
    _, runs, results = collections
    with pytest.raises(ValueError, match="same configuration"):
        ab_test.create_run(runs, results, A, EngineConfig.from_dict(A.to_dict()), ["1"])
    with pytest.raises(ValueError, match="No tickets"):
        ab_test.create_run(runs, results, A, B, [])


# --- running ---------------------------------------------------------------------


def test_run_classifies_each_ticket_with_both_configurations(collections):
    tickets, runs, results = collections
    before = list(tickets.find())
    run_id = new_run(collections)
    engine = FakeEngine()
    summary = ab_test.run_ab_test(run_id, tickets, runs, results, classify=engine)

    assert (
        sorted(c.primary for _, c in engine.calls)
        == ["current-model"] * 3 + ["new-model"] * 3
    )
    assert {c for _, c in engine.calls} == {A, B}
    doc = results.find_one({"run_id": run_id})
    assert doc["outputs"]["A"]["classification"] == "manual_triage"
    assert doc["outputs"]["B"]["usage"] == {"input_tokens": 10, "output_tokens": 5}
    assert "audit" not in doc["outputs"]["B"]
    assert runs.find_one({"_id": run_id})["status"] == "complete"
    assert summary["classified"] == 3 and summary["agreement"] == 0
    # An A/B run never changes the tickets themselves.
    assert list(tickets.find()) == before


def test_run_saves_a_failure_and_carries_on(collections):
    tickets, runs, results = collections
    run_id = new_run(collections, size=len(CATEGORY_FIXTURES))
    subject = CATEGORY_FIXTURES["flag_solve"]["subject"]
    engine = FakeEngine(
        fail={(subject, "new-model"): EngineError("bad reply", "new-model")}
    )
    summary = ab_test.run_ab_test(run_id, tickets, runs, results, classify=engine)

    assert summary["variants"]["B"]["errors"] == 1
    assert summary["variants"]["A"]["errors"] == 0
    assert summary["classified"] == len(CATEGORY_FIXTURES) - 1
    failed = results.find_one({"outputs.B.error": {"$exists": True}})
    assert "bad reply" in failed["outputs"]["B"]["error"]
    assert not ab_test.answered(failed) and not ab_test.differs(failed)


def test_a_fatal_error_stops_the_run_and_it_resumes(collections):
    tickets, runs, results = collections
    run_id = new_run(collections, size=3)
    first = sorted(d["ticket_id"] for d in results.find({"run_id": run_id}))[1]
    subject = tickets.find_one({"zendesk_id": first})["subject"]
    engine = FakeEngine(fail={(subject, "new-model"): status_error(401)})
    with pytest.raises(Exception, match="401"):
        ab_test.run_ab_test(run_id, tickets, runs, results, classify=engine)
    assert runs.find_one({"_id": run_id})["status"] == "partial"
    assert len(engine.calls) == 4  # ticket 1 both, ticket 2 A then B (fails)

    resumed = FakeEngine()
    ab_test.run_ab_test(run_id, tickets, runs, results, classify=resumed)
    # Only what was missing: ticket 2's B and ticket 3's A and B.
    assert len(resumed.calls) == 3
    assert runs.find_one({"_id": run_id})["status"] == "complete"


def test_stop_saves_what_finished(collections):
    tickets, runs, results = collections
    run_id = new_run(collections, size=3)
    engine = FakeEngine()
    with pytest.raises(ab_test.Stopped):
        ab_test.run_ab_test(
            run_id,
            tickets,
            runs,
            results,
            classify=engine,
            should_stop=lambda: len(engine.calls) >= 2,
        )
    assert len(engine.calls) == 2
    assert (
        results.count_documents({"run_id": run_id, "outputs.B": {"$exists": True}}) == 1
    )
    assert runs.find_one({"_id": run_id})["status"] == "partial"


def test_a_missing_ticket_is_an_error_not_a_crash(collections):
    tickets, runs, results = collections
    run_id = ab_test.create_run(runs, results, A, B, ["no-such-ticket"])
    summary = ab_test.run_ab_test(run_id, tickets, runs, results, classify=FakeEngine())
    assert summary["variants"]["A"]["errors"] == summary["variants"]["B"]["errors"] == 1


def test_a_confirmation_keeps_the_answer_it_replaced_and_both_costs():
    output = ab_test._output(
        answer(
            "flag_solve",
            confirmation_of={
                "model_version": "first",
                "classification": "manual_triage",
                "confidence": 40,
                "audit": {"usage": {"input_tokens": 7, "output_tokens": 3}},
            },
        )
    )
    assert output["confirmed_from"] == {
        "model_version": "first",
        "classification": "manual_triage",
        "confidence": 40,
    }
    assert output["usage"] == {"input_tokens": 17, "output_tokens": 8}


# --- reviewing -----------------------------------------------------------------------


def test_preferences_are_tallied_overall_and_where_the_variants_disagree(collections):
    tickets, runs, results = collections
    run_id = new_run(collections, size=3)
    ab_test.run_ab_test(run_id, tickets, runs, results, classify=FakeEngine())
    ids = sorted(d["ticket_id"] for d in results.find({"run_id": run_id}))
    ab_test.record_preference(results, run_id, ids[0], "B", "lead", "B solved it")
    ab_test.record_preference(results, run_id, ids[1], "tie")
    ab_test.record_preference(results, run_id, ids[2], "A")
    ab_test.record_preference(results, run_id, ids[2], None)  # cleared

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


def test_record_preference_rejects_unknown_choices_and_tickets(collections):
    _, runs, results = collections
    run_id = new_run(collections, size=1)
    ticket_id = results.find_one()["ticket_id"]
    with pytest.raises(ValueError, match="choice"):
        ab_test.record_preference(results, run_id, ticket_id, "C")
    with pytest.raises(ValueError, match="isn't in"):
        ab_test.record_preference(results, run_id, "no-such-ticket", "A")


def test_blind_choice_follows_the_ticket_order():
    doc = {"order": ["B", "A"]}
    assert ab_test.blind_choice(doc, 1) == "B" and ab_test.blind_choice(doc, 2) == "A"


# --- command line ------------------------------------------------------------------------


def test_b_starts_from_a_and_overrides_only_what_is_given():
    import argparse

    parser = argparse.ArgumentParser()
    ab_test._config_args(parser, "A")
    ab_test._config_args(parser, "B")
    args = parser.parse_args(["--b-primary", "new-model", "--b-confirmation", ""])
    a = ab_test._config_from(args, "A", A)
    b = ab_test._config_from(args, "B", a)
    assert a == A
    assert b == EngineConfig("new-model", confirmation=None)
