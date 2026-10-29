"""The taxonomy worksheet's code checks: confidence floors, trigger IDs,
verified re-engage attempts, arbitration, and which statuses are reviewed."""

import copy
import json
from types import SimpleNamespace

import mongomock
import pytest
from decision_engine import classify, guards, persistence
from decision_engine.payload import support_comment_times
from fixtures import CATEGORY_FIXTURES
from test_decision_engine import api_error, fake_response, model_reply

STEPHEN = "384720527257"  # on the roster: Support
T1, T2, T3 = (
    "2026-10-01T10:00:00+00:00",
    "2026-10-03T10:00:00+00:00",
    "2026-10-06T10:00:00+00:00",
)


def guarded(classification, confidence=90, support_times=None, **extra):
    reply = guards.validate_reply(
        json.loads(model_reply(classification, confidence=confidence, **extra))
    )
    return guards.apply_guards(reply, support_comment_times=support_times)


# --- trigger IDs -----------------------------------------------------------


@pytest.mark.parametrize(
    "signals, expected",
    [
        (["S1"], "flag_solve"),
        (["S7: all answered"], "flag_solve"),
        (["R1", "S2"], "flag_solve"),
        (["R1"], "manual_triage"),
        (["solved"], "manual_triage"),
        (["S99"], "manual_triage"),
        (["S0"], "manual_triage"),
        ([], "manual_triage"),
    ],
)
def test_a_category_needs_one_of_its_own_trigger_ids(signals, expected):
    result = guarded("flag_solve", signals_matched=signals)
    assert result["classification"] == expected


def test_on_track_and_manual_triage_need_no_trigger_id():
    assert guarded("on_track", signals_matched=[])["classification"] == "on_track"
    assert guarded("manual_triage", signals_matched=[])["classification"] == "manual_triage"


def test_decision_models_need_no_trigger_id():
    reply = guards.validate_reply(
        json.loads(model_reply("flag_solve", signals_matched=[]))
    )
    guards.apply_guards(reply, classification_only=True)
    assert reply["classification"] == "flag_solve"


# --- confidence floors -----------------------------------------------------


@pytest.mark.parametrize(
    "classification, confidence, expected",
    [
        ("escalation_internal", 70, "escalation_internal"),
        ("escalation_internal", 69, "manual_triage"),
        ("flag_reengage", 69, "manual_triage"),
        ("flag_solve", 80, "flag_solve"),
        ("flag_solve", 79, "manual_triage"),
        ("on_track", 40, "on_track"),
    ],
)
def test_confidence_floors(classification, confidence, expected):
    assert guarded(classification, confidence)["classification"] == expected


def test_low_confidence_leadership_goes_to_the_owner_with_a_visibility_note():
    result = guarded(
        "escalate_leadership",
        79,
        signals_matched=["L6"],
        leadership_recipient="Rich Shupe",
        internal_note="Customer says they won't renew.",
    )
    assert result["classification"] == "escalation_internal"
    assert result["leadership_recipient"] is None
    assert result["internal_note"].startswith("Leadership visibility: possible leadership concern (L6)")
    assert result["internal_note"].endswith("Customer says they won't renew.")


def test_confident_leadership_is_kept():
    result = guarded("escalate_leadership", 80, signals_matched=["L8"])
    assert result["classification"] == "escalate_leadership"


# --- re-engage attempts ----------------------------------------------------


def test_two_verified_attempts_escalate_to_the_owner():
    result = guarded("flag_reengage", reengage_attempts=[T1, T2], support_times=[T1, T2, T3])
    assert result["classification"] == "escalation_internal"
    assert "Re-engage limit reached (2 attempts" in result["internal_note"]


def test_attempts_match_however_the_zone_is_written():
    result = guarded(
        "flag_reengage",
        reengage_attempts=["2026-10-01T10:00:00Z", T2],
        support_times=[T1, T2],
    )
    assert result["classification"] == "escalation_internal"


def test_unverified_attempts_are_dropped_and_dont_count():
    result = guarded("flag_reengage", reengage_attempts=[T1, T3], support_times=[T1])
    assert result["classification"] == "flag_reengage"
    assert result["reengage_attempts"] == [T1]
    assert any("Ignored 1 cited re-engage" in o for o in result["guard_overrides"])


def test_the_same_comment_cited_twice_counts_once():
    result = guarded(
        "flag_reengage",
        reengage_attempts=[T1, "2026-10-01T10:00:00Z"],
        support_times=[T1],
    )
    assert result["classification"] == "flag_reengage"
    assert result["reengage_attempts"] == [T1]


def test_a_compliant_low_confidence_leadership_answer_is_kept():
    # What the prompt asks for below the Leadership bar: the owner, citing L.
    result = guarded(
        "escalation_internal",
        75,
        signals_matched=["L6"],
        internal_note="Leadership visibility: says they won't renew.",
    )
    assert result["classification"] == "escalation_internal"


def test_attempts_are_trusted_when_support_times_are_unknown():
    result = guarded("flag_reengage", reengage_attempts=[T1, T2])
    assert result["classification"] == "escalation_internal"


def test_support_comment_times_are_public_support_comments():
    ticket = {
        "transcript": [
            {"author": STEPHEN, "timestamp": "2026-10-01T10:00:00+00:00"},
            {"author": STEPHEN, "timestamp": "2026-10-02T10:00:00+00:00", "is_internal_note": True},
            {"author": "555", "role": "requester", "timestamp": "2026-10-03T10:00:00+00:00"},
            {"author": "888", "timestamp": "2026-10-04T10:00:00+00:00"},
        ]
    }
    # Internal notes count: they re-engage internal parties.
    assert support_comment_times(ticket) == [
        "2026-10-01T10:00:00+00:00",
        "2026-10-02T10:00:00+00:00",
    ]


# --- arbitration -----------------------------------------------------------


class SequenceClient:
    """Answers per model, in order; an Exception is raised instead."""

    def __init__(self, replies: dict):
        self.replies = {m: list(r) if isinstance(r, list) else [r] for m, r in replies.items()}
        self.models = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        model = kwargs["model"]
        self.models.append(model)
        reply = self.replies[model].pop(0)
        if isinstance(reply, Exception):
            raise reply
        return fake_response(reply)


def judged(preferred):
    return json.dumps({"preferred": preferred, "reasoning": "r"})


@pytest.fixture
def engine(monkeypatch):
    def install(arbiter_model="arbiter", **replies):
        client = SequenceClient(replies)
        monkeypatch.setattr(classify, "get_client", lambda: client)
        config = classify.EngineConfig("primary", confirmation="confirmation", arbiter=arbiter_model)
        return client, config

    return install


# The first answer is below the confirmation floor but a real category.
UNSURE_OWNER = model_reply("escalation_internal", confidence=72)
CONFIRMED_REENGAGE = model_reply("flag_reengage", confidence=90)


def test_agreeing_rulings_return_the_arbiter_answer(engine):
    client, config = engine(
        primary=UNSURE_OWNER,
        confirmation=CONFIRMED_REENGAGE,
        arbiter=[judged("B"), model_reply("flag_reengage", confidence=88)],
    )
    result = classify.classify_ticket("payload", config=config)

    assert client.models == ["primary", "confirmation", "arbiter", "arbiter"]
    assert result["classification"] == "flag_reengage"
    assert result["model_version"] == "arbiter"
    assert result["arbitration"]["preferred"] == "flag_reengage"
    assert result["confirmation_of"]["classification"] == "escalation_internal"
    assert len(result["model_calls"]) == 4


def test_the_judging_call_does_not_see_the_ticket(engine):
    client, config = engine(
        primary=UNSURE_OWNER,
        confirmation=CONFIRMED_REENGAGE,
        arbiter=[judged("A"), model_reply("escalation_internal", confidence=88)],
    )
    calls = []
    original = client._create
    client.messages = SimpleNamespace(create=lambda **kw: calls.append(kw) or original(**kw))
    classify.classify_ticket("SECRET TICKET PAYLOAD", config=config)

    judge = calls[2]
    assert "SECRET TICKET PAYLOAD" not in json.dumps(judge)
    assert judge["system"][1]["text"] == classify.JUDGE_INSTRUCTIONS


def test_split_rulings_go_to_manual_triage(engine):
    _, config = engine(
        primary=UNSURE_OWNER,
        confirmation=CONFIRMED_REENGAGE,
        arbiter=[judged("A"), model_reply("flag_reengage", confidence=88)],
    )
    result = classify.classify_ticket("payload", config=config)

    assert result["classification"] == result["tag"] == "manual_triage"
    assert result["draft_message"] is None
    assert result["arbitration"]["preferred"] == "escalation_internal"
    assert result["arbitration"]["ruling"] == "flag_reengage"
    assert "Models disagreed" in result["internal_note"]


@pytest.mark.parametrize("arbiter_model", [None, "primary"])
def test_no_distinct_arbiter_sends_disagreement_to_manual_triage(engine, arbiter_model):
    client, config = engine(
        arbiter_model=arbiter_model, primary=UNSURE_OWNER, confirmation=CONFIRMED_REENGAGE
    )
    result = classify.classify_ticket("payload", config=config)

    assert client.models == ["primary", "confirmation"]
    assert result["classification"] == "manual_triage"
    assert result["arbitration"]["error"] == "no distinct arbiter model"


def test_arbiter_failure_sends_disagreement_to_manual_triage(engine):
    _, config = engine(
        primary=UNSURE_OWNER, confirmation=CONFIRMED_REENGAGE, arbiter=[api_error()]
    )
    result = classify.classify_ticket("payload", config=config)
    assert result["classification"] == "manual_triage"
    assert result["arbitration"]["error"] == "APIConnectionError"


def test_unusable_judgement_sends_disagreement_to_manual_triage(engine):
    _, config = engine(
        primary=UNSURE_OWNER, confirmation=CONFIRMED_REENGAGE, arbiter=[judged("C")]
    )
    result = classify.classify_ticket("payload", config=config)
    assert result["classification"] == "manual_triage"
    assert result["arbitration"]["error"] == "EngineError"


def test_confirmation_still_replaces_an_abstention(engine):
    client, config = engine(
        primary=model_reply("manual_triage", confidence=95),
        confirmation=CONFIRMED_REENGAGE,
    )
    result = classify.classify_ticket("payload", config=config)
    assert client.models == ["primary", "confirmation"]
    assert result["classification"] == "flag_reengage"
    assert "arbitration" not in result


def test_config_round_trips_the_arbiter():
    config = classify.EngineConfig("p", arbiter="a")
    assert classify.EngineConfig.from_dict(config.to_dict()) == config
    assert "arbiter a" in config.describe()


# --- statuses --------------------------------------------------------------


@pytest.mark.parametrize("status, reviewed", [("open", True), ("pending", True), ("hold", True), (None, True), ("solved", False), ("closed", False)])
def test_solved_and_closed_tickets_are_not_reviewed(status, reviewed):
    tickets = mongomock.MongoClient().support_db.tickets
    ticket = copy.deepcopy(CATEGORY_FIXTURES["flag_solve"])
    ticket["status"] = status
    tickets.insert_one(ticket)

    assert (ticket["zendesk_id"] in persistence.awaiting_ids(tickets)) == reviewed
    claimed = persistence.claim_ticket(tickets, ticket["zendesk_id"], "w")
    assert (claimed is not None) == reviewed


# --- labelling and the roster ----------------------------------------------


def reviewed(zendesk_id, classification, status="open"):
    return {
        "zendesk_id": zendesk_id,
        "status": status,
        "subject": "s",
        "requester_id": "secret",
        "agent_reviews": [{"classification": "flag_solve"}, {"classification": classification}],
    }


def test_balanced_sample_caps_each_category_but_keeps_every_leadership_case(tmp_path):
    from decision_engine import evaluation

    tickets = mongomock.MongoClient().support_db.tickets
    tickets.insert_many(
        [reviewed(f"s{i}", "flag_solve") for i in range(5)]
        + [reviewed(f"l{i}", "escalate_leadership") for i in range(4)]
        + [reviewed("closed", "on_track", status="closed"), {"zendesk_id": "new"}]
    )
    path = tmp_path / "sample.json"

    counts = evaluation.export_balanced_sample(tickets, str(path), per_category=2, seed=1)

    assert counts["flag_solve"] == 2 and counts["escalate_leadership"] == 4
    assert counts["on_track"] == 0
    cases = json.loads(path.read_text())
    assert len(cases) == 6
    assert all("agent_reviews" not in c["ticket"] and "requester_id" not in c["ticket"] for c in cases)


def test_agreement_between_two_labellers():
    from decision_engine import evaluation

    def labelled(**labels):
        return [{"ticket": {"zendesk_id": t}, "expected": c} for t, c in labels.items()]

    a = labelled(t1="flag_solve", t2="flag_solve", t3="on_track", t4="flag_reengage")
    b = labelled(t1="flag_solve", t2="on_track", t3="on_track", t5="flag_solve")
    result = evaluation.agreement(a, b)

    assert result["compared"] == 3
    assert result["agreement"] == pytest.approx(2 / 3)
    assert result["kappa"] == pytest.approx((2 / 3 - 4 / 9) / (1 - 4 / 9))
    assert result["disagreements"] == [{"zendesk_id": "t2", "a": "flag_solve", "b": "on_track"}]
    assert "67% agree" in evaluation.format_agreement(result)
    assert evaluation.agreement(a, [])["compared"] == 0


def test_roster_review_is_due_after_a_quarter():
    from datetime import timedelta

    from decision_engine import leadership

    reviewed_on = leadership.ROSTER_REVIEWED
    assert not leadership.roster_review_due(reviewed_on + timedelta(days=92))
    assert leadership.roster_review_due(reviewed_on + timedelta(days=93))


# --- engine settings -------------------------------------------------------


def test_engine_settings_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("CHEAPER_INFERENCE_PRIMARY_MODEL_NAME", "p")
    monkeypatch.setenv("DECISION_ENGINE_MAX_TOKENS", "20000")
    monkeypatch.setenv("DECISION_ENGINE_EFFORT", "High")
    monkeypatch.setenv("DECISION_ENGINE_TIMEOUT_SECONDS", "300")
    config = classify.EngineConfig.from_env()
    assert (config.max_tokens, config.effort) == (20000, "high")
    assert classify.request_timeout() == 300.0


@pytest.mark.parametrize("value", ["", "lots", "0", "-5"])
def test_unusable_max_tokens_falls_back_to_the_default(monkeypatch, value):
    monkeypatch.setenv("DECISION_ENGINE_MAX_TOKENS", value)
    assert classify.max_tokens_setting() == classify.MAX_TOKENS == 16000


def test_unknown_effort_is_ignored(monkeypatch):
    monkeypatch.setenv("DECISION_ENGINE_EFFORT", "extreme")
    assert classify.effort_setting() is None


def test_engine_settings_are_in_the_settings_dialog():
    from core import settings

    names = {s.name for s in settings.SETTINGS}
    assert {
        "CHEAPER_INFERENCE_ARBITER_MODEL_NAME",
        "DECISION_ENGINE_MAX_TOKENS",
        "DECISION_ENGINE_EFFORT",
        "DECISION_ENGINE_TIMEOUT_SECONDS",
    } <= names


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "1e999"])
def test_non_finite_timeouts_fall_back_to_the_default(monkeypatch, value):
    monkeypatch.setenv("DECISION_ENGINE_TIMEOUT_SECONDS", value)
    assert classify.request_timeout() == classify.REQUEST_TIMEOUT_SECONDS


def test_claims_outlast_every_call_at_the_configured_timeout(monkeypatch):
    from decision_engine import ab_test

    monkeypatch.delenv("DECISION_ENGINE_TIMEOUT_SECONDS", raising=False)
    # 5 calls x 120 s x 4 tries, plus the 5-minute margin: 45 minutes.
    assert persistence.claim_ttl().total_seconds() == 45 * 60
    monkeypatch.setenv("DECISION_ENGINE_TIMEOUT_SECONDS", "10")
    assert persistence.claim_ttl() == persistence.MIN_CLAIM_TTL
    monkeypatch.delenv("DECISION_ENGINE_TIMEOUT_SECONDS")
    default_ab = ab_test.claim_ttl()

    monkeypatch.setenv("DECISION_ENGINE_TIMEOUT_SECONDS", "600")
    worst = 5 * 600 * (classify.MAX_RETRIES + 1)
    assert persistence.claim_ttl().total_seconds() > worst
    assert ab_test.claim_ttl().total_seconds() > worst > default_ab.total_seconds()


def test_a_live_claim_lasts_the_configured_lifetime(monkeypatch):
    monkeypatch.setenv("DECISION_ENGINE_TIMEOUT_SECONDS", "600")
    tickets = mongomock.MongoClient().support_db.tickets
    ticket = copy.deepcopy(CATEGORY_FIXTURES["flag_solve"])
    tickets.insert_one(ticket)

    claimed = persistence.claim_ticket(tickets, ticket["zendesk_id"], "w")
    claim = claimed["review_claim"]
    held = claim["expires_at"] - claim["claimed_at"]
    assert held == persistence.claim_ttl()


@pytest.mark.parametrize(
    "name, value, read, default",
    [
        ("DECISION_ENGINE_MAX_TOKENS", "128001", "max_tokens_setting", 16000),
        ("DECISION_ENGINE_MAX_TOKENS", "1" + "0" * 400, "max_tokens_setting", 16000),
        ("DECISION_ENGINE_TIMEOUT_SECONDS", "3601", "request_timeout", 120.0),
        ("DECISION_ENGINE_TIMEOUT_SECONDS", "1e300", "request_timeout", 120.0),
    ],
)
def test_oversized_settings_fall_back_to_the_default(monkeypatch, name, value, read, default):
    monkeypatch.setenv(name, value)
    assert getattr(classify, read)() == default


def test_settings_at_their_limits_are_kept_and_claims_still_work(monkeypatch):
    from decision_engine import ab_test

    monkeypatch.setenv("DECISION_ENGINE_MAX_TOKENS", "128000")
    monkeypatch.setenv("DECISION_ENGINE_TIMEOUT_SECONDS", "3600")
    assert classify.max_tokens_setting() == 128000
    assert classify.request_timeout() == 3600.0
    worst = 5 * 3600 * (classify.MAX_RETRIES + 1)
    assert persistence.claim_ttl().total_seconds() > worst
    assert ab_test.claim_ttl().total_seconds() > worst


def test_a_lowered_timeout_does_not_cut_short_an_ab_claim(monkeypatch):
    from datetime import datetime, timedelta, timezone

    from decision_engine import ab_test

    results = mongomock.MongoClient().db.results
    results.insert_one({"_id": 1, "claim": None})
    monkeypatch.setenv("DECISION_ENGINE_TIMEOUT_SECONDS", "3600")
    assert ab_test._claim(results, 1, "slow")
    taken = results.find_one({"_id": 1})["claim"]
    assert ab_test._aware(taken["expires_at"]) - ab_test._aware(taken["at"]) == ab_test.claim_ttl()

    # An hour later, another worker runs with a much shorter timeout.
    monkeypatch.setenv("DECISION_ENGINE_TIMEOUT_SECONDS", "10")
    later = datetime.now(timezone.utc) + timedelta(hours=1)
    monkeypatch.setattr(ab_test, "_now", lambda: later)
    assert ab_test._claim(results, 1, "fast") is None
    assert results.find_one({"_id": 1})["claim"]["worker"] == "slow"
    assert ab_test.interrupted({"started": {"A": {"worker": "slow"}}, "claim": taken}) == []

    # Past its own expiry, it can be taken over.
    monkeypatch.setattr(ab_test, "_now", lambda: later + timedelta(days=1))
    assert ab_test._claim(results, 1, "fast")


def test_a_lowered_timeout_does_not_cut_short_an_answer_lock(monkeypatch):
    from datetime import datetime, timedelta, timezone

    from decision_engine import ab_test

    results = mongomock.MongoClient().db.results
    doc = {"content_hash": "h"}
    monkeypatch.setenv("DECISION_ENGINE_TIMEOUT_SECONDS", "3600")
    assert ab_test._lock_answer(results, doc, "k", "slow")

    monkeypatch.setenv("DECISION_ENGINE_TIMEOUT_SECONDS", "10")
    later = datetime.now(timezone.utc) + timedelta(hours=1)
    monkeypatch.setattr(ab_test, "_now", lambda: later)
    assert not ab_test._lock_answer(results, doc, "k", "fast")


def test_claims_saved_before_expiry_was_recorded_still_expire_by_age(monkeypatch):
    from datetime import datetime, timedelta, timezone

    from decision_engine import ab_test

    monkeypatch.delenv("DECISION_ENGINE_TIMEOUT_SECONDS", raising=False)
    results = mongomock.MongoClient().db.results
    old = datetime.now(timezone.utc) - ab_test.claim_ttl() - timedelta(minutes=1)
    results.insert_one({"_id": 1, "claim": {"worker": "gone", "at": old}})
    assert ab_test._claim(results, 1, "new")
