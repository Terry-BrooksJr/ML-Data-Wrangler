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
