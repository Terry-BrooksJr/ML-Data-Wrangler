import json
from types import SimpleNamespace

import anthropic
import pytest
from decision_engine import classify, decisions
from decision_engine.guards import CLASSIFICATIONS
from test_decision_engine import model_reply, status_error

DECISION = "jev-1.13.0"


def decision_reply(choice, confidence=0.9, **extra):
    reply = {
        "answers": [
            {
                "type": "choice",
                "name": "classification",
                "choice": choice,
                "confidence": confidence,
                "probabilities": [
                    {"value": choice, "probability": confidence},
                    {"value": "manual_triage", "probability": round(1 - confidence, 2)},
                ],
            }
        ],
        "model": DECISION,
        "usage": {
            "input_tokens": 322,
            "output_tokens": 33,
            "input_tokens_details": {"cached_tokens": 12, "cache_write_tokens": 0},
        },
        "request_id": "req-decision",
        "billing": {"billed_cost_usd": "0.000008"},
    }
    reply.update(extra)
    return reply


class Client:
    """Answers GET /v1/models, POST /v1/decisions and messages.create."""

    def __init__(self, decision=None, text=None, models_error=None):
        self.decision = decision
        self.text = text or {}
        self.models_error = models_error
        self.posts = []
        self.models = []
        self.listings = 0
        self.messages = SimpleNamespace(create=self._create)

    def get(self, path, cast_to=None):
        assert path == "/v1/models"
        self.listings += 1
        if self.models_error:
            raise self.models_error
        return {
            "data": [
                {"id": DECISION, "type": "decision"},
                {"id": "text-model", "type": "text"},
                {"id": "checker", "type": "text"},
            ]
        }

    def post(self, path, cast_to=None, body=None):
        assert path == "/v1/decisions"
        self.posts.append(body)
        self.models.append(body["model"])
        if isinstance(self.decision, Exception):
            raise self.decision
        return self.decision

    def _create(self, **kwargs):
        from test_decision_engine import fake_response

        self.models.append(kwargs["model"])
        return fake_response(self.text[kwargs["model"]])


@pytest.fixture(autouse=True)
def fresh_model_list():
    decisions.reset()
    yield
    decisions.reset()


@pytest.fixture
def client(monkeypatch):
    def install(**kwargs):
        fake = Client(**kwargs)
        monkeypatch.setattr(classify, "get_client", lambda: fake)
        return fake

    return install


def test_decide_asks_one_choice_question_over_the_categories(client):
    fake = client(decision=decision_reply("flag_solve", 0.82))
    result = decisions.decide(fake, DECISION, "the payload")

    body = fake.posts[0]
    assert body["model"] == DECISION and body["input"] == "the payload"
    (question,) = body["questions"]
    assert question["type"] == "choice" and question["name"] == "classification"
    assert [c["value"] for c in question["choices"]] == list(CLASSIFICATIONS)
    assert "customer_text" in question["instructions"]

    assert result["classification"] == "flag_solve" and result["confidence"] == 82
    assert result["decision_only"] is True
    assert result["probabilities"] == {"flag_solve": 0.82, "manual_triage": 0.18}
    assert result["internal_note"] is None and result["draft_message"] is None
    assert result["model_version"] == DECISION
    assert DECISION not in result["reasoning"]  # A/B reviews are blind
    assert result["audit"]["usage"] == {
        "input_tokens": 322,
        "output_tokens": 33,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 12,
    }
    assert result["audit"]["request_id"] == "req-decision"
    assert result["audit"]["prompt_version"] == decisions.INSTRUCTIONS_VERSION
    assert json.loads(result["audit"]["raw_response"])["model"] == DECISION


def test_code_checks_apply_except_the_draft_rule(client):
    fake = client(decision=decision_reply("flag_reengage", 0.9))
    kept = decisions.decide(fake, DECISION, "p")
    assert kept["classification"] == "flag_reengage"

    over_limit = decisions.decide(fake, DECISION, "p", reengage_count=2)
    assert over_limit["classification"] == "escalation_internal"

    fake.decision = decision_reply("flag_solve", 0.6)
    assert decisions.decide(fake, DECISION, "p")["classification"] == "manual_triage"

    fake.decision = decision_reply("escalate_leadership", 0.95)
    routed = decisions.decide(fake, DECISION, "p", assignee="384720527257")
    assert routed["leadership_recipient"] == "Allain Dollete, Rich Shupe"


@pytest.mark.parametrize(
    "reply",
    [
        {},
        {"answers": []},
        decision_reply("not_a_category"),
        decision_reply("flag_solve", confidence=1.5),
        {
            "answers": [
                {"name": "classification", "choice": "flag_solve", "confidence": "high"}
            ]
        },
        {
            "answers": [
                {"name": "classification", "choice": "flag_solve", "confidence": True}
            ]
        },
    ],
)
def test_an_unusable_decision_reply_is_an_engine_error(client, reply):
    fake = client(decision=reply)
    with pytest.raises(
        classify.EngineError, match="no usable classification"
    ) as raised:
        decisions.decide(fake, DECISION, "p")
    assert "raw_response" in raised.value.audit


def test_the_engine_refuses_a_decision_model_unless_allowed(client, monkeypatch):
    monkeypatch.setenv("CHEAPER_INFERENCE_PRIMARY_MODEL_NAME", DECISION)
    fake = client(decision=decision_reply("flag_solve"))
    with pytest.raises(decisions.DecisionModelNotAllowed, match="A/B test") as raised:
        classify.classify_ticket("p")
    assert classify.is_fatal(
        raised.value
    )  # a batch stops instead of erroring per ticket
    assert fake.posts == []

    result = classify.classify_ticket("p", allow_decision_models=True)
    assert result["decision_only"] and result["classification"] == "flag_solve"
    assert result["model_calls"][0]["model"] == DECISION


def test_a_low_confidence_decision_is_confirmed_by_a_text_model(client):
    fake = client(
        decision=decision_reply("manual_triage", 0.5),
        text={"checker": model_reply("flag_solve", confidence=90)},
    )
    config = classify.EngineConfig(DECISION, confirmation="checker")
    result = classify.classify_ticket("p", config=config, allow_decision_models=True)
    assert fake.models == [DECISION, "checker"]
    assert result["classification"] == "flag_solve" and "decision_only" not in result
    assert result["confirmation_of"]["model_version"] == DECISION


def test_text_models_still_use_messages(client):
    fake = client(text={"text-model": model_reply("flag_solve")})
    config = classify.EngineConfig("text-model")
    assert (
        classify.classify_ticket("p", config=config)["classification"] == "flag_solve"
    )
    assert fake.posts == [] and fake.models == ["text-model"]


def test_a_fatal_decision_error_stops(client):
    client(decision=status_error(402))
    config = classify.EngineConfig(DECISION)
    with pytest.raises(anthropic.APIStatusError):
        classify.classify_ticket("p", config=config, allow_decision_models=True)


def test_the_model_list_is_read_once_and_a_failure_is_not_cached(client):
    fake = client(models_error=RuntimeError("down"))
    assert decisions.is_decision_model(fake, DECISION) is False
    fake.models_error = None
    assert decisions.is_decision_model(fake, DECISION) is True
    assert decisions.is_decision_model(fake, "text-model") is False
    assert fake.listings == 2
