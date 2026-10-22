"""The Tag Normalizer's language-model option (tag_llm), with a fake client."""

import json
import os
import time
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import embeddings
import numpy as np
import pytest
import tag_llm
import tag_normalizer
import tag_tab
from PyQt5.QtWidgets import QApplication, QFileDialog
from tag_normalizer import collect, LLM, mapping
from tag_tab import TagTab

# The reference's own worked example (section 6).
EXAMPLE_INPUT = [
    "billing_issue",
    "payment_problem",
    "Call-Followup",
    "post_call_followup",
    "reopened",
    "troubleshooting_scoring",
    "client_education_scoring",
    "intergration_lti",
    "integration_lti",
    "product_escalations",
    "csm_escalated",
    "feature_enablement_math",
]
EXAMPLE_MAPPING = {
    "billing_issue": "billing_issue",
    "payment_problem": "billing_issue",
    "Call-Followup": "call_followup",
    "post_call_followup": "call_followup",
    "reopened": "re-opened",
    "troubleshooting_scoring": "troubleshooting_scoring",
    "client_education_scoring": "client_education_scoring",
    "intergration_lti": "intergration_lti",
    "integration_lti": "intergration_lti",
    "product_escalations": "product_escalation",
    "csm_escalated": "csm_escalated",
    "feature_enablement_math": "feature_enablement_math",
}
EXAMPLE_REPLY = {
    "mapping": EXAMPLE_MAPPING,
    "groups": [],
    "decisions": [
        {
            "tag": "payment_problem",
            "canonical": "billing_issue",
            "action": "merge",
            "rules": ["R4"],
            "confidence": 0.9,
            "reason": "Same underlying issue.",
        },
        {
            "tag": "post_call_followup",
            "canonical": "call_followup",
            "action": "merge",
            "rules": ["R4"],
            "confidence": 0.85,
            "reason": "Cluster C1.",
        },
    ],
    "flags": [
        {
            "tags": ["intergration_", "integration_"],
            "type": "registry_misspelling",
            "suggestion": "Rename the registry prefix intergration_ to integration_ and retag.",
            "rules": ["R8"],
        },
    ],
}


class FakeClient:
    """Stands in for anthropic.Anthropic: messages.stream(...) returns `reply` as text."""

    def __init__(self, reply, stop_reason="end_turn", wrap=True):
        self.reply, self.stop_reason, self.wrap = reply, stop_reason, wrap
        self.calls = []
        self.messages = self

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        text = self.reply if isinstance(self.reply, str) else json.dumps(self.reply)
        if self.wrap:
            text = f"Here is the result:\n```json\n{text}\n```"
        message = SimpleNamespace(
            content=[
                SimpleNamespace(type="thinking", thinking=""),
                SimpleNamespace(type="text", text=text),
            ],
            stop_reason=self.stop_reason,
            usage=SimpleNamespace(
                input_tokens=100,
                output_tokens=50,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=90,
            ),
        )
        return _Ctx(SimpleNamespace(get_final_message=lambda: message))


class _Ctx:
    def __init__(self, stream):
        self.stream = stream

    def __enter__(self):
        return self.stream

    def __exit__(self, *exc):
        return False


def fake_voyage(texts):
    """Every tag on its own axis, except billing/payment, which point the same way."""
    same = {"billing issue", "payment problem"}
    vectors = np.zeros((len(texts), len(texts) + 1))
    for i, text in enumerate(texts):
        vectors[i, 0 if text in same else i + 1] = 1.0
    return vectors


# --- the reference ------------------------------------------------------------------


def test_protected_tags_come_from_the_registry():
    protected = tag_llm.protected_tags()
    # automated: true, and automated: pending
    assert {
        "agent_ooo",
        "re-opened",
        "central_2fa_reset",
        "closed",
        "client_churn",
        "support_vp_escalation",
    } <= protected
    # automated: false / unknown, and placeholders for names lost in the source
    assert not {"call_followup", "csm_escalated", "post_call_followup"} & protected
    assert not any("<" in tag for tag in protected)


def test_request_sends_every_spelling_with_its_count():
    text = tag_llm.request_text(
        collect(["Call-Followup", "call_followup", "Call-Followup"])
    )
    items = json.loads(text[text.index("[") :])
    assert items == [
        {"tag": "Call-Followup", "count": 2},
        {"tag": "call_followup", "count": 1},
    ]


# --- a round trip -------------------------------------------------------------------


def test_worked_example_round_trip(monkeypatch):
    monkeypatch.setenv("TAG_NORMALIZER_MODEL", "tag-model-x")
    client = FakeClient(EXAMPLE_REPLY)
    result = tag_llm.normalize(collect(EXAMPLE_INPUT), client=client)

    assert mapping(result.groups) == EXAMPLE_MAPPING
    assert result.model == "tag-model-x"
    assert result.usage["cache_read_input_tokens"] == 90
    (call,) = client.calls
    assert call["model"] == "tag-model-x"
    # The reference, then the priorities that override it; both cached.
    assert call["system"] == [
        {"type": "text", "text": tag_llm.reference()},
        {
            "type": "text",
            "text": tag_llm.priorities(),
            "cache_control": {"type": "ephemeral"},
        },
    ]
    billing = next(g for g in result.groups if g.canonical == "billing_issue")
    assert [t.spelling for t in billing.members] == ["billing_issue", "payment_problem"]
    assert billing.similarity == [1.0, 0.9]
    assert result.flags == EXAMPLE_REPLY["flags"]


def test_model_defaults_to_the_primary_model(monkeypatch):
    monkeypatch.delenv("TAG_NORMALIZER_MODEL", raising=False)
    monkeypatch.setenv("CHEAPER_INFERENCE_PRIMARY_MODEL_NAME", "primary-y")
    assert tag_llm.model_name() == "primary-y"


# --- checking the reply -------------------------------------------------------------


def check(reply_mapping, raw_tags, decisions=()):
    return tag_llm.check_reply(
        {"mapping": reply_mapping, "decisions": list(decisions)}, collect(raw_tags)
    )


def test_missing_tag_is_kept_and_flagged():
    result = check({"a": "a"}, ["a", "b"])
    assert mapping(result.groups) == {"a": "a", "b": "b"}
    assert [f["type"] for f in result.flags] == ["missing_from_reply"]


def test_protected_tag_is_never_renamed():
    result = check(
        {"closed_no_response": "closed", "reopened": "re-opened"},
        ["closed_no_response", "reopened"],
    )
    mapped = mapping(result.groups)
    assert mapped["closed_no_response"] == "closed_no_response"  # automated: R3
    assert mapped["reopened"] == "re-opened"  # merging *into* a protected tag is fine
    (flag,) = result.flags
    assert flag["type"] == "protected_tag" and flag["rules"] == ["R3"]


def test_low_confidence_merge_is_kept_apart():
    result = check(
        {"custom": "customization", "customization": "customization"},
        ["custom", "customization"],
        [
            {
                "tag": "custom",
                "canonical": "customization",
                "action": "merge",
                "confidence": 0.6,
                "reason": "Cluster C5.",
            }
        ],
    )
    assert mapping(result.groups)["custom"] == "custom"
    (flag,) = result.flags
    assert flag["type"] == "possible_synonym" and "0.60" in flag["suggestion"]


def test_tags_not_in_the_list_are_ignored_with_a_warning():
    result = check({"a": "a", "zzz": "a"}, ["a"])
    assert mapping(result.groups) == {"a": "a"}
    assert "zzz" in result.warnings[0]


def test_format_only_renames_keep_their_spelling():
    raw = [
        "sentiment__positive",
        "self-hosted",
        "oil_change",
        "tire rotation",
        "car_care",
    ]
    result = check(
        {
            # Only the form changes, and nothing else maps there: undone.
            "sentiment__positive": "sentiment_positive",
            "self-hosted": "self_hosted",
            # Different words for one concept, under a new tag: kept.
            "oil_change": "vehicle_maintenance",
            "tire rotation": "vehicle_maintenance",
            "car_care": "vehicle_maintenance",
        },
        raw,
        [{"tag": "sentiment__positive", "action": "normalize_form", "confidence": 1.0}],
    )
    mapped = mapping(result.groups)
    assert mapped["sentiment__positive"] == "sentiment__positive"
    assert mapped["self-hosted"] == "self-hosted"
    assert {mapped[t] for t in ("oil_change", "tire rotation", "car_care")} == {
        "vehicle_maintenance"
    }
    assert result.decisions == []  # the undone rename's decision is dropped
    assert "2 tag(s)" in result.warnings[0]


def test_format_changes_that_merge_or_reach_the_registry_are_kept():
    result = check(
        {
            "Call-Followup": "call_followup",
            "call followups": "call_followup",
            "reopened": "re-opened",
        },
        ["Call-Followup", "call followups", "reopened"],
    )
    assert set(mapping(result.groups).values()) == {"call_followup", "re-opened"}
    assert result.warnings == []


def test_reply_without_mapping_is_an_error():
    with pytest.raises(tag_llm.TagModelError):
        tag_llm.check_reply({"groups": []}, collect(["a"]))


@pytest.mark.parametrize(
    "client",
    [
        FakeClient(EXAMPLE_REPLY, stop_reason="max_tokens"),
        FakeClient(EXAMPLE_REPLY, stop_reason="refusal"),
        FakeClient("no json here", wrap=False),
        FakeClient("[1, 2]", wrap=False),
    ],
)
def test_unusable_replies_raise(client):
    with pytest.raises(tag_llm.TagModelError):
        tag_llm.normalize(collect(["a"]), client=client, model="m")


def test_too_many_tags_is_refused_before_any_call():
    client = FakeClient(EXAMPLE_REPLY)
    with pytest.raises(ValueError):
        tag_llm.normalize(
            collect(f"t{i}" for i in range(tag_llm.MAX_TAGS + 1)),
            client=client,
            model="m",
        )
    assert client.calls == []


# --- command line -------------------------------------------------------------------


def test_cli_llm_backend_writes_decisions_and_flags(tmp_path, monkeypatch, capsys):
    seen = {}

    def fake_normalize(tags, candidates=()):
        seen["candidates"] = candidates
        return tag_llm.check_reply(EXAMPLE_REPLY, tags, "m")

    monkeypatch.setattr(tag_llm, "normalize", fake_normalize)
    monkeypatch.setattr(tag_normalizer, "embedder", lambda backend: fake_voyage)
    source = tmp_path / "tags.txt"
    source.write_text(", ".join(EXAMPLE_INPUT))
    out = tmp_path / "mapping.json"
    assert (
        tag_normalizer.main([str(source), "--backend", "llm", "--out", str(out)]) == 0
    )
    data = json.loads(out.read_text())
    assert data["mapping"] == EXAMPLE_MAPPING
    assert data["flags"] == EXAMPLE_REPLY["flags"] and data["model"] == "m"
    assert len(data["decisions"]) == 2
    assert "registry_misspelling" in capsys.readouterr().out
    assert [[s for s, _ in c] for c in seen["candidates"]] == [
        ["billing_issue", "payment_problem"]
    ]


def test_cli_no_candidates_skips_voyage(tmp_path, monkeypatch):
    def no_voyage(backend):
        raise AssertionError("Voyage was called")

    monkeypatch.setattr(tag_normalizer, "embedder", no_voyage)
    monkeypatch.setattr(
        tag_llm,
        "normalize",
        lambda tags, candidates=(): tag_llm.check_reply({"mapping": {}}, tags, "m"),
    )
    source = tmp_path / "tags.txt"
    source.write_text("a, b")
    assert (
        tag_normalizer.main([str(source), "--backend", "llm", "--no-candidates"]) == 0
    )


# --- the tab ------------------------------------------------------------------------


@pytest.fixture
def qapp():
    return QApplication.instance() or QApplication([])


def wait_until_idle(qapp, tab, timeout=5.0):
    deadline = time.monotonic() + timeout
    while tab.is_busy() and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    qapp.processEvents()
    assert not tab.is_busy()


def test_tab_shows_model_groups_and_flags(qapp, tmp_path, monkeypatch):
    seen = []

    def fake_normalize(tags, candidates=()):
        seen.append([[s for s, _ in c] for c in candidates])
        return tag_llm.check_reply(EXAMPLE_REPLY, tags, "m")

    embedded = []

    def factory(backend):
        assert backend == tag_normalizer.VOYAGE
        return lambda texts: embedded.extend(texts) or fake_voyage(texts)

    tab = TagTab(factory, model_normalizer=fake_normalize)
    tab.backend_combo.setCurrentIndex(tab.backend_combo.findData(LLM))
    assert not tab.threshold.isEnabled() and not tab.keep_structured_apart.isEnabled()
    tab.input.setPlainText(", ".join(EXAMPLE_INPUT))
    assert tab.start_find()
    wait_until_idle(qapp, tab)

    assert seen == [[["billing_issue", "payment_problem"]]]
    assert len(embedded) == len(EXAMPLE_INPUT)
    assert tab.mapping() == EXAMPLE_MAPPING
    assert tab.tree.headerItem().text(2) == "Confidence"
    assert (
        tab.flags_view.isVisibleTo(tab)
        and "registry_misspelling" in tab.flags_view.toPlainText()
    )
    billing = next(
        tab.tree.topLevelItem(i)
        for i in range(tab.tree.topLevelItemCount())
        if tab.tree.topLevelItem(i).text(0) == "billing_issue"
    )
    assert billing.child(1).toolTip(0) == "Same underlying issue."

    out = tmp_path / "m.json"
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", lambda *a: (str(out), "JSON (*.json)")
    )
    tab.export_mapping()
    assert json.loads(out.read_text())["flags"] == EXAMPLE_REPLY["flags"]

    # A second run reuses the cached vectors.
    tab.start_find()
    wait_until_idle(qapp, tab)
    assert len(embedded) == len(EXAMPLE_INPUT)

    # Back to embeddings: the threshold returns, and the model's notes go.
    tab.backend_combo.setCurrentIndex(tab.backend_combo.findData(tag_normalizer.SPACY))
    assert tab.threshold.isEnabled()


def test_tab_runs_without_voyage(qapp):
    def no_key(backend):
        def embed(texts):
            raise embeddings.EmbeddingNotConfigured("EMBEDDING_API_KEY is not set")

        return embed

    calls = []

    def fake_normalize(tags, candidates=()):
        calls.append(candidates)
        return tag_llm.check_reply({"mapping": {"a": "a", "b": "b"}}, tags, "m")

    tab = TagTab(no_key, model_normalizer=fake_normalize)
    tab.backend_combo.setCurrentIndex(tab.backend_combo.findData(LLM))
    tab.input.setPlainText("a, b")
    tab.start_find()
    wait_until_idle(qapp, tab)
    assert calls == [[]]
    assert "EMBEDDING_API_KEY" in tab.flags_view.toPlainText()


# --- Zendesk AI tags ----------------------------------------------------------------

AI_TAGS = [
    "intent__software__sync__issue",
    "intent_confidence__high",
    "Intent__misc__other",
]


def test_intent_tags_are_never_sent_and_stay_as_they_are():
    client = FakeClient(
        {"mapping": {"sync_problem": "sync_problem", "refund": "refund"}}
    )
    candidates = [
        [("sync_problem", 1.0), ("intent__software__sync__issue", 0.95)],
        [("refund", 1.0), ("billing_issue", 0.9)],
    ]
    result = tag_llm.normalize(
        collect(["sync_problem", "refund", "billing_issue"] + AI_TAGS),
        client=client,
        model="m",
        candidates=candidates,
    )
    (call,) = client.calls
    request = call["messages"][0]["content"]
    assert "intent" not in request.lower()
    # The candidate that held an intent tag lost it, and with one tag left, went.
    assert result.candidates == [["refund", "billing_issue"]]
    mapped = mapping(result.groups)
    assert all(mapped[t] == t for t in AI_TAGS)
    assert sorted(result.ignored) == sorted(AI_TAGS)
    assert "[skipped] 3 Zendesk AI tag(s)" in tag_llm.describe_flags(result)


def test_nothing_may_be_merged_into_an_intent_tag():
    result = check(
        {"connection_issue": "intent__software__connection__issue"},
        ["connection_issue"],
    )
    assert mapping(result.groups) == {"connection_issue": "connection_issue"}
    (flag,) = result.flags
    assert flag["type"] == "zendesk_ai_tag"


def test_candidate_pairs_closest_first_without_facet_pairs():
    tags = collect(
        [
            "sentiment__positive",
            "sentiment__negative",
            "happy_customer",
            "refund",
            "Refund",
        ]
    )
    vectors = np.array([[1.0, 0.0], [0.99, 0.1], [0.995, 0.05], [0.0, 1.0]])
    pairs = tag_llm.candidate_pairs(tags, vectors)
    names = [tuple(s for s, _ in p) for p in pairs]
    # Spellings of one tag pair at 1.0; facet values are never paired.
    assert names[0] == ("refund", "Refund")
    assert ("sentiment__positive", "sentiment__negative") not in names
    assert set(names[1:]) == {
        ("sentiment__positive", "happy_customer"),
        ("sentiment__negative", "happy_customer"),
    }
    sims = [p[1][1] for p in pairs]
    assert sims == sorted(sims, reverse=True)


def test_candidate_pairs_are_capped(monkeypatch):
    monkeypatch.setattr(tag_llm, "MAX_CANDIDATE_PAIRS", 3)
    tags = collect([f"t{i}" for i in range(10)])
    assert len(tag_llm.candidate_pairs(tags, np.ones((10, 2)))) == 3


def test_request_lists_candidate_groups():
    text = tag_llm.request_text(collect(["a", "b"]), [[("a", 1.0), ("b", 0.9)]])
    head, groups = text.rsplit("\n\n", 1)
    assert "Candidate pairs" in head
    assert json.loads(groups) == [{"tags": ["a", "b"], "similarity": 0.9}]


def test_model_results_cant_be_saved_but_embedding_results_can(qapp, tmp_path):
    def fake_normalize(tags, candidates=()):
        return tag_llm.check_reply(EXAMPLE_REPLY, tags, "m")

    tab = TagTab(
        lambda backend: fake_voyage,
        database_factory=lambda: None,
        run_dir=tmp_path,
        model_normalizer=fake_normalize,
    )
    tab.backend_combo.setCurrentIndex(tab.backend_combo.findData(LLM))
    tab.input.setPlainText(", ".join(EXAMPLE_INPUT))
    tab.start_find()
    wait_until_idle(qapp, tab)
    assert tab.export_button.isEnabled() and not tab.save_button.isEnabled()
    assert "can't be saved" in tab.save_button.toolTip()
    assert tab.save_run() is False

    tab.backend_combo.setCurrentIndex(tab.backend_combo.findData(tag_normalizer.VOYAGE))
    tab.start_find()
    wait_until_idle(qapp, tab)
    assert tab.save_button.isEnabled()
    assert tab.save_button.toolTip() == tag_tab.SAVE_TOOLTIP
