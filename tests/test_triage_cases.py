"""Manual-triage case files, reviewer evidence and re-runs (triage.py, evidence.py)."""

import copy
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import boto3
import mongomock
import pytest
from bson import ObjectId
from moto import mock_aws

from decision_engine import classify, evidence, feedback, persistence, training_export, triage
from decision_engine.guards import InvalidReply, validate_reply
from decision_engine.prompts import PROMPT_VERSION
from fixtures import CATEGORY_FIXTURES
from test_decision_engine import FakeClient, model_reply, RoutingClient, fake_response

JIRA_XML = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="0.92">
  <channel>
    <title>Jira</title>
    <link>https://jira.example.com</link>
    <description>An XML representation of a search request</description>
    <language>en-us</language>
    <build-info><version>9.0</version><build-number>1</build-number><build-date>01-01-2026</build-date></build-info>
    <item>
      <title>[LRN-1234] Inline PDF viewer</title>
      <link>https://jira.example.com/browse/LRN-1234</link>
      <project id="1" key="LRN">Learnosity</project>
      <description>&lt;p&gt;Render PDFs &lt;b&gt;inline&lt;/b&gt;.&lt;/p&gt;&lt;p&gt;Second line&lt;/p&gt;</description>
      <environment></environment>
      <key id="99">LRN-1234</key>
      <summary>Inline PDF viewer</summary>
      <type id="1">Story</type>
      <priority id="3">Medium</priority>
      <status id="6" description="Done">Released</status>
      <statusCategory id="3" key="done" colorName="success"/>
      <resolution id="1">Fixed</resolution>
      <assignee accountid="a1">Dev Person</assignee>
      <reporter accountid="a2">Terry Brooks</reporter>
      <labels><label>cengage</label></labels>
      <created>Tue, 9 Aug 2022 14:46:21 -0500</created>
      <updated>Mon, 2 Nov 2026 09:00:00 -0500</updated>
      <votes>0</votes>
      <watches>2</watches>
      <comments>
        <comment id="10" author="dev" created="Mon, 2 Nov 2026 09:00:00 -0500">&lt;p&gt;Shipped in v2026.10&lt;/p&gt;</comment>
      </comments>
      <issuelinks>
        <issuelinktype id="5">
          <name>Relates</name>
          <outwardlinks description="relates to">
            <issuelink><issuekey id="7">LRN-1000</issuekey></issuelink>
          </outwardlinks>
        </issuelinktype>
      </issuelinks>
      <customfields>
        <customfield id="customfield_1" key="com.example:text">
          <customfieldname>Fix release</customfieldname>
          <customfieldvalues><customfieldvalue>2026.10</customfieldvalue></customfieldvalues>
        </customfield>
      </customfields>
    </item>
  </channel>
</rss>
"""

# The text pypdf reads from an Aha idea's History page printed to PDF.
AHA_TEXT = """  Build   Knowledge  Strategy  Releases  Ideas  Features  Roadmaps
Created by Terry Brooks in Enhancement Requests on Dec 12, 2025 IdeaLAPIPLAT-I-543
Future PDF Viewer support?  
Does Learnosity have any plans to support an inline PDF viewer?
5:05pmAudrey Auyang updated idea LAPIPLAT-I-543 Future PDF Viewer support?
Workflow statusScoping → Parked
6:12pmVlad Manu deleted comment on idea LAPIPLAT-I-543 Future PDF Viewer support?
Comment by Vlad Manu
Try this JSON
6:11pmVlad Manu commented on idea LAPIPLAT-I-543 Future PDF Viewer support?
Comment by Vlad Manu
Try this JSON
5:49pmAudrey Auyang created internal comment on idea LAPIPLAT-I-543 Future PDF Viewer support?
Comment by Audrey Auyang
Strategic Alignment: request from Cengage
6:14pmNina van der Behr updated idea LAPIPLAT-I-543 Future PDF Viewer support?
Workflow statusNew → Scoping Assigned to userDefault (Unassigned) → Audrey Auyang
NameFuture PDF Viewer support? Score0 WorkspaceBLD Build VisibilityVisible to our employees
Status Parked 
Assigned to
 Audrey Auyang
04 June
12 December
11/2/26, 9:13 AM LAPIPLAT-I-543 Future PDF Viewer support? | Aha!
https://example.aha.io/ideas/ideas/LAPIPLAT-I-543?active_tab=history 1/1
"""


@pytest.fixture(autouse=True)
def no_storage(monkeypatch):
    for name in ("MODEL_BUCKET", "TRIAGE_CASES_COLLECTION", "CHEAPER_INFERENCE_ARBITER_MODEL_NAME"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def tickets():
    return mongomock.MongoClient().support_db.tickets


@pytest.fixture
def cases(tickets):
    return triage.get_cases_collection(tickets.database)


def thinking_response(text, thinking):
    response = fake_response(text)
    response.content = [SimpleNamespace(type="thinking", thinking=thinking), *response.content]
    return response


class ThinkingClient(FakeClient):
    def __init__(self, text, thinking="I weighed S3 against R1."):
        super().__init__(text)
        self.thinking = thinking

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return thinking_response(self.text, self.thinking)


def manual_reply(**extra):
    return model_reply(
        "manual_triage",
        30,
        signals_matched=[],
        open_questions=["Has LRN-1234 been released?"],
        **extra,
    )


def review_manual_ticket(monkeypatch, tickets, client=None):
    ticket = copy.deepcopy(CATEGORY_FIXTURES["manual_triage"])
    tickets.insert_one(ticket)
    client = client or ThinkingClient(manual_reply())
    monkeypatch.setattr(classify, "get_client", lambda: client)
    persistence.process_one_ticket(ticket["zendesk_id"], tickets)
    return ticket["zendesk_id"]


# --- the reply and the review --------------------------------------------------


def test_open_questions_are_optional_and_checked():
    base = json.loads(model_reply("on_track"))
    base.pop("open_questions", None)
    assert validate_reply(base)["open_questions"] == []
    assert validate_reply({**base, "open_questions": [" a ", ""]})["open_questions"] == ["a"]
    with pytest.raises(InvalidReply, match="open_questions"):
        validate_reply({**base, "open_questions": "a"})


def test_thinking_is_kept_for_the_case_but_not_on_the_review(monkeypatch, tickets, cases):
    zendesk_id = review_manual_ticket(monkeypatch, tickets)

    review = tickets.find_one({"zendesk_id": zendesk_id})["agent_reviews"][0]
    assert "thinking" not in review["audit"]
    assert review["open_questions"] == ["Has LRN-1234 been released?"]
    case = cases.find_one({"review_id": review["review_id"]})
    assert case["trail"]["stored"] == "inline"
    assert case["trail"]["trail"]["first"]["thinking"] == ["I weighed S3 against R1."]


# --- opening a case -----------------------------------------------------------------


def test_a_manual_triage_review_opens_a_case(monkeypatch, tickets, cases):
    zendesk_id = review_manual_ticket(monkeypatch, tickets)

    case = cases.find_one({"zendesk_id": zendesk_id})
    assert case["status"] == triage.OPEN
    assert case["route"] == triage.ABSTAINED
    assert case["open_questions"] == ["Has LRN-1234 been released?"]
    assert case["answers"][0]["classification"] == "manual_triage"
    assert case["prompt_version"] == PROMPT_VERSION
    assert case["evidence"] == [] and case["reruns"] == []


def test_other_classifications_open_no_case(monkeypatch, tickets, cases):
    ticket = copy.deepcopy(CATEGORY_FIXTURES["escalation_internal"])
    tickets.insert_one(ticket)
    monkeypatch.setattr(classify, "get_client", lambda: FakeClient(model_reply("escalation_internal")))
    persistence.process_one_ticket(ticket["zendesk_id"], tickets)
    assert cases.count_documents({}) == 0


def test_opening_a_case_twice_keeps_one(monkeypatch, tickets, cases):
    zendesk_id = review_manual_ticket(monkeypatch, tickets)
    review = tickets.find_one({"zendesk_id": zendesk_id})["agent_reviews"][0]
    first = cases.find_one({})["_id"]
    assert triage.open_case(cases, zendesk_id, review, {"classification": "manual_triage"}) == first
    assert cases.count_documents({}) == 1


def test_a_case_failure_never_loses_the_review(monkeypatch, tickets):
    monkeypatch.setattr(triage, "open_case", lambda *a: (_ for _ in ()).throw(RuntimeError("boom")))
    zendesk_id = review_manual_ticket(monkeypatch, tickets)
    assert len(tickets.find_one({"zendesk_id": zendesk_id})["agent_reviews"]) == 1


def test_routes():
    assert triage.route_of({"engine_error": "x"}) == triage.ENGINE_ERROR
    assert (
        triage.route_of({"classification": "manual_triage", "arbitration": {"model": "a"}})
        == triage.DISAGREEMENT
    )
    assert (
        triage.route_of({"guard_overrides": ["Downgraded from flag_solve: confidence 60"]})
        == triage.GUARD_OVERRIDE
    )
    assert triage.route_of({"classification": "manual_triage"}) == triage.ABSTAINED


def test_a_disagreement_keeps_both_answers_and_the_judges_reasoning(monkeypatch, tickets, cases):
    monkeypatch.setenv("CHEAPER_INFERENCE_PRIMARY_MODEL_NAME", "primary")
    monkeypatch.setenv("CHEAPER_INFERENCE_CONFIRMATION_MODEL_NAME", "confirmation")
    monkeypatch.setenv("CHEAPER_INFERENCE_ARBITER_MODEL_NAME", "arbiter")

    class Arbiter(RoutingClient):
        def _create(self, **kwargs):
            if kwargs["model"] == "arbiter" and len(kwargs["system"]) == 2:
                self.models.append("judge")
                return fake_response(json.dumps({"preferred": "A", "reasoning": "A cites S3 properly"}))
            return super()._create(**kwargs)

    client = Arbiter(
        {
            # Above the action floor, below the confirmation floor.
            "primary": model_reply("escalation_internal", 72, signals_matched=["O1"]),
            "confirmation": model_reply("on_track", 70, signals_matched=[]),
            "arbiter": model_reply("on_track", 80, signals_matched=[]),
        }
    )
    ticket = copy.deepcopy(CATEGORY_FIXTURES["flag_solve"])
    tickets.insert_one(ticket)
    monkeypatch.setattr(classify, "get_client", lambda: client)
    persistence.process_one_ticket(ticket["zendesk_id"], tickets)

    case = cases.find_one({"zendesk_id": ticket["zendesk_id"]})
    assert case["route"] == triage.DISAGREEMENT
    first, confirmation = case["answers"]
    assert (first["role"], first["classification"]) == ("first", "escalation_internal")
    # The confirmation's own answer, not the synthetic manual triage.
    assert confirmation == {
        "role": "confirmation",
        "model": "confirmation",
        "classification": "on_track",
        "confidence": 70,
        "signals_matched": [],
        "reasoning": "test reasoning",
        "open_questions": [],
    }
    assert case["arbitration"]["preferred_reasoning"] == "A cites S3 properly"


def test_a_long_trail_without_storage_is_cut_to_fit(monkeypatch):
    monkeypatch.setattr(triage, "MAX_INLINE_TRAIL_BYTES", 20_000)
    stored = triage.store_trail("x", {"first": {"thinking": ["t" * 60_000], "raw_response": "r"}})
    assert stored["stored"] == "inline"
    assert len(json.dumps(stored["trail"])) <= 20_000
    assert stored["trail"]["first"]["thinking"][0].endswith("[cut to fit]")


@mock_aws
def test_trails_go_to_s3_when_storage_is_set_up(monkeypatch, tickets, cases):
    monkeypatch.setenv("MODEL_BUCKET", "models")
    monkeypatch.setenv("B2_KEY_ID", "key")
    monkeypatch.setenv("B2_APPLICATION_KEY", "secret")
    monkeypatch.setenv("MODEL_SIGNING_KEY", "k" * 40)
    monkeypatch.setenv("MODEL_STORAGE_REGION", "us-east-1")
    monkeypatch.delenv("B2_ENDPOINT_URL", raising=False)
    boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="models")

    review_manual_ticket(monkeypatch, tickets)

    case = cases.find_one({})
    assert case["trail"]["stored"] == "s3"
    assert case["trail"]["key"] == f"triage-cases/{case['_id']}.json"
    assert triage.load_trail(case)["first"]["thinking"] == ["I weighed S3 against R1."]


# --- evidence -----------------------------------------------------------------------


def test_jira_export_is_parsed():
    found = evidence.extract(evidence.JIRA, JIRA_XML.encode(), "LRN-1234.xml")
    issue = found.data["issues"][0]
    assert issue["key"] == "LRN-1234"
    assert issue["status"] == "Released" and issue["resolution"] == "Fixed"
    assert issue["status_category"] == "done"
    assert issue["description"] == "Render PDFs inline.\nSecond line"
    assert issue["comments"] == [
        {"author": "dev", "created": "Mon, 2 Nov 2026 09:00:00 -0500", "text": "Shipped in v2026.10"}
    ]
    assert issue["links"] == [{"type": "Relates", "relation": "relates to", "issues": ["LRN-1000"]}]
    assert issue["custom_fields"] == [{"name": "Fix release", "values": ["2026.10"]}]
    assert evidence.describe({"kind": "jira", "data": found.data}) == "Jira LRN-1234 (Released, Fixed)"


@pytest.mark.parametrize(
    "content, reason",
    [
        ("<rss><channel>", "well-formed"),
        ("<feed/>", "an <rss> root"),
        ("<rss version='2'><channel/></rss>", "no issues"),
        (JIRA_XML.replace('<key id="99">LRN-1234</key>', ""), "no issue key"),
        (JIRA_XML.replace('<status id="6" description="Done">Released</status>', ""), "no status"),
        (
            '<?xml version="1.0"?><!DOCTYPE r [<!ENTITY a "aaaa"><!ENTITY b "&a;&a;">]><rss>&b;</rss>',
            "isn't allowed",
        ),
        (
            '<?xml version="1.0"?><!DOCTYPE r [<!ENTITY x SYSTEM "file:///etc/passwd">]><rss>&x;</rss>',
            "isn't allowed",
        ),
        ("", "empty"),
    ],
)
def test_unparseable_jira_is_rejected(content, reason):
    with pytest.raises(evidence.EvidenceError, match=reason):
        evidence.extract(evidence.JIRA, content)


def test_aha_page_text_is_parsed():
    data = evidence.extract(evidence.AHA, AHA_TEXT).data
    assert data["reference"] == "LAPIPLAT-I-543"
    assert data["title"] == "Future PDF Viewer support?"
    assert data["status"] == "Parked"
    assert data["assigned_to"] == "Audrey Auyang"
    assert data["created_by"] == "Terry Brooks" and data["created_on"] == "Dec 12, 2025"
    assert data["url"].startswith("https://example.aha.io/ideas/ideas/LAPIPLAT-I-543")
    assert [(w["from"], w["to"]) for w in data["workflow_changes"]] == [
        ("Scoping", "Parked"),
        ("New", "Scoping"),
    ]
    # The deleted comment and its live duplicate count once.
    assert [(c["author"], c["internal"], c["text"]) for c in data["comments"]] == [
        ("Vlad Manu", False, "Try this JSON"),
        ("Audrey Auyang", True, "Strategic Alignment: request from Cengage"),
    ]
    assert data["activity_days"] == ["04 June", "12 December"]
    assert "" not in json.dumps(data)


@pytest.mark.parametrize(
    "content, reason",
    [
        ("just some notes", "No Aha reference"),
        (b"%PDF-1.7 not really a pdf", "PDF"),
        (b"\xff\xfe\x00bad", "UTF-8"),
    ],
)
def test_unparseable_aha_is_rejected(content, reason):
    with pytest.raises(evidence.EvidenceError, match=reason):
        evidence.extract(evidence.AHA, content)


def open_case(monkeypatch, tickets, cases):
    review_manual_ticket(monkeypatch, tickets)
    return cases.find_one({})


def add_jira(cases, case, content=JIRA_XML):
    found = evidence.extract(evidence.JIRA, content, "export.xml")
    return triage.add_evidence(
        cases, case["_id"], found.kind, found.data, found.source, found.sha256, "reviewer"
    )


def test_evidence_is_add_only_and_never_duplicated(monkeypatch, tickets, cases):
    case = open_case(monkeypatch, tickets, cases)
    entry = add_jira(cases, case)
    assert entry["added_by"] == "reviewer" and entry["source"] == "export.xml"
    with pytest.raises(triage.EvidenceRejected, match="already added"):
        add_jira(cases, case)
    stored = cases.find_one({})["evidence"]
    assert len(stored) == 1 and "LRN-1234" in json.dumps(stored[0]["data"])
    # Only what was extracted: no copy of the file.
    assert JIRA_XML not in json.dumps(stored, default=str)


def test_evidence_has_limits(monkeypatch, tickets, cases):
    case = open_case(monkeypatch, tickets, cases)
    with pytest.raises(LookupError):
        triage.add_evidence(cases, ObjectId(), "jira", {}, "x", "y")
    monkeypatch.setattr(triage, "MAX_EVIDENCE_ITEMS", 1)
    add_jira(cases, case)
    with pytest.raises(triage.EvidenceRejected, match="already holds 1"):
        add_jira(cases, case, JIRA_XML.replace("Second line", "Third line"))
    monkeypatch.setattr(triage, "MAX_EVIDENCE_BYTES", 10)
    with pytest.raises(triage.EvidenceRejected, match="limit"):
        add_jira(cases, case, JIRA_XML.replace("Second line", "Fourth line"))


# --- re-running ---------------------------------------------------------------------


def test_a_rerun_needs_an_arbiter_and_new_evidence(monkeypatch, tickets, cases):
    case = open_case(monkeypatch, tickets, cases)
    with pytest.raises(triage.RerunUnavailable, match="No arbiter"):
        triage.rerun_case(tickets, cases, case["_id"])
    monkeypatch.setenv("CHEAPER_INFERENCE_ARBITER_MODEL_NAME", "arbiter")
    with pytest.raises(triage.RerunUnavailable, match="Add Jira or Aha"):
        triage.rerun_case(tickets, cases, case["_id"])


def test_a_rerun_sends_the_case_and_evidence_and_saves_a_new_review(monkeypatch, tickets, cases):
    case = open_case(monkeypatch, tickets, cases)
    add_jira(cases, case)
    monkeypatch.setenv("CHEAPER_INFERENCE_ARBITER_MODEL_NAME", "arbiter")
    client = FakeClient(model_reply("flag_solve", 90, signals_matched=["S3"]))
    monkeypatch.setattr(classify, "get_client", lambda: client)

    result = triage.rerun_case(tickets, cases, case["_id"], requested_by="reviewer")

    assert result["classification"] == "flag_solve"
    assert client.calls[0]["model"] == "arbiter"
    payload = client.calls[0]["messages"][0]["content"]
    data = json.loads(payload.split("\n\n", 1)[1])
    assert data["triage_case"]["why_manual_triage"] == triage.ROUTE_LABELS[triage.ABSTAINED]
    assert data["triage_case"]["answers"][0]["working_notes"] == (
        "<customer_text>I weighed S3 against R1.</customer_text>"
    )
    assert "primary" not in json.dumps(data["triage_case"])  # no model names
    issue = data["reviewer_evidence"][0]["data"]["issues"][0]
    assert issue["status"] == "<customer_text>Released</customer_text>"

    ticket = tickets.find_one({"zendesk_id": case["zendesk_id"]})
    original, rerun = ticket["agent_reviews"]
    assert rerun["rerun_of"] == original["review_id"]
    assert rerun["case_id"] == case["_id"]
    assert rerun["input"]["payload"] == payload
    assert ticket["awaiting_review"] is False
    stored = cases.find_one({})
    assert stored["status"] == triage.RESOLVED
    assert stored["resolved_by_review"] == rerun["review_id"]
    assert stored["reruns"][0]["classification"] == "flag_solve"
    assert "rerun_claim" not in stored

    with pytest.raises(triage.RerunUnavailable, match="since the last re-run"):
        triage.rerun_case(tickets, cases, case["_id"])


def test_a_manual_triage_rerun_leaves_the_case_open(monkeypatch, tickets, cases):
    case = open_case(monkeypatch, tickets, cases)
    add_jira(cases, case)
    monkeypatch.setenv("CHEAPER_INFERENCE_ARBITER_MODEL_NAME", "arbiter")
    monkeypatch.setattr(classify, "get_client", lambda: FakeClient(manual_reply()))

    triage.rerun_case(tickets, cases, case["_id"])

    stored = cases.find_one({})
    assert stored["status"] == triage.OPEN and len(stored["reruns"]) == 1


def test_a_failed_rerun_releases_its_claim(monkeypatch, tickets, cases):
    case = open_case(monkeypatch, tickets, cases)
    add_jira(cases, case)
    monkeypatch.setenv("CHEAPER_INFERENCE_ARBITER_MODEL_NAME", "arbiter")
    monkeypatch.setattr(classify, "get_client", lambda: FakeClient("not json"))

    with pytest.raises(classify.EngineError):
        triage.rerun_case(tickets, cases, case["_id"])

    stored = cases.find_one({})
    assert "rerun_claim" not in stored and stored["reruns"] == []
    assert len(tickets.find_one({})["agent_reviews"]) == 1


def test_a_running_rerun_blocks_another(monkeypatch, tickets, cases):
    case = open_case(monkeypatch, tickets, cases)
    add_jira(cases, case)
    monkeypatch.setenv("CHEAPER_INFERENCE_ARBITER_MODEL_NAME", "arbiter")
    assert triage._claim(cases, case["_id"], "other") is not None
    with pytest.raises(triage.RerunUnavailable, match="already being re-run"):
        triage.rerun_case(tickets, cases, case["_id"])


# --- lessons learned and training data ----------------------------------------------


def resolve_a_case(monkeypatch, tickets, cases):
    case = open_case(monkeypatch, tickets, cases)
    add_jira(cases, case)
    monkeypatch.setenv("CHEAPER_INFERENCE_ARBITER_MODEL_NAME", "arbiter")
    monkeypatch.setattr(
        classify, "get_client", lambda: FakeClient(model_reply("flag_solve", 90, signals_matched=["S3"]))
    )
    triage.rerun_case(tickets, cases, case["_id"])
    return cases.find_one({})


def test_resolved_cases_feed_lessons_learned(monkeypatch, tickets, cases):
    case = resolve_a_case(monkeypatch, tickets, cases)
    week = datetime.now(timezone.utc)
    resolved = triage.resolved_cases_for_week(cases, week)
    assert [c["_id"] for c in resolved] == [case["_id"]]

    summary = feedback.suggest_lessons_summary([], resolved)
    assert summary == (
        "- 1 manual-triage ticket(s) resolved as flag_solve on re-run with Jira evidence. "
        "It answered: Has LRN-1234 been released?."
    )
    log, lessons = feedback.get_feedback_collections(tickets.database)
    lesson_id = feedback.draft_lessons_learned(log, lessons, week, summary, [case["_id"]])
    assert lessons.find_one({"_id": lesson_id})["based_on_case_ids"] == [case["_id"]]
    with pytest.raises(ValueError, match="resolved triage cases"):
        feedback.draft_lessons_learned(log, lessons, week, summary)


def test_a_rerun_marked_correct_is_exported_with_its_evidence(monkeypatch, tickets, cases, tmp_path):
    resolve_a_case(monkeypatch, tickets, cases)
    ticket = tickets.find_one({})
    rerun = ticket["agent_reviews"][1]
    feedback.record_outcome(tickets, ticket["zendesk_id"], rerun["review_id"], "good")
    log, _ = feedback.get_feedback_collections(tickets.database)

    examples = list(training_export.review_candidates(tickets, log, {"reviews": 0, "contradicted": 0, "uncertain_history": 0}))

    assert len(examples) == 1
    assert examples[0]["label"] == "re-run with reviewer evidence, marked correct"
    assert examples[0]["rerun_of"] == str(rerun["rerun_of"])
    assert "reviewer_evidence" in examples[0]["payload"]
    assert examples[0]["answer"]["open_questions"] == []


def test_a_rerun_without_its_saved_input_is_not_exported(monkeypatch, tickets, cases):
    resolve_a_case(monkeypatch, tickets, cases)
    ticket = tickets.find_one({})
    rerun = ticket["agent_reviews"][1]
    tickets.update_one({}, {"$set": {"agent_reviews.1.input.payload": None}})
    feedback.record_outcome(tickets, ticket["zendesk_id"], rerun["review_id"], "good")
    log, _ = feedback.get_feedback_collections(tickets.database)
    counts = {"reviews": 0, "contradicted": 0, "uncertain_history": 0}

    assert list(training_export.review_candidates(tickets, log, counts)) == []
    assert counts["reviews"] == 1
