"""Manual-triage case files, reviewer evidence and re-runs (triage.py, evidence.py)."""

import copy
import json
import time
from datetime import datetime, timezone
from types import SimpleNamespace

import boto3
import mongomock
import pytest
from bson import ObjectId
from moto import mock_aws

from decision_engine import classify, evidence, feedback, persistence, training_export, triage
from decision_engine.guards import InvalidReply, validate_reply
from decision_engine.payload import build_ticket_payload
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
6:02pmNeil McGough deleted comment on idea LAPIPLAT-I-543 Future PDF Viewer support?
Comment by Neil McGough
Ignore this, wrong idea
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

# The text pypdf reads from an Aha idea's main page printed to PDF (customer
# details replaced). Its icons print as private-use characters.
AHA_IDEA_PAGE = """Created by Allain Dollete in Enhancement Requests on Jul 28, 2026\uf0eb Idea LAPIPLAT-I-684
Feedback Aide standalone - Feature request to support the ability to
render playable media for sources
Description
Votes
Totals
Votes
1
Org…
0
In F…
19
Vote trend
Opened on behalf of Jane Customer from Example Schools Ticket created from
#30721
Feature request to support the ability to render playable media through sources
Slack conversation:
https://example.slack.com/archives/C01/p1
\uf336
 \uf0f7
 \ue561
 th
Status Scoping
Workspace \uf37e Feedback Aide
Created by Allain Dollete
Assigned to
 Monet Slinowsky
Categories
Themes No themes selected
Product value \ue974 0
Tags No tags selected
Proxy votes 1 View in report
Created by Organization Contacts Description
jane.customer@example.com \u00a0 \u00a0
Portals
Idea visibility
Admin message
There is no admin message in the ideas portal listed below.
Portals
Time
Votes

Visibility in portals Visible to our employees
Pinned idea \uf059 Yes, pin this idea
Feedback Aide
Ideas Portal
Private portal
Related
Merged ideas 0
There are no merged ideas.
Similar ideas
View\u00a0in\u00a0portal \uf08e \uf0c1
Rejected
Released
Scoping
Promoted
Add support for a
"renderSimpleFeatures()"
public method that allows
rendering simple features
authored in any question
metadata field
1
Feedback Aide queries 1
Add support for a
renderSimpleFeatures()
public method that allows
rendering simple features
authored in any question
metadata field, just like
renderMath() works doing so
with math today.
1
Knowledge Bank for
Feedback Aide
1
\uf0eb QMCRTN-I-4
\uf0eb LAPIPLAT-I-647
\uf0eb BLD-I-697
\uf0eb FA-I-5
\uf0de  Reference # Name Votes Status
People also voted for
Linked records 0
There are no linked records.
Research
Concepts
Add notes, whiteboards, and prototypes to kick off the creative
process.
Promoted
Released
Released
Promoted
Released
Released
Support the ability to display
alternate responses
alongside correct answers
2
Feedback Aide Latex support 1
Improve accessibility of File
Upload
1
Process images in sources 1
Unbundled (stateless)
Feedback Aide in QM
0
Support stem with resources
(Feedback Aide)
5
\uf0eb LAPIPLAT-I-68
\uf0eb LAPIPLAT-I-442
\uf0eb LAPIPLAT-I-514
\uf0eb FA-I-29
\uf0eb OD3-I-1811
\uf0eb LAPIPLAT-I-299
BrokenExpose Feedback Aide
validation/scoring errors in Items
API session data
1\uf0eb BLD-I-685
\uf61f
\uf0de  Reference # Name Votes Status
\uf0de  Reference # Name Votes Status
Ideas
Link to existing ideas submitted by customers and the team.\uf0eb
"""

# Another main page: merged into another idea, its related-ideas tables split
# across the PDF's pages.
AHA_MERGED_IDEA_PAGE = """Created by Samantha Redwin in Enhancement Requests on Jun 11, 2026\uf0eb Idea LAPIPLAT-I-662
Add Video to Feedback Aide Sources
Description
Votes
Totals
Votes
1
Org…
0
Prox…
0
Vote trend
Proxy votes View in report
Dynamic elements such as video or audio players are generally not supported in
Feedback Aide.
Acme have specifically requested the addition of video sources.
\uf336
 \uf0f7
 \ue4d2
Time
Votes
\ue526 Merged into FA-I-21
Status Parked
Workspace \uf37e Feedback Aide
Created by Samantha Redwin
Assigned to
 Monet Slinowsky
Categories
Themes No themes selected
Product value \ue974 0
Tags No tags selected
There are no proxy votes for this idea.
Portals
Idea visibility
Admin message
There is no admin message in the ideas portal listed below.
Portals
Feedback Aide
Ideas Portal
Private portal
Related
Merged ideas 0
There are no merged ideas.
Visibility in portals Visible to our employees
Pinned idea \uf059 Yes, pin this idea
View\u00a0in\u00a0portal \uf08e \uf0c1
Similar ideas
People also voted for
Scoping
Scoping
Released
Promoted
Released
Parked
Promoted
Rejected
Released
Parked
Enabling Audio in LongtextV2 1
Feedback Aide standalone -
Feature request to support
the ability to render playable
media for sources
1
Feedback Aide queries 1
Add Playback Speed To
Hosted Video Player
1
Support the ability to add
close captions to simple
feature videos
1
Add descriptive audio track
support to video player
3
Improve dynamic content
authoring
11
Support Programmatic Media
Device Selection and Volume
Control
1
Content moderation backend
flag in Feedback Aide
0
Handle videos as sources 2
\uf0eb LAPIPLAT-I-660
\uf0eb LAPIPLAT-I-684
\uf0eb LAPIPLAT-I-647
\uf0eb LAPIPLAT-I-421
\uf0eb LAPIPLAT-I-9
\uf0eb LAPIPLAT-I-239
\uf0eb LAPIPLAT-I-275
\uf0eb LAPIPLAT-I-630
\uf0eb FA-I-2
\uf0eb FA-I-21
Parked
Parked
Scoping
Handle videos as sources 2
Score images 3
Support the conversion of
short text to longtext
1
\uf0eb FA-I-21
\uf0eb FA-I-31
\uf0eb LAPIPLAT-I-659
\uf0de  Reference # Name Votes Status
\uf0de  Reference # Name Votes Status
Linked records 0
There are no linked records.
Research
Concepts
Add notes, whiteboards, and prototypes to kick off the creative
process.
Ideas
Link to existing ideas submitted by customers and the team.
Perspectives
Bring quantitative data to validate your plans.
Aha! Discovery
Conduct interviews to capture key insights
Scoping
Promoted
Enabling Audio in LongtextV2 1
IBHC as an MCP 1
\uf0eb LAPIPLAT-I-660
\uf0eb LAPIPLAT-I-661
\uf61f
\uf0eb
\uf086
\uf086
TRY
Try free
\uf0de  Reference # Name Votes Status
"""


@pytest.fixture(autouse=True)
def no_storage(monkeypatch):
    for name in ("MODEL_BUCKET", "TRIAGE_CASES_COLLECTION", "CHEAPER_INFERENCE_ARBITER_MODEL_NAME"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(triage, "_last_sweep", None)  # each test may sweep once


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
    extra.setdefault("open_questions", ["Has LRN-1234 been released?"])
    return model_reply("manual_triage", 30, signals_matched=[], **extra)


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


def test_a_case_failure_flags_the_review_and_is_recovered_at_any_age(monkeypatch, tickets, cases):
    real_open = triage.open_case
    monkeypatch.setattr(triage, "open_case", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    zendesk_id = review_manual_ticket(monkeypatch, tickets)
    review = tickets.find_one({"zendesk_id": zendesk_id})["agent_reviews"][0]
    assert cases.count_documents({}) == 0
    assert review["case_pending"] is True
    assert review["pending_trail"]["trail"]["first"]["thinking"] == ["I weighed S3 against R1."]

    monkeypatch.setattr(triage, "open_case", real_open)
    # No rows: the flag finds it, however old the review.
    assert triage.recover_missing_cases(tickets, cases) == 1
    case = cases.find_one({})
    assert case["review_id"] == review["review_id"]
    assert case["open_questions"] == ["Has LRN-1234 been released?"]
    assert case["trail"]["trail"]["first"]["thinking"] == ["I weighed S3 against R1."]
    assert case["cleanup_done"] is True
    review = tickets.find_one({})["agent_reviews"][0]
    assert not {"case_pending", "pending_trail"} & set(review)
    assert "raw_response" not in review["audit"]
    assert triage.recover_missing_cases(tickets, cases) == 0


def test_an_unflagged_review_in_view_still_gets_its_case(monkeypatch, tickets, cases):
    real_open = triage.open_case
    monkeypatch.setattr(triage, "open_case", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(triage, "mark_case_pending", lambda *a: (_ for _ in ()).throw(RuntimeError("down")))
    zendesk_id = review_manual_ticket(monkeypatch, tickets)
    monkeypatch.setattr(triage, "open_case", real_open)
    review = tickets.find_one({})["agent_reviews"][0]
    assert "case_pending" not in review

    rows = [{"zendesk_id": zendesk_id, "review": review}]
    assert triage.recover_missing_cases(tickets, cases, rows, {}) == 1
    found = triage.cases_for_reviews(cases, [review["review_id"]])
    assert triage.recover_missing_cases(tickets, cases, rows, found) == 0


def test_cleanup_that_failed_is_retried(monkeypatch, tickets, cases):
    real_drop = triage.drop_kept_replies
    monkeypatch.setattr(triage, "drop_kept_replies", lambda *a: (_ for _ in ()).throw(RuntimeError("blip")))
    review_manual_ticket(monkeypatch, tickets)
    assert cases.find_one({})["cleanup_done"] is False
    assert "raw_response" in tickets.find_one({})["agent_reviews"][0]["audit"]

    monkeypatch.setattr(triage, "drop_kept_replies", real_drop)
    assert triage.recover_missing_cases(tickets, cases) == 0
    assert cases.find_one({})["cleanup_done"] is True
    assert "raw_response" not in tickets.find_one({})["agent_reviews"][0]["audit"]


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
        "guard_overrides": [],
    }
    assert case["arbitration"]["preferred_reasoning"] == "A cites S3 properly"


def test_a_long_trail_without_storage_is_cut_to_fit(monkeypatch):
    monkeypatch.setattr(triage, "MAX_INLINE_TRAIL_BYTES", 20_000)
    stored = triage.store_trail("x", {"first": {"thinking": ["t" * 60_000], "raw_response": "r"}})
    assert stored["stored"] == "inline"
    assert len(json.dumps(stored["trail"])) <= 20_000
    assert stored["trail"]["first"]["thinking"][0].endswith("[cut to fit]")


def use_storage(monkeypatch):
    monkeypatch.setenv("MODEL_BUCKET", "models")
    monkeypatch.setenv("B2_KEY_ID", "key")
    monkeypatch.setenv("B2_APPLICATION_KEY", "secret")
    monkeypatch.setenv("MODEL_SIGNING_KEY", "k" * 40)
    monkeypatch.setenv("MODEL_STORAGE_REGION", "us-east-1")
    monkeypatch.delenv("MODEL_PREFIX", raising=False)
    monkeypatch.delenv("B2_ENDPOINT_URL", raising=False)
    boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="models")


@mock_aws
def test_trails_go_to_s3_when_storage_is_set_up(monkeypatch, tickets, cases):
    use_storage(monkeypatch)

    review_manual_ticket(monkeypatch, tickets)

    case = cases.find_one({})
    assert case["trail"]["stored"] == "s3"
    # Under the storage prefix, which a prefix-limited storage key can write.
    assert case["trail"]["key"] == f"topic-models/_triage-cases/{case['_id']}.json"
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
    # Vlad's comment was deleted later, and Neil's deletion has no live copy:
    # neither is evidence.
    assert [(c["author"], c["internal"], c["text"]) for c in data["comments"]] == [
        ("Audrey Auyang", True, "Strategic Alignment: request from Cengage"),
    ]
    assert "Try this JSON" not in json.dumps(data["comments"])
    assert "wrong idea" not in json.dumps(data["comments"])
    created = next(a for a in data["activity"] if a["action"] == "commented on")
    assert created["deleted_later"] is True
    assert data["activity_days"] == ["04 June", "12 December"]
    assert data["page"] == "history" and "related_ideas" not in data
    assert "" not in json.dumps(data)


def test_an_aha_idea_page_is_parsed():
    data = evidence.extract(evidence.AHA, AHA_IDEA_PAGE).data
    assert data["page"] == "idea"
    assert data["reference"] == "LAPIPLAT-I-684"
    # The title wraps onto a second line, and the page has no print footer.
    assert data["title"] == (
        "Feedback Aide standalone - Feature request to support the ability to "
        "render playable media for sources"
    )
    assert data["status"] == "Scoping" and data["assigned_to"] == "Monet Slinowsky"
    assert data["created_by"] == "Allain Dollete" and data["created_on"] == "Jul 28, 2026"
    # The main page prints no link: it's built from the reference.
    assert data["url"] == "https://questionmarkcorp.aha.io/ideas/ideas/LAPIPLAT-I-684"
    assert data["zendesk_ticket"] == "30721"
    assert (data["requested_by"], data["requested_by_org"]) == ("Jane Customer", "Example Schools")
    assert data["workspace"] == "Feedback Aide"
    assert data["categories"] is data["themes"] is data["tags"] is None
    assert data["product_value"] == 0 and data["visibility"] == "Visible to our employees"
    assert data["votes"] == 1 and data["vote_totals"] == {"Votes": 1, "Org": 0, "In F": 19}
    assert data["merged_into"] is None
    assert data["merged_ideas"] == 0 and data["linked_records"] == 0
    assert data["comments"] == [] and data["workflow_changes"] == []
    assert "History tab" in data["history"] and "activity_days" not in data
    # The description stops at the sidebar: no other idea's status, and not
    # the proxy voter's email.
    assert data["description"].startswith("Opened on behalf of Jane Customer")
    assert data["description"].endswith("https://example.slack.com/archives/C01/p1")
    assert "Released" not in data["description"] and "Status" not in data["description"]
    assert "jane.customer@example.com" not in json.dumps(data)
    assert [(r["reference"], r["status"], r["votes"]) for r in data["related_ideas"]] == [
        ("QMCRTN-I-4", "Rejected", 1),
        ("LAPIPLAT-I-647", "Released", 1),
        ("BLD-I-697", "Scoping", 1),
        ("FA-I-5", "Promoted", 1),
        ("LAPIPLAT-I-68", "Promoted", 2),
        ("LAPIPLAT-I-442", "Released", 1),
        ("LAPIPLAT-I-514", "Released", 1),
        ("FA-I-29", "Promoted", 1),
        ("OD3-I-1811", "Released", 0),
        ("LAPIPLAT-I-299", "Released", 5),
    ]
    assert data["related_ideas"][1]["name"] == "Feedback Aide queries"
    assert "none of them is this idea's status" in data["related_ideas_note"]
    assert evidence.describe({"kind": evidence.AHA, "data": data}).endswith("(Scoping)")


def test_an_aha_idea_merged_into_another_says_so():
    data = evidence.extract(evidence.AHA, AHA_MERGED_IDEA_PAGE).data
    assert (data["reference"], data["title"]) == ("LAPIPLAT-I-662", "Add Video to Feedback Aide Sources")
    assert (data["status"], data["merged_into"]) == ("Parked", "FA-I-21")
    assert data["url"] == "https://questionmarkcorp.aha.io/ideas/ideas/LAPIPLAT-I-662"
    assert data["description"] == (
        "Dynamic elements such as video or audio players are generally not supported in\n"
        "Feedback Aide.\nAcme have specifically requested the addition of video sources."
    )
    assert data["zendesk_ticket"] is None and data["requested_by"] is None
    related = {r["reference"]: r for r in data["related_ideas"]}
    # The idea it was merged into, listed in two tables, appears once.
    assert related["FA-I-21"] == {
        "reference": "FA-I-21", "name": "Handle videos as sources", "status": "Parked", "votes": 2
    }
    assert related["LAPIPLAT-I-684"]["status"] == "Scoping"
    assert related["LAPIPLAT-I-275"]["votes"] == 11
    # The table continued on the PDF's second page.
    assert related["LAPIPLAT-I-661"]["name"] == "IBHC as an MCP"
    assert len(related) == 13 and "LAPIPLAT-I-662" not in related
    assert evidence.describe({"kind": evidence.AHA, "data": data}) == (
        "Aha LAPIPLAT-I-662: Add Video to Feedback Aide Sources (Parked, merged into FA-I-21)"
    )


def test_an_aha_feature_without_a_printed_link_gets_a_feature_link():
    text = AHA_MERGED_IDEA_PAGE.replace("Idea LAPIPLAT-I-662", "Feature FA-100")
    data = evidence.extract(evidence.AHA, text).data
    assert data["reference"] == "FA-100"
    assert data["url"] == "https://questionmarkcorp.aha.io/features/FA-100"


def test_related_ideas_whose_columns_dont_pair_up_are_left_out():
    # One status missing from the first table: its rows can't be paired, so
    # they're left out rather than given a wrong status.
    text = AHA_IDEA_PAGE.replace("Rejected\nReleased\nScoping\nPromoted\n", "Released\nScoping\nPromoted\n")
    related = {r["reference"]: r for r in evidence.extract(evidence.AHA, text).data["related_ideas"]}
    assert "QMCRTN-I-4" not in related and "FA-I-5" not in related
    assert related["LAPIPLAT-I-299"]["status"] == "Released"


def test_aha_statuses_may_carry_punctuation():
    text = AHA_IDEA_PAGE.replace("Rejected\nReleased\nScoping\nPromoted\n", "Won’t do\nReleased\nScoping\nPromoted\n")
    related = {r["reference"]: r for r in evidence.extract(evidence.AHA, text).data["related_ideas"]}
    assert related["QMCRTN-I-4"]["status"] == "Won’t do"


def test_only_the_tables_references_are_related_ideas():
    # A reference on a line of its own in the description isn't a table row.
    text = AHA_IDEA_PAGE.replace("#30721\n", "#30721\nFA-I-999\n")
    data = evidence.extract(evidence.AHA, text).data
    assert "FA-I-999" in data["description"]
    assert "FA-I-999" not in {r["reference"] for r in data["related_ideas"]}


def test_an_aha_description_keeps_lines_that_look_like_the_sidebar():
    text = AHA_MERGED_IDEA_PAGE.replace(
        "Acme have specifically requested the addition of video sources.\n",
        "Acme have specifically requested the addition of video sources.\n"
        "Status of the request: urgent\nMerged into LAPIPLAT-I-1 last year, then split out\nOK\n",
    )
    data = evidence.extract(evidence.AHA, text).data
    assert data["description"].endswith(
        "Status of the request: urgent\nMerged into LAPIPLAT-I-1 last year, then split out\nOK"
    )
    assert data["merged_into"] == "FA-I-21"


def test_a_description_mentioning_a_merge_isnt_one():
    text = AHA_IDEA_PAGE.replace("#30721\n", "#30721\nMerged into FA-I-7\n")
    data = evidence.extract(evidence.AHA, text).data
    assert data["merged_into"] is None and "Merged into FA-I-7" in data["description"]


def test_a_long_wrapped_aha_title_is_kept_whole():
    text = AHA_MERGED_IDEA_PAGE.replace(
        "Add Video to Feedback Aide Sources\n", "Add\nVideo\nto\nFeedback\nAide\nSources\n"
    )
    assert evidence.extract(evidence.AHA, text).data["title"] == "Add Video to Feedback Aide Sources"


def test_many_reference_blocks_parse_quickly():
    blocks = "".join(f"Parked\nIdea {n} 1\nFA-I-{n}\nnote\n" for n in range(20000))
    text = AHA_IDEA_PAGE.replace("Similar ideas\n", "Similar ideas\n" + blocks)
    started = time.monotonic()
    related = evidence.extract(evidence.AHA, text).data["related_ideas"]
    assert time.monotonic() - started < 5
    assert len(related) == evidence.MAX_TABLE_BLOCKS


def test_a_linked_jira_key_isnt_taken_for_the_aha_item():
    text = AHA_TEXT.replace(
        "Does Learnosity have any plans", "Linked Jira: LRN-1234. Does Learnosity have any plans"
    ).replace("https://example.aha.io/ideas/ideas/LAPIPLAT-I-543", "https://example.aha.io/ideas/ideas/OTHER-I-1 https://example.aha.io/ideas/ideas/LAPIPLAT-I-543")
    data = evidence.extract(evidence.AHA, text).data
    assert data["reference"] == "LAPIPLAT-I-543"
    assert data["url"].startswith("https://example.aha.io/ideas/ideas/LAPIPLAT-I-543")
    assert "LRN-1234" in data["description"]


def test_a_page_naming_two_aha_items_is_rejected():
    text = AHA_TEXT.replace("IdeaLAPIPLAT-I-543", "IdeaLAPIPLAT-I-999", 1)
    with pytest.raises(evidence.EvidenceError, match="LAPIPLAT-I-543, LAPIPLAT-I-999"):
        evidence.extract(evidence.AHA, text)


def test_only_one_of_two_identical_comments_is_withdrawn_by_one_deletion():
    page = """Created by Terry Brooks in Ideas on Dec 12, 2025 IdeaLAPIPLAT-I-543
Future PDF Viewer support?
6:20pmAudrey Auyang deleted comment on idea LAPIPLAT-I-543 Future PDF Viewer support?
Comment by Vlad Manu
+1
6:11pmVlad Manu commented on idea LAPIPLAT-I-543 Future PDF Viewer support?
Comment by Vlad Manu
+1
6:10pmVlad Manu commented on idea LAPIPLAT-I-543 Future PDF Viewer support?
Comment by Vlad Manu
+1
"""
    data = evidence.extract(evidence.AHA, page).data
    # The deletion was by Audrey, of Vlad's comment: matched by the comment's author.
    assert [(c["author"], c["text"]) for c in data["comments"]] == [("Vlad Manu", "+1")]


@pytest.mark.parametrize(
    "content, reason",
    [
        ("just some notes", "No Aha reference"),
        ("Linked Jira: LRN-1234 and LAPIPLAT-I-543 in passing", "No Aha reference"),
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
    with pytest.raises(triage.EvidenceRejected, match="the limit is 0 KB"):
        add_jira(cases, case, JIRA_XML.replace("Second line", "Fourth line"))


def test_a_cases_evidence_has_a_total_limit(monkeypatch, tickets, cases):
    case = open_case(monkeypatch, tickets, cases)
    first = add_jira(cases, case)
    size = cases.find_one({})["evidence_bytes"]
    # Room for the first entry and a little more, not for a second.
    monkeypatch.setattr(triage, "MAX_CASE_EVIDENCE_BYTES", size + 100)
    with pytest.raises(triage.EvidenceRejected, match="would total"):
        add_jira(cases, case, JIRA_XML.replace("Second line", "Third line"))
    assert [e["evidence_id"] for e in cases.find_one({})["evidence"]] == [first["evidence_id"]]


# --- re-running ---------------------------------------------------------------------


def test_a_rerun_needs_an_arbiter_and_new_evidence(monkeypatch, tickets, cases):
    case = open_case(monkeypatch, tickets, cases)
    with pytest.raises(triage.RerunUnavailable, match="No arbiter"):
        triage.rerun_case(tickets, cases, case["_id"])
    monkeypatch.setenv("CHEAPER_INFERENCE_ARBITER_MODEL_NAME", "arbiter")
    with pytest.raises(triage.RerunUnavailable, match="Add Jira or Aha"):
        triage.rerun_case(tickets, cases, case["_id"])


QUEUE_FIELDS = ("awaiting_review", "review_claim", "review_attempts", "last_engine_error", "status")


def queue_state(tickets):
    ticket = tickets.find_one({})
    return {field: ticket.get(field, "<missing>") for field in QUEUE_FIELDS}


def test_a_rerun_sends_the_case_and_evidence_and_saves_a_new_review(monkeypatch, tickets, cases):
    monkeypatch.setenv("CHEAPER_INFERENCE_PRIMARY_MODEL_NAME", "first-model-x7")
    case = open_case(monkeypatch, tickets, cases)
    # Put it back in the queue, so a re-run that touched it would show.
    tickets.update_one({}, {"$set": {"awaiting_review": True, "review_attempts": 1}})
    before = queue_state(tickets)
    assert case["answers"][0]["model"] == "first-model-x7"
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
    assert "first-model-x7" not in payload  # no model names
    assert data["triage_case"]["answers"][0]["signals_matched"] == []
    issue = data["reviewer_evidence"][0]["data"]["issues"][0]
    assert issue["status"] == "<customer_text>Released</customer_text>"

    ticket = tickets.find_one({"zendesk_id": case["zendesk_id"]})
    original, rerun = ticket["agent_reviews"]
    assert rerun["rerun_of"] == original["review_id"]
    assert rerun["case_id"] == case["_id"]
    assert rerun["input"]["payload"] == payload
    assert queue_state(tickets) == before
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
    with pytest.raises(triage.RerunUnavailable, match="in progress"):
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
    lesson_id = feedback.draft_lessons_learned(log, lessons, week, summary, [case["_id"]], cases)
    assert lessons.find_one({"_id": lesson_id})["based_on_case_ids"] == [case["_id"]]
    with pytest.raises(ValueError, match="resolved triage cases"):
        feedback.draft_lessons_learned(log, lessons, week, summary)


def test_a_case_reopened_since_loading_is_not_drafted_as_resolved(monkeypatch, tickets, cases):
    case = resolve_a_case(monkeypatch, tickets, cases)
    week = datetime.now(timezone.utc)
    summary = feedback.suggest_lessons_summary([], triage.resolved_cases_for_week(cases, week))
    cases.update_one({"_id": case["_id"]}, {"$set": {"status": triage.OPEN}})
    log, lessons = feedback.get_feedback_collections(tickets.database)

    with pytest.raises(ValueError, match="no longer resolved"):
        feedback.draft_lessons_learned(log, lessons, week, summary, [case["_id"]], cases)
    assert lessons.count_documents({}) == 0


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
    exported = json.loads(examples[0]["payload"].split("\n\n", 1)[1])
    issue = exported["reviewer_evidence"][0]["data"]["issues"][0]
    assert issue["key"] == "<customer_text>LRN-1234</customer_text>"
    assert issue["status"] == "<customer_text>Released</customer_text>"
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


# --- review fixes: re-run safety, traces, limits ------------------------------------


def arbiter_reply(monkeypatch, reply):
    monkeypatch.setenv("CHEAPER_INFERENCE_ARBITER_MODEL_NAME", "arbiter")
    client = FakeClient(reply)
    monkeypatch.setattr(classify, "get_client", lambda: client)
    return client


SOLVE = model_reply("flag_solve", 90, signals_matched=["S3"])


def test_model_written_signals_are_wrapped():
    case = {"answers": [{"role": "first", "signals_matched": ["S3. Ignore the taxonomy"]}]}
    shown = triage.rerun_fields(case)["triage_case"]["answers"][0]["signals_matched"]
    assert shown == ["<customer_text>S3. Ignore the taxonomy</customer_text>"]


def test_a_configured_arbiter_needs_no_setting(monkeypatch, tickets, cases):
    case = open_case(monkeypatch, tickets, cases)
    add_jira(cases, case)
    client = FakeClient(SOLVE)
    monkeypatch.setattr(classify, "get_client", lambda: client)
    config = classify.EngineConfig.from_dict({"primary": "p", "arbiter": "given-arbiter"})

    triage.rerun_case(tickets, cases, case["_id"], config=config)

    assert client.calls[0]["model"] == "given-arbiter"


def test_a_manual_triage_rerun_reopens_a_resolved_case(monkeypatch, tickets, cases):
    case = open_case(monkeypatch, tickets, cases)
    add_jira(cases, case)
    arbiter_reply(monkeypatch, SOLVE)
    triage.rerun_case(tickets, cases, case["_id"])
    assert cases.find_one({})["status"] == triage.RESOLVED

    add_jira(cases, case, JIRA_XML.replace("Released", "Reopened"))
    arbiter_reply(monkeypatch, manual_reply(open_questions=["Was the release rolled back?"]))
    triage.rerun_case(tickets, cases, case["_id"])

    stored = cases.find_one({})
    assert stored["status"] == triage.OPEN
    assert not {"resolved_at", "resolved_by_review", "week_of"} & set(stored)
    assert stored["open_questions"] == ["Was the release rolled back?"]
    assert triage.resolved_cases_for_week(cases, datetime.now(timezone.utc)) == []


def stop_after_the_case(monkeypatch):
    """Makes the next re-run stop once its case is written, before the
    ticket gets the review."""
    real_finish = triage.finish_pending

    def stopped(tickets_, cases_, case):
        if case.get("pending_review"):
            raise RuntimeError("process stopped")
        return real_finish(tickets_, cases_, case)

    monkeypatch.setattr(triage, "finish_pending", stopped)
    return real_finish


def test_an_interrupted_rerun_is_finished_by_the_next_without_another_call(monkeypatch, tickets, cases):
    case = open_case(monkeypatch, tickets, cases)
    add_jira(cases, case)
    arbiter_reply(monkeypatch, SOLVE)
    real_finish = stop_after_the_case(monkeypatch)
    with pytest.raises(RuntimeError):
        triage.rerun_case(tickets, cases, case["_id"])
    monkeypatch.setattr(triage, "finish_pending", real_finish)
    stored = cases.find_one({})
    assert len(stored["reruns"]) == 1 and stored["pending_review"]
    assert len(tickets.find_one({})["agent_reviews"]) == 1

    client = arbiter_reply(monkeypatch, SOLVE)
    with pytest.raises(triage.RerunUnavailable, match="since the last re-run"):
        triage.rerun_case(tickets, cases, case["_id"])

    assert client.calls == []
    stored = cases.find_one({})
    reviews = tickets.find_one({})["agent_reviews"]
    assert len(reviews) == 2 and reviews[1]["review_id"] == stored["reruns"][0]["review_id"]
    assert "pending_review" not in stored and "rerun_claim" not in stored


def test_an_interrupted_rerun_is_finished_when_the_tab_loads(monkeypatch, tickets, cases):
    case = open_case(monkeypatch, tickets, cases)
    add_jira(cases, case)
    arbiter_reply(monkeypatch, SOLVE)
    real_finish = stop_after_the_case(monkeypatch)
    with pytest.raises(RuntimeError):
        triage.rerun_case(tickets, cases, case["_id"])
    monkeypatch.setattr(triage, "finish_pending", real_finish)

    triage.recover_missing_cases(tickets, cases)
    triage.recover_missing_cases(tickets, cases)  # again: nothing added twice

    reviews = tickets.find_one({})["agent_reviews"]
    assert len(reviews) == 2 and reviews[1]["case_id"] == case["_id"]
    assert "pending_review" not in cases.find_one({})
    assert triage.cases_for_reviews(cases, [reviews[1]["review_id"]])[reviews[1]["review_id"]]["_id"] == case["_id"]


def test_a_rerun_that_lost_its_claim_saves_nothing(monkeypatch, tickets, cases):
    case = open_case(monkeypatch, tickets, cases)
    add_jira(cases, case)
    client = arbiter_reply(monkeypatch, SOLVE)
    real_create = client.messages.create

    def slow_answer(**kwargs):
        # The claim expires during the call and another worker takes it.
        cases.update_one({}, {"$set": {"rerun_claim": {"worker": "other", "expires_at": datetime.max}}})
        return real_create(**kwargs)

    client.messages.create = slow_answer
    with pytest.raises(triage.RerunUnavailable, match="this answer was discarded"):
        triage.rerun_case(tickets, cases, case["_id"])

    assert len(tickets.find_one({})["agent_reviews"]) == 1
    stored = cases.find_one({})
    assert stored["reruns"] == [] and stored["rerun_claim"]["worker"] == "other"
    assert "pending_review" not in stored


def test_a_rerun_whose_claim_expired_saves_nothing_even_if_no_one_took_it(monkeypatch, tickets, cases):
    case = open_case(monkeypatch, tickets, cases)
    add_jira(cases, case)
    client = arbiter_reply(monkeypatch, SOLVE)
    real_create = client.messages.create

    def very_slow_answer(**kwargs):
        cases.update_one({}, {"$set": {"rerun_claim.expires_at": datetime(2020, 1, 1)}})
        return real_create(**kwargs)

    client.messages.create = very_slow_answer
    with pytest.raises(triage.RerunUnavailable, match="this answer was discarded"):
        triage.rerun_case(tickets, cases, case["_id"])
    assert len(tickets.find_one({})["agent_reviews"]) == 1
    assert cases.find_one({})["reruns"] == []


def test_a_rerun_whose_ticket_vanished_is_taken_back(monkeypatch, tickets, cases):
    case = open_case(monkeypatch, tickets, cases)
    add_jira(cases, case)
    client = arbiter_reply(monkeypatch, SOLVE)
    real_create = client.messages.create

    def ticket_deleted_meanwhile(**kwargs):
        tickets.delete_many({})
        return real_create(**kwargs)

    client.messages.create = ticket_deleted_meanwhile
    with pytest.raises(LookupError, match="removed while it was re-run"):
        triage.rerun_case(tickets, cases, case["_id"])

    stored = cases.find_one({})
    assert stored["reruns"] == [] and stored["status"] == triage.OPEN
    assert stored["open_questions"] == ["Has LRN-1234 been released?"]
    assert not {"pending_review", "resolved_at", "week_of", "rerun_claim"} & set(stored)


def test_evidence_used_while_waiting_for_the_claim_isnt_used_again(monkeypatch, tickets, cases):
    case = open_case(monkeypatch, tickets, cases)
    entry = add_jira(cases, case)
    client = arbiter_reply(monkeypatch, SOLVE)
    real_claim = triage._claim

    def after_another_rerun(cases_, case_id, worker):
        cases_.update_one({}, {"$push": {"reruns": {"review_id": ObjectId(), "evidence_ids": [entry["evidence_id"]]}}})
        return real_claim(cases_, case_id, worker)

    monkeypatch.setattr(triage, "_claim", after_another_rerun)
    with pytest.raises(triage.RerunUnavailable, match="since the last re-run"):
        triage.rerun_case(tickets, cases, case["_id"])
    assert client.calls == []
    assert "rerun_claim" not in cases.find_one({})


def test_each_rerun_trail_has_its_own_key(monkeypatch, tickets, cases):
    case = open_case(monkeypatch, tickets, cases)
    add_jira(cases, case)
    names = []
    monkeypatch.setattr(triage, "store_trail", lambda name, trail, inline_room=None: names.append(name) or {"stored": None})
    arbiter_reply(monkeypatch, SOLVE)
    triage.rerun_case(tickets, cases, case["_id"])
    rerun_review = tickets.find_one({})["agent_reviews"][1]
    assert names == [f"{case['_id']}-rerun-{rerun_review['review_id']}"]


def test_no_trail_is_kept_inline_once_the_case_is_near_the_limit(monkeypatch):
    monkeypatch.setattr(triage, "MAX_CASE_BYTES", 1000)
    stored = triage.store_trail("x", {"first": {"thinking": ["t"], "raw_response": "r"}}, inline_room=100)
    assert stored == {"stored": "dropped"}


@pytest.mark.parametrize("confirmation", [None, "confirmation"])
def test_raw_replies_the_trail_keeps_leave_the_review(monkeypatch, tickets, cases, confirmation):
    if confirmation:
        monkeypatch.setenv("CHEAPER_INFERENCE_CONFIRMATION_MODEL_NAME", confirmation)
    else:
        monkeypatch.delenv("CHEAPER_INFERENCE_CONFIRMATION_MODEL_NAME", raising=False)
    zendesk_id = review_manual_ticket(monkeypatch, tickets)
    review = tickets.find_one({"zendesk_id": zendesk_id})["agent_reviews"][0]
    assert "raw_response" not in review["audit"]
    assert "raw_response" not in ((review.get("confirmation_of") or {}).get("audit") or {})
    trail = cases.find_one({})["trail"]["trail"]
    assert set(trail) == ({"first", "confirmation"} if confirmation else {"first"})
    assert all(entry["raw_response"].startswith("{") for entry in trail.values())


def test_a_failed_attempts_thinking_stays_off_the_ticket(monkeypatch, tickets):
    ticket = copy.deepcopy(CATEGORY_FIXTURES["manual_triage"])
    tickets.insert_one(ticket)
    monkeypatch.setattr(classify, "get_client", lambda: ThinkingClient("not json"))
    with pytest.raises(classify.EngineError):
        persistence.process_one_ticket(ticket["zendesk_id"], tickets)
    error = tickets.find_one({})["last_engine_error"]
    assert error["audit"]["raw_response"] == "not json"
    assert "thinking" not in error["audit"]


def test_a_disagreement_keeps_the_arbiters_trace(monkeypatch, tickets, cases):
    monkeypatch.setenv("CHEAPER_INFERENCE_PRIMARY_MODEL_NAME", "primary")
    monkeypatch.setenv("CHEAPER_INFERENCE_CONFIRMATION_MODEL_NAME", "confirmation")
    monkeypatch.setenv("CHEAPER_INFERENCE_ARBITER_MODEL_NAME", "arbiter")

    class Arbiter(RoutingClient):
        def _create(self, **kwargs):
            if kwargs["model"] == "arbiter" and len(kwargs["system"]) == 2:
                return thinking_response(
                    json.dumps({"preferred": "A", "reasoning": "A cites O1"}), "judging A vs B"
                )
            return super()._create(**kwargs)

    client = Arbiter(
        {
            "primary": model_reply("escalation_internal", 72, signals_matched=["O1"], open_questions=["Who owns it?"]),
            "confirmation": model_reply("on_track", 70, signals_matched=[]),
            "arbiter": model_reply("on_track", 80, signals_matched=[]),
        }
    )
    ticket = copy.deepcopy(CATEGORY_FIXTURES["flag_solve"])
    tickets.insert_one(ticket)
    monkeypatch.setattr(classify, "get_client", lambda: client)
    persistence.process_one_ticket(ticket["zendesk_id"], tickets)

    case = cases.find_one({})
    trail = case["trail"]["trail"]
    assert trail["arbiter_judge"]["thinking"] == ["judging A vs B"]
    assert '"preferred": "A"' in trail["arbiter_judge"]["raw_response"]
    assert '"classification": "on_track"' in trail["arbiter_ruling"]["raw_response"]
    # The first answer's details survive the confirmation.
    assert case["answers"][0]["signals_matched"] == ["O1"]
    assert case["answers"][0]["open_questions"] == ["Who owns it?"]
    review = tickets.find_one({})["agent_reviews"][0]
    assert "judge_audit" not in review["arbitration"] and "arbiter_audits" not in review


def test_a_guard_downgrade_before_confirmation_is_the_route():
    result = {
        "classification": "manual_triage",
        "guard_overrides": [],
        "confirmation_of": {"guard_overrides": ["Downgraded from flag_solve: confidence 60"]},
    }
    assert triage.route_of(result) == triage.GUARD_OVERRIDE


def test_rerun_lookups_are_indexed(monkeypatch, tickets, cases):
    open_case(monkeypatch, tickets, cases)
    keys = [list(index["key"]) for index in cases.index_information().values()]
    assert [("reruns.review_id", 1)] in keys


@pytest.mark.parametrize(
    "content, reason",
    [
        ("<rss>" + "<a>" * 70 + "</a>" * 70 + "</rss>", "deeper than"),
        ("<rss><channel>" + "<item/>" * 51 + "</channel></rss>", "over 50 issues"),
    ],
)
def test_oversized_xml_is_refused_while_streaming(content, reason):
    with pytest.raises(evidence.EvidenceError, match=reason):
        evidence.extract(evidence.JIRA, content)


def test_xml_with_too_many_elements_is_refused(monkeypatch):
    monkeypatch.setattr(evidence, "MAX_XML_ELEMENTS", 50)
    with pytest.raises(evidence.EvidenceError, match="over 50 elements"):
        evidence.extract(evidence.JIRA, "<rss>" + "<a/>" * 60 + "</rss>")


def test_a_pdf_with_too_many_pages_is_refused():
    from io import BytesIO

    from pypdf import PdfWriter

    writer = PdfWriter()
    for _ in range(evidence.MAX_PDF_PAGES + 1):
        writer.add_blank_page(width=100, height=100)
    out = BytesIO()
    writer.write(out)
    with pytest.raises(evidence.EvidenceError, match="41 pages"):
        evidence.extract(evidence.AHA, out.getvalue())


# --- second review: commits, traces, recovery ------------------------------------


def test_a_claim_lost_before_the_call_stops_it(monkeypatch, tickets, cases):
    case = open_case(monkeypatch, tickets, cases)
    add_jira(cases, case)
    client = arbiter_reply(monkeypatch, SOLVE)
    real_claim = triage._claim

    def then_taken_over(cases_, case_id, worker):
        claimed = real_claim(cases_, case_id, worker)
        cases_.update_one({}, {"$set": {"rerun_claim.worker": "other"}})
        return claimed

    monkeypatch.setattr(triage, "_claim", then_taken_over)
    with pytest.raises(triage.RerunUnavailable, match="took this case over"):
        triage.rerun_case(tickets, cases, case["_id"])
    assert client.calls == []
    assert cases.find_one({})["reruns"] == []


def test_a_rerun_never_removes_a_saved_review(monkeypatch, tickets, cases):
    case = open_case(monkeypatch, tickets, cases)
    add_jira(cases, case)
    arbiter_reply(monkeypatch, SOLVE)
    pulls = []
    real_update = tickets.update_one
    monkeypatch.setattr(
        tickets, "update_one",
        lambda f, u, *a, **k: pulls.append(u) if "$pull" in u else real_update(f, u, *a, **k),
    )
    triage.rerun_case(tickets, cases, case["_id"])
    assert pulls == []


def test_another_workers_claim_disables_a_rerun(monkeypatch, tickets, cases):
    case = open_case(monkeypatch, tickets, cases)
    add_jira(cases, case)
    monkeypatch.setenv("CHEAPER_INFERENCE_ARBITER_MODEL_NAME", "arbiter")
    assert triage._claim(cases, case["_id"], "other") is not None
    claimed = cases.find_one({})
    assert triage.rerun_problem(claimed) == "A re-run of this case is in progress."
    assert triage.rerun_problem(claimed, worker="other") is None
    cases.update_one({}, {"$set": {"rerun_claim.expires_at": datetime(2020, 1, 1)}})
    assert triage.rerun_problem(cases.find_one({})) is None  # expired


def test_a_manual_rerun_with_no_questions_clears_them(monkeypatch, tickets, cases):
    case = open_case(monkeypatch, tickets, cases)
    add_jira(cases, case)
    arbiter_reply(monkeypatch, manual_reply(open_questions=[]))
    triage.rerun_case(tickets, cases, case["_id"])
    assert cases.find_one({})["open_questions"] == []


def test_the_first_answers_guard_overrides_are_kept(monkeypatch, tickets, cases):
    monkeypatch.setenv("CHEAPER_INFERENCE_PRIMARY_MODEL_NAME", "primary")
    monkeypatch.setenv("CHEAPER_INFERENCE_CONFIRMATION_MODEL_NAME", "confirmation")
    client = RoutingClient({
        # Below the solve floor: a check downgrades it before the confirmation.
        "primary": model_reply("flag_solve", 70, signals_matched=["S3"]),
        "confirmation": manual_reply(),
    })
    monkeypatch.setattr(classify, "get_client", lambda: client)
    ticket = copy.deepcopy(CATEGORY_FIXTURES["flag_solve"])
    tickets.insert_one(ticket)
    persistence.process_one_ticket(ticket["zendesk_id"], tickets)

    case = cases.find_one({})
    assert case["route"] == triage.GUARD_OVERRIDE
    assert any(o.startswith("Downgraded from flag_solve") for o in case["guard_overrides"])
    assert case["answers"][0]["guard_overrides"] == case["guard_overrides"]
    shown = triage.rerun_fields(case)["triage_case"]["answers"][0]
    assert shown["overridden_by_checks"][0].startswith("<customer_text>Downgraded from flag_solve")


def arbiter_case(monkeypatch, ruling_reply):
    monkeypatch.setenv("CHEAPER_INFERENCE_PRIMARY_MODEL_NAME", "primary")
    monkeypatch.setenv("CHEAPER_INFERENCE_CONFIRMATION_MODEL_NAME", "confirmation")
    monkeypatch.setenv("CHEAPER_INFERENCE_ARBITER_MODEL_NAME", "arbiter")

    class Arbiter(RoutingClient):
        def _create(self, **kwargs):
            if kwargs["model"] == "arbiter" and len(kwargs["system"]) == 2:
                return fake_response(json.dumps({"preferred": "B", "reasoning": "B fits"}))
            return super()._create(**kwargs)

    client = Arbiter({
        "primary": model_reply("escalation_internal", 72, signals_matched=["O1"]),
        "confirmation": model_reply("on_track", 70, signals_matched=[]),
        "arbiter": ruling_reply,
    })
    monkeypatch.setattr(classify, "get_client", lambda: client)
    return classify.classify_ticket(build_ticket_payload(CATEGORY_FIXTURES["flag_solve"]))


def test_an_agreeing_arbiter_keeps_its_trace(monkeypatch):
    result = arbiter_case(monkeypatch, model_reply("on_track", 80, signals_matched=[]))
    assert result["classification"] == "on_track"
    assert set(result["arbiter_audits"]) == {"judge", "ruling"}


def test_a_failed_ruling_keeps_its_reply(monkeypatch):
    result = arbiter_case(monkeypatch, "not json")
    assert result["classification"] == "manual_triage"
    assert result["arbiter_audits"]["ruling"]["raw_response"] == "not json"
    assert '"preferred": "B"' in result["arbiter_audits"]["judge"]["raw_response"]


def test_repeated_engine_errors_keep_the_failed_replies(monkeypatch, tickets, cases):
    ticket = copy.deepcopy(CATEGORY_FIXTURES["manual_triage"])
    tickets.insert_one(ticket)
    monkeypatch.setattr(classify, "get_client", lambda: ThinkingClient("not json", "half a thought"))
    for _ in range(persistence.MAX_ENGINE_ATTEMPTS - 1):
        with pytest.raises(classify.EngineError):
            persistence.process_one_ticket(ticket["zendesk_id"], tickets)
    persistence.process_one_ticket(ticket["zendesk_id"], tickets)

    case = cases.find_one({})
    assert case["route"] == triage.ENGINE_ERROR
    trail = case["trail"]["trail"]
    assert trail["first"]["thinking"] == ["half a thought"]
    assert trail["first"]["raw_response"] == "not json"
    assert trail["previous_attempt"]["raw_response"] == "not json"
    assert "thinking" not in str(tickets.find_one({}).get("agent_reviews"))


@pytest.mark.parametrize("value", [False, 0, "", {}])
def test_a_falsey_non_list_of_questions_is_invalid(value):
    reply = json.loads(model_reply("on_track"))
    with pytest.raises(InvalidReply, match="open_questions"):
        validate_reply({**reply, "open_questions": value})


def test_the_aha_link_must_name_the_item_exactly():
    text = AHA_TEXT.replace(
        "Does Learnosity have any plans",
        "See https://example.aha.io/ideas/ideas/LAPIPLAT-I-5430 too. Does Learnosity have any plans",
    )
    data = evidence.extract(evidence.AHA, text).data
    assert data["url"].startswith("https://example.aha.io/ideas/ideas/LAPIPLAT-I-543?")


def test_weekly_lookups_are_indexed(monkeypatch, tickets, cases):
    open_case(monkeypatch, tickets, cases)
    keys = [list(index["key"]) for index in cases.index_information().values()]
    assert [("status", 1), ("week_of", 1)] in keys


# --- third review: budgets, PDFs, sweep, names ------------------------------------


def test_evidence_is_measured_as_the_arbiter_sees_it(monkeypatch, tickets, cases):
    case = open_case(monkeypatch, tickets, cases)
    data = {"issues": [{"key": "LRN-1", "summary": "<" * 1000}]}
    # Escaping turns each "<" into "&lt;", and every string gains its tags.
    assert triage.evidence_size(data) > 4000
    monkeypatch.setattr(triage, "MAX_EVIDENCE_BYTES", 3000)
    with pytest.raises(triage.EvidenceRejected, match="limit is 2 KB"):
        triage.add_evidence(cases, case["_id"], "jira", data, "x.xml", "sha")


def test_a_rerun_over_the_input_budget_sends_nothing(monkeypatch, tickets, cases):
    case = open_case(monkeypatch, tickets, cases)
    add_jira(cases, case)
    client = arbiter_reply(monkeypatch, SOLVE)
    monkeypatch.setattr(triage, "MAX_RERUN_INPUT_TOKENS", 100)
    with pytest.raises(triage.RerunUnavailable, match="Nothing was sent"):
        triage.rerun_case(tickets, cases, case["_id"])
    assert client.calls == []
    assert "rerun_claim" not in cases.find_one({})


def test_token_estimates_err_high():
    assert triage.estimated_tokens("word " * 400) >= 400  # ~1 token per English word
    assert triage.estimated_tokens("漢" * 300) >= 300  # ~1 token per CJK character


def test_an_unflagged_review_outside_the_view_is_swept_up(monkeypatch, tickets, cases):
    monkeypatch.setattr(triage, "open_case", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(triage, "mark_case_pending", lambda *a: (_ for _ in ()).throw(RuntimeError("down")))
    review_manual_ticket(monkeypatch, tickets)
    monkeypatch.undo()
    monkeypatch.setattr(triage, "_last_sweep", None)
    monkeypatch.delenv("MODEL_BUCKET", raising=False)

    assert triage.recover_missing_cases(tickets, cases) == 1  # no rows given
    assert cases.count_documents({}) == 1
    cases.delete_many({})
    assert triage.recover_missing_cases(tickets, cases) == 0  # swept recently


def test_the_cases_collection_cant_share_a_name(monkeypatch, tickets):
    monkeypatch.setenv("TRIAGE_CASES_COLLECTION", "tickets")
    with pytest.raises(ValueError, match="same as MONGO_COLLECTION"):
        triage.get_cases_collection(tickets.database)
    assert triage.cases_collection_problem("lessons_learned", {"LESSONS_COLLECTION": "lessons_learned"})
    assert triage.cases_collection_problem("triage_cases", triage.OTHER_COLLECTIONS) is None


def pdf_with_content(size: int) -> bytes:
    from io import BytesIO

    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, NameObject

    writer = PdfWriter()
    page = writer.add_blank_page(width=100, height=100)
    stream = DecodedStreamObject()
    stream.set_data(b"BT ET\n" * (size // 6))
    page[NameObject("/Contents")] = writer._add_object(stream)
    out = BytesIO()
    writer.write(out)
    return out.getvalue()


def test_pdf_content_is_bounded_before_text_is_extracted(monkeypatch):
    from pypdf import PageObject

    extracted = []
    monkeypatch.setattr(PageObject, "extract_text", lambda self, *a, **k: extracted.append(1) or "")
    monkeypatch.setattr(evidence, "MAX_PDF_CONTENT_BYTES", 1000)
    with pytest.raises(evidence.EvidenceError, match="of content"):
        evidence.extract(evidence.AHA, pdf_with_content(6000))
    assert extracted == []


def test_pypdfs_limits_are_lowered_for_the_read_only(monkeypatch):
    from pypdf import get_configuration

    before = get_configuration()
    seen = []
    real = evidence._read_pdf_text
    monkeypatch.setattr(evidence, "_read_pdf_text", lambda c: seen.append(get_configuration()) or real(c))
    with pytest.raises(evidence.EvidenceError):
        evidence.extract(evidence.AHA, pdf_with_content(60))  # no text
    assert seen[0].zlib_maximum_output_length == evidence.MAX_PDF_STREAM_BYTES
    assert seen[0].xform_maximum_invocations_per_extraction == 500
    assert get_configuration() == before


def test_any_pypdf_failure_is_a_rejection(monkeypatch):
    import pypdf

    def broken(*a, **k):
        raise KeyError("/Root")

    monkeypatch.setattr(pypdf, "PdfReader", broken)
    with pytest.raises(evidence.EvidenceError, match="can't be read \\(KeyError"):
        evidence.extract(evidence.AHA, b"%PDF-1.7 whatever")


def test_finishing_an_older_rerun_keeps_a_newer_ones_pending_review(monkeypatch, tickets, cases):
    case = open_case(monkeypatch, tickets, cases)
    add_jira(cases, case)
    arbiter_reply(monkeypatch, SOLVE)
    real_finish = stop_after_the_case(monkeypatch)
    with pytest.raises(RuntimeError):
        triage.rerun_case(tickets, cases, case["_id"])
    monkeypatch.setattr(triage, "finish_pending", real_finish)
    stale = cases.find_one({})
    assert triage.finish_pending(tickets, cases, stale)
    newer = {"entry": {"review_id": "newer-rerun"}, "before": {}}
    cases.update_one({"_id": case["_id"]}, {"$set": {"pending_review": newer}})

    assert triage.finish_pending(tickets, cases, stale)  # a second worker with the old snapshot

    assert cases.find_one({})["pending_review"] == newer


def test_the_cases_collection_cant_be_a_tag_collection(monkeypatch, tickets):
    monkeypatch.setenv("TRIAGE_CASES_COLLECTION", "normalized_tags")
    with pytest.raises(ValueError, match="same as NORMALIZED_TAGS_COLLECTION"):
        triage.get_cases_collection(tickets.database)


def test_an_open_case_past_the_row_limit_is_still_listed(monkeypatch, tickets, cases):
    from ui.tabs import triage_tab

    case = open_case(monkeypatch, tickets, cases)
    assert triage_tab.open_case_reviews(tickets, triage_tab.recent_reviews(tickets)) == []

    rows = triage_tab.open_case_reviews(tickets, [])  # newer reviews filled the table

    assert [row["review"]["review_id"] for row in rows] == [case["review_id"]]
    cases.update_one({"_id": case["_id"]}, {"$set": {"status": triage.RESOLVED}})
    assert triage_tab.open_case_reviews(tickets, []) == []


def test_a_sweep_that_failed_runs_again_next_refresh(monkeypatch, tickets, cases):
    monkeypatch.setattr(triage, "open_case", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(triage, "mark_case_pending", lambda *a: (_ for _ in ()).throw(RuntimeError("down")))
    review_manual_ticket(monkeypatch, tickets)
    monkeypatch.setattr(triage, "_last_sweep", None)

    assert triage.recover_missing_cases(tickets, cases) == 0
    assert triage._last_sweep is None
    monkeypatch.undo()
    monkeypatch.setattr(triage, "_last_sweep", None)
    monkeypatch.delenv("MODEL_BUCKET", raising=False)
    assert triage.recover_missing_cases(tickets, cases) == 1
    assert triage._last_sweep is not None
    assert "agent_reviews.classification_1" in tickets.index_information()


def refuse_uploads(monkeypatch):
    """Makes every trail upload fail, as B2 does for a key limited to another prefix."""
    from core import model_storage

    real_client = model_storage.make_client

    def client(cfg):
        made = real_client(cfg)
        made.put_object = lambda **_: (_ for _ in ()).throw(ConnectionError("connection closed"))
        return made

    monkeypatch.setattr(model_storage, "make_client", client)
    return real_client


@mock_aws
def test_a_failed_upload_is_kept_in_mongodb_and_moved_later(monkeypatch, tickets, cases):
    use_storage(monkeypatch)
    real_client = refuse_uploads(monkeypatch)
    review_manual_ticket(monkeypatch, tickets)
    case = cases.find_one({})
    assert case["trail"]["stored"] == "inline"
    assert "B2_KEY_ID" in case["trail"]["upload_error"]
    assert triage.kept_trail_count(cases) == 1
    with pytest.raises(triage.TrailUploadFailed, match="connection closed"):
        triage.upload_kept_trails(cases)
    assert cases.find_one({})["trail"]["stored"] == "inline"

    from core import model_storage

    monkeypatch.setattr(model_storage, "make_client", real_client)
    assert triage.upload_kept_trails(cases) == 1

    moved = cases.find_one({})
    assert moved["trail"]["stored"] == "s3" and "trail" not in moved["trail"]
    assert triage.load_trail(moved)["first"]["thinking"] == ["I weighed S3 against R1."]
    assert triage.kept_trail_count(cases) == 0
    assert triage.upload_kept_trails(cases) == 0


@mock_aws
def test_a_reruns_kept_trail_is_moved_too(monkeypatch, tickets, cases):
    case = open_case(monkeypatch, tickets, cases)  # storage off: kept inline
    add_jira(cases, case)
    arbiter_reply(monkeypatch, SOLVE)
    triage.rerun_case(tickets, cases, case["_id"])
    assert triage.kept_trail_count(cases) == 1
    use_storage(monkeypatch)

    assert triage.upload_kept_trails(cases) == 2

    stored = cases.find_one({})
    rerun = stored["reruns"][0]
    assert rerun["trail"]["key"] == f"topic-models/_triage-cases/{case['_id']}-rerun-{rerun['review_id']}.json"
    assert stored["trail"]["stored"] == "s3"


def test_retrying_without_storage_says_so(monkeypatch, cases):
    monkeypatch.delenv("MODEL_BUCKET", raising=False)
    with pytest.raises(triage.TrailUploadFailed, match="isn't set up"):
        triage.upload_kept_trails(cases)


def test_the_trail_folder_can_never_be_a_model_run():
    from core import model_storage

    with pytest.raises(ValueError):
        model_storage.check_run_id(triage.TRAIL_FOLDER.rstrip("/"))


@mock_aws
def test_a_trail_kept_on_a_pending_review_is_moved_too(monkeypatch, tickets, cases):
    real_open = triage.open_case
    monkeypatch.setattr(triage, "open_case", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    review_manual_ticket(monkeypatch, tickets)  # storage off: the trail stays on the review
    monkeypatch.setattr(triage, "open_case", real_open)
    assert triage.kept_trail_count(cases) == 0
    assert triage.kept_trail_count(cases, tickets) == 1
    use_storage(monkeypatch)

    assert triage.upload_kept_trails(cases, tickets) == 1

    case = cases.find_one({})
    assert case["trail"]["key"] == f"topic-models/_triage-cases/{case['_id']}.json"
    assert triage.load_trail(case)["first"]["thinking"] == ["I weighed S3 against R1."]
    assert triage.kept_trail_count(cases, tickets) == 0


# How Jira's export writes a comment that @mentions someone: rel twice.
MENTION_COMMENT = (
    '<comment id="11" author="pm" created="Mon, 2 Nov 2026 10:00:00 -0500">'
    '<p><a href="https://jira.example.com/ViewProfile.jspa" class="user-hover" '
    'rel="712020:abc" data-account-id="712020:abc" rel="noreferrer">Dev Person</a> '
    "fixed in the next release</p></comment>"
)


def test_a_comment_mentioning_someone_still_parses():
    xml = JIRA_XML.replace("</comments>", MENTION_COMMENT + "</comments>")
    found = evidence.extract(evidence.JIRA, xml.encode())
    (issue,) = found.data["issues"]
    assert issue["comments"][-1]["text"] == "Dev Person fixed in the next release"
    # Hashed as given, so the same export can't be added twice.
    assert found.sha256 == evidence._sha256(xml.encode())


def test_only_repeated_attributes_are_repaired():
    assert evidence._drop_duplicate_attributes(
        b'<a rel="1" b=\'2\' rel="3">x <y></y> rel="4"</a>'
    ) == b'<a rel="1" b=\'2\'>x <y></y> rel="4"</a>'
    # Another fault in a file with repeated attributes is still reported.
    broken = JIRA_XML.replace("</comments>", MENTION_COMMENT + "</comments>").replace("</channel>", "")
    with pytest.raises(evidence.EvidenceError, match="well-formed"):
        evidence.extract(evidence.JIRA, broken.encode())


def test_the_repair_leaves_cdata_and_quoted_brackets_alone():
    repair = evidence._drop_duplicate_attributes
    # CDATA is text, not markup: tag-like text in it is kept as written.
    cdata = b'<s><![CDATA[Support <input x="1" x="2"> syntax]]></s><a x="1" x="2"/>'
    assert repair(cdata) == b'<s><![CDATA[Support <input x="1" x="2"> syntax]]></s><a x="1"/>'
    # A ">" inside a quoted value doesn't end the tag.
    assert repair(b'<a title="a > b" rel="id" rel="noreferrer">Dev</a>') == (
        b'<a title="a > b" rel="id">Dev</a>'
    )
    assert repair(b"<!-- <a x='1' x='2'> --><?pi <b c='1' c='2'>?>") == (
        b"<!-- <a x='1' x='2'> --><?pi <b c='1' c='2'>?>"
    )


def test_a_cdata_summary_survives_a_repaired_export():
    xml = JIRA_XML.replace("</comments>", MENTION_COMMENT + "</comments>").replace(
        "<summary>Inline PDF viewer</summary>",
        '<summary><![CDATA[Support <input x="1" x="2"> syntax]]></summary>',
    )
    (issue,) = evidence.extract(evidence.JIRA, xml.encode()).data["issues"]
    assert issue["summary"] == 'Support <input x="1" x="2"> syntax'


def test_the_repair_takes_linear_time_on_long_whitespace():
    import time

    content = b"<a" + b" " * 2_000_000 + b"><b x='1' x='2'/></a>"
    started = time.monotonic()
    assert evidence._drop_duplicate_attributes(content).endswith(b"><b x='1'/></a>")
    assert time.monotonic() - started < 2  # the regex took ~1 s for 8,000 spaces


@pytest.mark.parametrize("encoding", ["utf-16", "utf-16-le", "utf-16-be"])
def test_a_utf16_export_with_a_mention_still_parses(encoding):
    xml = JIRA_XML.replace("</comments>", MENTION_COMMENT + "</comments>").replace(
        'encoding="UTF-8"', 'encoding="UTF-16"'
    )
    (issue,) = evidence.extract(evidence.JIRA, xml.encode(encoding)).data["issues"]
    assert issue["comments"][-1]["text"] == "Dev Person fixed in the next release"
