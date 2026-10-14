import json
from datetime import datetime, timezone

import mongomock
import pytest

from decision_engine.payload import build_ticket_payload
from wrangler import DataWrangler

TICKET = {
    "id": 101,
    "subject": "Import fails",
    "description": "The import fails.",
    "via": {"channel": "web"},
    "status": "open",
    "type": "incident",
    "priority": "high",
    "group_id": 24165535,
    "assignee_id": 384720527257,
    "requester_id": 555,
    "organization_id": 777,
    "email_cc_ids": [888],
    "tags": ["enterprise"],
    "created_at": "2026-08-18T18:53:13Z",
    "updated_at": "2026-08-19T20:01:24Z",
    "fields": [
        {"id": 27124341, "value": "developer_support"},
        {"id": 27159242, "value": None},
        {"id": 27219391, "value": "learnosity_bug"},
    ],
}


def comment(comment_id, author_id, body, created_at, public=True):
    return {
        "id": comment_id,
        "type": "Comment",
        "author_id": author_id,
        "plain_body": body,
        "created_at": created_at,
        "public": public,
    }


COMMENTS = [
    comment(1, 555, "The import fails.", "2026-08-18T18:53:13Z"),
    comment(2, 384720527257, "Escalated to engineering.", "2026-08-19T09:00:00Z", public=False),
]


@pytest.fixture
def data_dir(tmp_path):
    (tmp_path / "comments").mkdir()
    (tmp_path / "tickets-only.json").write_text(json.dumps([TICKET]))
    (tmp_path / "comments" / "101.json").write_text(json.dumps(COMMENTS))
    return tmp_path


@pytest.fixture
def wrangler(data_dir):
    return DataWrangler(comments_dir=data_dir / "comments", ticket_file=data_dir / "tickets-only.json")


@pytest.fixture
def tickets():
    return mongomock.MongoClient().support_db.tickets


def test_document_maps_fields(wrangler):
    doc = wrangler.ticket_to_document(101)

    assert doc["zendesk_id"] == "101"
    assert doc["assignee"] == "384720527257"
    assert doc["priority"] == "high"
    assert doc["request_type"] == "developer_support"
    assert doc["historical_outcome"] == "learnosity_bug"
    assert doc["organization_id"] == "777"
    assert doc["customer_id"] is None
    assert doc["created_at"] == datetime(2026, 8, 18, 18, 53, 13, tzinfo=timezone.utc)
    assert doc["raw_xml"].startswith('<?xml version="1.0"')

    first, second = doc["transcript"]
    assert first["role"] == "requester" and first["is_internal_note"] is False
    assert second["role"] is None and second["is_internal_note"] is True
    assert second["comment_id"] == "2"


def test_insert_marks_ticket_awaiting_review(wrangler, tickets):
    summary = wrangler.write_ticket_to_mongo(101, tickets)

    assert summary == {"zendesk_id": "101", "inserted": True, "new_comments": 2}
    stored = tickets.find_one({"zendesk_id": "101"})
    assert stored["awaiting_review"] is True
    assert stored["agent_reviews"] == [] and stored["reengage_attempts"] == []


def test_rewrite_is_idempotent(wrangler, tickets):
    wrangler.write_ticket_to_mongo(101, tickets)
    tickets.update_one({"zendesk_id": "101"}, {"$set": {"awaiting_review": False}})

    summary = wrangler.write_ticket_to_mongo(101, tickets)

    assert summary["new_comments"] == 0
    assert tickets.count_documents({}) == 1
    stored = tickets.find_one({"zendesk_id": "101"})
    assert len(stored["transcript"]) == 2
    assert stored["awaiting_review"] is False


def test_new_comments_append_and_reopen_review(wrangler, tickets, data_dir):
    wrangler.write_ticket_to_mongo(101, tickets)
    tickets.update_one(
        {"zendesk_id": "101"},
        {"$set": {"awaiting_review": False}, "$push": {"agent_reviews": {"classification": "flag_reengage"}}},
    )
    updated_comments = COMMENTS + [comment(3, 555, "Any update?", "2026-08-25T10:00:00Z")]
    (data_dir / "comments" / "101.json").write_text(json.dumps(updated_comments))
    updated_ticket = dict(TICKET, status="pending", priority="urgent")

    summary = wrangler.write_ticket_to_mongo(updated_ticket, tickets)

    assert summary["new_comments"] == 1
    stored = tickets.find_one({"zendesk_id": "101"})
    assert [entry["comment_id"] for entry in stored["transcript"]] == ["1", "2", "3"]
    assert stored["status"] == "pending" and stored["priority"] == "urgent"
    assert stored["awaiting_review"] is True
    assert len(stored["agent_reviews"]) == 1  # review history is never touched


def test_rewrite_is_idempotent_without_comment_ids(wrangler, tickets, data_dir):
    # An empty comment file falls back to the description, which has no
    # Zendesk comment ID, so de-duplication compares author/date/text.
    (data_dir / "comments" / "101.json").write_text("[]")
    wrangler.write_ticket_to_mongo(101, tickets)

    summary = wrangler.write_ticket_to_mongo(101, tickets)

    assert summary["new_comments"] == 0
    assert len(tickets.find_one({"zendesk_id": "101"})["transcript"]) == 1


def test_write_all_tickets(wrangler, tickets):
    summaries = wrangler.write_tickets_to_mongo(collection=tickets)

    assert [s["zendesk_id"] for s in summaries] == ["101"]


def test_stored_document_feeds_decision_engine(wrangler, tickets):
    wrangler.write_ticket_to_mongo(101, tickets)

    payload = build_ticket_payload(tickets.find_one({"zendesk_id": "101"}))

    data = json.loads(payload.split("\n\n", 1)[1])
    assert data["ticket_id"] == "101"
    assert len(data["transcript"]) == 2


# --- HTML entity cleaning ------------------------------------------------------

from wrangler import clean_text  # noqa: E402


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Hi&nbsp;team,&nbsp;\nthanks", "Hi team,\nthanks"),
        ("R&amp;D &lt;script&gt; &quot;x&quot; &#39;y&#39; &#8217;", "R&D <script> \"x\" 'y' \u2019"),
        ("double&amp;nbsp;escaped &amp;amp;", "double escaped &"),
        ("zero\u200bwidth\ufeff\u2060 and\u00a0nbsp\u202fnarrow", "zerowidth and nbsp narrow"),
        ("trailing   \nline two\t ", "trailing\nline two"),
        ("keeps\n\nparagraphs", "keeps\n\nparagraphs"),
        ("def f():\n    if x:\n\treturn  1  # aligned", "def f():\n    if x:\n\treturn  1  # aligned"),
        ("crlf\r\nand\rcr", "crlf\nand\ncr"),
        ("esc &#27;]52;c;evil&#7; bell \x1b[31mred\x9b", "esc ]52;c;evil bell [31mred"),
        ("family \U0001F468\u200d\U0001F469 and \u0645\u200c\u06cc", "family \U0001F468\u200d\U0001F469 and \u0645\u200c\u06cc"),
        ("", ""),
        (None, None),
    ],
    ids=["nbsp", "named_and_numeric", "double_escaped", "unicode_spaces", "trailing_spaces", "newlines",
         "indentation_kept", "line_endings", "control_chars", "joiners_kept", "empty", "none"],
)
def test_clean_text(raw, expected):
    assert clean_text(raw) == expected


def test_clean_text_is_idempotent():
    once = clean_text("A&amp;amp;B&nbsp;&nbsp;C")
    assert clean_text(once) == once


def entity_data_dir(data_dir, comment_ids=True):
    ticket = {**TICKET, "subject": "Import&nbsp;fails &amp; stalls"}
    comments = [
        comment(1 if comment_ids else None, 555, "The&nbsp;import fails.&nbsp;\n&lt;details&gt;", "2026-08-18T18:53:13Z"),
    ]
    (data_dir / "tickets-only.json").write_text(json.dumps([ticket]))
    (data_dir / "comments" / "101.json").write_text(json.dumps(comments))
    return data_dir


def test_ticket_text_is_cleaned_when_written(wrangler, tickets, data_dir):
    entity_data_dir(data_dir)
    wrangler.write_ticket_to_mongo(101, tickets)

    stored = tickets.find_one({"zendesk_id": "101"})
    assert stored["subject"] == "Import fails & stalls"
    assert stored["transcript"][0]["body"] == "The import fails.\n<details>"
    assert "&nbsp;" in stored["raw_xml"] or "&amp;nbsp;" in stored["raw_xml"]  # archive keeps the original


def test_reingest_matches_bodies_stored_before_cleaning(wrangler, tickets, data_dir):
    entity_data_dir(data_dir, comment_ids=False)
    wrangler.write_ticket_to_mongo(101, tickets)
    # Simulate a ticket stored before cleaning existed.
    tickets.update_one({"zendesk_id": "101"}, {"$set": {
        "transcript.0.body": "The&nbsp;import fails.&nbsp;\n&lt;details&gt;", "awaiting_review": False}})

    summary = wrangler.write_ticket_to_mongo(101, tickets)

    stored = tickets.find_one({"zendesk_id": "101"})
    assert summary["new_comments"] == 0
    assert len(stored["transcript"]) == 1 and stored["awaiting_review"] is False


def test_clean_stored_tickets_backfills(tickets):
    tickets.insert_many([
        {"zendesk_id": "1", "subject": "A&nbsp;B", "awaiting_review": False,
         "transcript": [{"body": "clean"}, {"body": "x&amp;y"}, {"body": None}]},
        {"zendesk_id": "2", "subject": "Already clean", "transcript": [{"body": "fine"}]},
    ])

    preview = DataWrangler.clean_stored_tickets(tickets, dry_run=True)
    assert preview == {"scanned": 2, "tickets_changed": 1, "comments_changed": 1, "skipped_changed": 0}
    assert tickets.find_one({"zendesk_id": "1"})["subject"] == "A&nbsp;B"  # dry run wrote nothing

    assert DataWrangler.clean_stored_tickets(tickets) == preview
    first = tickets.find_one({"zendesk_id": "1"})
    assert first["subject"] == "A B"
    assert [c["body"] for c in first["transcript"]] == ["clean", "x&y", None]
    assert first["awaiting_review"] is False  # review state untouched

    assert DataWrangler.clean_stored_tickets(tickets)["tickets_changed"] == 0  # idempotent


def test_concurrent_ingest_appends_a_comment_once(wrangler, tickets, data_dir, monkeypatch):
    wrangler.write_ticket_to_mongo(101, tickets)
    stale = tickets.find_one({"zendesk_id": "101"}, {"transcript": 1})
    new = comment(3, 555, "Any update?", "2026-08-20T09:00:00Z")
    (data_dir / "comments" / "101.json").write_text(json.dumps(COMMENTS + [new]))

    # Both writers read the transcript before either appends.
    real_find_one = tickets.find_one
    monkeypatch.setattr(tickets, "find_one", lambda f, p=None: stale if p == {"transcript": 1} else real_find_one(f, p))
    first = wrangler.write_ticket_to_mongo(101, tickets)
    second = wrangler.write_ticket_to_mongo(101, tickets)

    transcript = real_find_one({"zendesk_id": "101"})["transcript"]
    assert [c["comment_id"] for c in transcript] == ["1", "2", "3"]
    assert first["new_comments"] == 1 and second["new_comments"] == 0


def test_concurrent_ingest_without_comment_ids(wrangler, tickets, data_dir, monkeypatch):
    entity_data_dir(data_dir, comment_ids=False)
    wrangler.write_ticket_to_mongo(101, tickets)
    tickets.update_one({"zendesk_id": "101"}, {"$set": {"transcript": []}})
    stale = tickets.find_one({"zendesk_id": "101"}, {"transcript": 1})

    real_find_one = tickets.find_one
    monkeypatch.setattr(tickets, "find_one", lambda f, p=None: stale if p == {"transcript": 1} else real_find_one(f, p))
    wrangler.write_ticket_to_mongo(101, tickets)
    wrangler.write_ticket_to_mongo(101, tickets)

    assert len(real_find_one({"zendesk_id": "101"})["transcript"]) == 1


def test_reingest_cleans_bodies_stored_by_older_versions(wrangler, tickets, data_dir):
    wrangler.write_ticket_to_mongo(101, tickets)
    tickets.update_one({"zendesk_id": "101"}, {"$set": {"transcript.1.body": "Escalated&nbsp;to engineering."}})

    summary = wrangler.write_ticket_to_mongo(101, tickets)

    assert summary["new_comments"] == 0
    assert tickets.find_one({"zendesk_id": "101"})["transcript"][1]["body"] == "Escalated to engineering."


def test_backfill_does_not_revert_newer_text(tickets, monkeypatch):
    tickets.insert_one({"zendesk_id": "1", "subject": "Old&nbsp;subject", "transcript": [{"body": "a&amp;b"}]})
    real_update_one = tickets.update_one

    def ingest_lands_first(query, update, *args, **kwargs):
        # Another writer updates the subject after the backfill read it.
        real_update_one({"zendesk_id": "1"}, {"$set": {"subject": "Newer subject"}})
        return real_update_one(query, update, *args, **kwargs)

    monkeypatch.setattr(tickets, "update_one", ingest_lands_first)
    summary = DataWrangler.clean_stored_tickets(tickets)

    stored = tickets.find_one({"zendesk_id": "1"})
    assert stored["subject"] == "Newer subject"
    assert summary["skipped_changed"] == 1 and summary["tickets_changed"] == 0
