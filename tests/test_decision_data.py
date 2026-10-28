"""What the model is told about a ticket's authors and Jira links."""

import json

import mongomock
import pytest
from decision_engine import authors, jira_links
from decision_engine.payload import build_ticket_payload
from fixtures import CATEGORY_FIXTURES
from test_wrangler_mongo import data_dir, wrangler  # noqa: F401 (fixtures)

STEPHEN = "384720527257"  # on the leadership roster


def payload_data(ticket):
    return json.loads(build_ticket_payload(ticket).split("\n\n", 1)[1])


@pytest.fixture
def tickets():
    return mongomock.MongoClient().support_db.tickets


# --- authors ---------------------------------------------------------------


@pytest.mark.parametrize(
    "author, stored, expected",
    [
        ("555", "requester", "requester"),
        (STEPHEN, None, authors.SUPPORT),
        (int(STEPHEN), None, authors.SUPPORT),
        ("12289975534365", None, authors.SUPPORT),
        (authors.JIRA_INTEGRATION_USER_ID, None, authors.JIRA_INTEGRATION),
        ("555", None, None),
        (None, None, None),
    ],
)
def test_author_role(author, stored, expected):
    assert authors.author_role(author, stored) == expected


def test_author_role_uses_internal_authors(monkeypatch):
    monkeypatch.setitem(authors.INTERNAL_AUTHORS, "777", "Product")
    assert authors.author_role("777") == authors.INTERNAL


def test_payload_fills_in_author_roles():
    ticket = {
        "zendesk_id": "1",
        "transcript": [
            {"author": "555", "role": "requester", "body": "Help"},
            {"author": STEPHEN, "role": None, "body": "On it"},
            {"author": authors.JIRA_INTEGRATION_USER_ID, "role": None, "body": "Linked"},
            {"author": "888", "role": None, "body": "Me too"},
        ],
    }
    roles = [c["role"] for c in payload_data(ticket)["transcript"]]
    assert roles == ["requester", "support", "jira_integration", None]


def test_internal_only_candidates_skip_anyone_with_a_public_comment(tickets):
    def note(author, internal, role=None):
        return {"author": author, "role": role, "is_internal_note": internal}

    tickets.insert_many(
        [
            {
                "zendesk_id": "1",
                "transcript": [
                    note("700", True),
                    note("700", True),
                    note("800", True),
                    note("800", False),  # public: could be a customer
                    note(STEPHEN, True),  # already known
                    note("555", True, role="requester"),
                ],
            },
            {"zendesk_id": "2", "transcript": [note("700", True), note("701", True)]},
        ]
    )
    assert authors.internal_only_candidates(tickets) == [
        {"author": "700", "comments": 3, "tickets": 2},
        {"author": "701", "comments": 1, "tickets": 1},
    ]


# --- Jira links ------------------------------------------------------------


def link(ticket_id, issue_id, key=None, created="2016-07-01T10:23:12.000Z"):
    return {
        "id": 1,
        "ticket_id": ticket_id,
        "issue_id": issue_id,
        "issue_key": key,
        "url": "https://example.zendesk.com/api/services/jira/links/1",
        "created_at": created,
        "updated_at": created,
    }


def test_read_export_takes_a_list_or_a_page(tmp_path):
    as_list, as_page = tmp_path / "list.json", tmp_path / "page.json"
    as_list.write_text(json.dumps([link("1", "10")]))
    as_page.write_text(json.dumps({"links": [link("1", "10")], "meta": {}}))
    assert jira_links.read_export(str(as_list)) == jira_links.read_export(str(as_page))

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"tickets": []}))
    with pytest.raises(ValueError):
        jira_links.read_export(str(bad))


def test_links_by_ticket_keeps_one_per_issue_preferring_a_key():
    by_ticket = jira_links.links_by_ticket(
        [
            link("2", "20", created="2016-01-02T00:00:00Z"),
            link("2", "21", created="2016-01-01T00:00:00Z"),
            link("2", "20", key="LRN-1", created="2016-01-02T00:00:00Z"),
            link("3", None),
        ]
    )
    assert by_ticket == {
        "2": [
            {"issue_id": "21", "issue_key": None, "linked_at": "2016-01-01T00:00:00Z"},
            {"issue_id": "20", "issue_key": "LRN-1", "linked_at": "2016-01-02T00:00:00Z"},
        ]
    }


def test_store_links_updates_stored_tickets_only(tickets):
    tickets.insert_one({"zendesk_id": "2"})
    by_ticket = jira_links.links_by_ticket([link("2", "20", "LRN-1"), link("9", "90")])

    assert jira_links.store_links(tickets, by_ticket) == 1
    assert tickets.find_one({"zendesk_id": "2"})["jira_links"] == by_ticket["2"]
    assert tickets.find_one({"zendesk_id": "9"}) is None


def test_payload_shows_links_without_status():
    ticket = dict(CATEGORY_FIXTURES["flag_solve"])
    ticket["jira_links"] = [
        {"issue_id": "20", "issue_key": "LRN-1", "linked_at": "2016-01-02"},
        {"issue_id": "21", "issue_key": None, "linked_at": "2016-01-03"},
    ]
    assert payload_data(ticket)["jira_links"] == [
        {"issue": "LRN-1", "linked_at": "2016-01-02"},
        {"issue": "issue 21", "linked_at": "2016-01-03"},
    ]
    assert "jira_ids" not in payload_data(ticket)


def test_reingesting_a_ticket_keeps_its_jira_links(wrangler, tickets):  # noqa: F811
    wrangler.write_ticket_to_mongo(101, tickets)
    links = jira_links.links_by_ticket([link("101", "20", "LRN-1")])
    jira_links.store_links(tickets, links)

    wrangler.write_ticket_to_mongo(101, tickets)

    assert tickets.find_one({"zendesk_id": "101"})["jira_links"] == links["101"]
