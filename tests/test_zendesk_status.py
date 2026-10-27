import mongomock

from decision_engine import zendesk_status
from decision_engine.zendesk_status import REVIEW_ADDED_KEY


def test_backfill_adds_false_and_keeps_marked_tickets():
    tickets = mongomock.MongoClient().db.tickets
    tickets.insert_many(
        [
            {"zendesk_id": "1"},
            {"zendesk_id": "2", REVIEW_ADDED_KEY: True},
            {"zendesk_id": "3"},
        ]
    )

    assert zendesk_status.backfill(tickets) == 2
    assert zendesk_status.backfill(tickets) == 0
    flags = {t["zendesk_id"]: t[REVIEW_ADDED_KEY] for t in tickets.find()}
    assert flags == {"1": False, "2": True, "3": False}


def test_mark_review_added():
    tickets = mongomock.MongoClient().db.tickets
    tickets.insert_one({"zendesk_id": "1", REVIEW_ADDED_KEY: False})

    assert zendesk_status.mark_review_added(tickets, "1")
    assert zendesk_status.review_added(tickets.find_one({"zendesk_id": "1"}))
    assert not zendesk_status.mark_review_added(tickets, "missing")
    assert not zendesk_status.review_added({})


def test_mark_review_added_tracks_each_review():
    tickets = mongomock.MongoClient().db.tickets
    tickets.insert_one({"zendesk_id": "1", REVIEW_ADDED_KEY: False})

    assert zendesk_status.mark_review_added(tickets, "1", "r1")
    ticket = tickets.find_one({"zendesk_id": "1"})
    assert zendesk_status.review_added(ticket)
    assert zendesk_status.review_added(ticket, "r1")
    assert not zendesk_status.review_added(ticket, "r2")


def test_flag_set_before_reviews_were_tracked_covers_every_review():
    ticket = {"zendesk_id": "1", REVIEW_ADDED_KEY: True}
    assert zendesk_status.review_added(ticket, "r1")
    assert zendesk_status.review_added(ticket, "r2")
