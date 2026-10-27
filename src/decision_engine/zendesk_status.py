"""Whether a ticket's review has been copied into Zendesk by hand.

The engine never writes to Zendesk, so a reviewer pastes the review in
themselves and then marks the ticket with the Review Added to Zendesk
button. The flag is a boolean on the ticket document, False until then;
new tickets are stored with it (wrangler.write_ticket_to_mongo) and
backfill() adds it to tickets stored before it existed.

Backfill the tickets collection once with:

    python -m decision_engine.zendesk_status
"""

from pymongo.collection import Collection

REVIEW_ADDED_KEY = "Review Added to Zendesk?"


def review_added(ticket: dict) -> bool:
    """The ticket's flag; tickets without it count as not added."""
    return bool(ticket.get(REVIEW_ADDED_KEY))


def mark_review_added(tickets: Collection, zendesk_id: str) -> bool:
    """Sets the flag on one ticket.

    Returns:
        False if no ticket has that ID.
    """
    result = tickets.update_one(
        {"zendesk_id": zendesk_id}, {"$set": {REVIEW_ADDED_KEY: True}}
    )
    return result.matched_count > 0


def backfill(tickets: Collection) -> int:
    """Adds the flag, set to False, to every ticket that lacks it. Tickets
    already marked are left alone, so it is safe to re-run.

    Returns:
        How many tickets were updated.
    """
    return tickets.update_many(
        {REVIEW_ADDED_KEY: {"$exists": False}}, {"$set": {REVIEW_ADDED_KEY: False}}
    ).modified_count


if __name__ == "__main__":
    from core import settings

    from .persistence import get_tickets_collection

    settings.load()
    print(f"Backfilled {backfill(get_tickets_collection())} tickets.")
