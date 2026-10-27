"""Whether a ticket's review has been copied into Zendesk by hand.

The engine never writes to Zendesk, so a reviewer pastes the review in
themselves and then marks it with the Review Added to Zendesk button. The
flag is a boolean on the ticket document, False until then; new tickets are
stored with it (wrangler.write_ticket_to_mongo) and backfill() adds it to
tickets stored before it existed. A ticket can be reviewed more than once,
so marking also records which review was added, in ADDED_REVIEWS_KEY.

Backfill the tickets collection once with:

    python -m decision_engine.zendesk_status
"""

from typing import Any

from pymongo.collection import Collection

REVIEW_ADDED_KEY = "Review Added to Zendesk?"
# The review_ids of the ticket's reviews that were added to Zendesk.
ADDED_REVIEWS_KEY = "reviews_added_to_zendesk"


def review_added(ticket: dict, review_id: Any = None) -> bool:
    """Whether the ticket, or one of its reviews, was added to Zendesk.

    With a review_id, only that review counts, except on tickets marked
    before reviews were tracked: their flag covers every review. Tickets
    without the flag count as not added.
    """
    added = ticket.get(ADDED_REVIEWS_KEY)
    if review_id is not None and added:
        return review_id in added
    return bool(ticket.get(REVIEW_ADDED_KEY))


def mark_review_added(
    tickets: Collection, zendesk_id: str, review_id: Any = None
) -> bool:
    """Sets the flag on one ticket and, given its review_id, records that review.

    Returns:
        False if no ticket has that ID.
    """
    update: dict = {"$set": {REVIEW_ADDED_KEY: True}}
    if review_id is not None:
        update["$addToSet"] = {ADDED_REVIEWS_KEY: review_id}
    result = tickets.update_one({"zendesk_id": zendesk_id}, update)
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
