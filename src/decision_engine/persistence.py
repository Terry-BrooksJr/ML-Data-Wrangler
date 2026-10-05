import os
from datetime import datetime, timezone
from functools import lru_cache

from bson import ObjectId
from loguru import logger
from pymongo import MongoClient
from pymongo.collection import Collection

from .classify import classify_ticket
from .payload import build_ticket_payload


@lru_cache(maxsize=1)
def get_tickets_collection() -> Collection:
    """Connects on first use so importing this module needs no Mongo access.

    Database and collection default to the brief's support_db.tickets until
    real names are confirmed.
    """
    mongo = MongoClient(os.environ["MONGO_URI"])
    database = mongo[os.environ.get("MONGO_DB", "support_db")]
    return database[os.environ.get("MONGO_COLLECTION", "tickets")]


def process_one_ticket(zendesk_id: str, tickets: Collection = None):
    """
    Reads one awaiting-review ticket, classifies it, and appends the result
    to agent_reviews. Does NOT call Zendesk — tagging/noting the real ticket
    is Module 3 (write-back), kept strictly separate so dry-run mode stays
    meaningful and the read-only/write-back boundary isn't blurred.
    """
    tickets = tickets if tickets is not None else get_tickets_collection()
    ticket_doc = tickets.find_one({"zendesk_id": zendesk_id, "awaiting_review": True})
    if not ticket_doc:
        return None

    payload = build_ticket_payload(ticket_doc)
    result = classify_ticket(payload)

    review_entry = {
        "review_id": ObjectId(),  # outcomes[].review_id references this
        "reviewed_at": datetime.now(timezone.utc),
        "model_version": result["model_version"],
        "classification": result["classification"],
        "confidence": result["confidence"],
        "signals_matched": result.get("signals_matched", []),
        "reasoning": result.get("reasoning", ""),
        "tag_applied": result.get("tag"),
        "internal_note": result.get("internal_note", ""),
        "draft_message": result.get("draft_message"),
        "leadership_recipient": result.get("leadership_recipient"),
    }

    # Matching on awaiting_review again means a concurrent run that already
    # reviewed this ticket can't append a second entry.
    tickets.update_one(
        {"zendesk_id": zendesk_id, "awaiting_review": True},
        {"$push": {"agent_reviews": review_entry}, "$set": {"awaiting_review": False}},
    )
    return result


def run_review_batch(tickets: Collection = None):
    tickets = tickets if tickets is not None else get_tickets_collection()
    zendesk_ids = [doc["zendesk_id"] for doc in tickets.find({"awaiting_review": True}, {"zendesk_id": 1})]
    for zendesk_id in zendesk_ids:
        result = process_one_ticket(zendesk_id, tickets)
        if result is None:
            logger.info(f"{zendesk_id}: already reviewed, skipped")
            continue
        logger.info(f"{zendesk_id}: {result['classification']} ({result['confidence']})")


if __name__ == "__main__":
    run_review_batch()
