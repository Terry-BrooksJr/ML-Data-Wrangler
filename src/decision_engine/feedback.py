"""Feedback loop: outcomes, misclassification_log and lessons_learned (brief §4.2).

Outcomes are recorded manually for now (detection_method "manual"); per the
discovery notes TicketOutcome can't be used to backfill them (open item #9).
The accuracy report reads each review's latest outcome.
"""

import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional

from bson import ObjectId
from pymongo.collection import Collection

from .guards import CLASSIFICATIONS

OUTCOMES = ("good", "bad", "pending")
DETECTION_METHODS = ("automatic", "manual")
MISCLASSIFICATION_COLLECTION = os.environ.get(
    "MISCLASSIFICATION_COLLECTION", "misclassification_log"
)
LESSONS_COLLECTION = os.environ.get("LESSONS_COLLECTION", "lessons_learned")


def week_of(moment: datetime) -> datetime:
    """Midnight UTC on the Monday of `moment`'s week."""
    moment = (
        moment.astimezone(timezone.utc)
        if moment.tzinfo
        else moment.replace(tzinfo=timezone.utc)
    )
    monday = moment - timedelta(days=moment.weekday())
    return monday.replace(hour=0, minute=0, second=0, microsecond=0)


def _as_object_id(review_id) -> ObjectId:
    return review_id if isinstance(review_id, ObjectId) else ObjectId(str(review_id))


def find_review(tickets: Collection, zendesk_id: str, review_id) -> dict:
    """Returns one agent_reviews entry.

    Raises:
        KeyError: if the ticket or review doesn't exist.
    """
    review_id = _as_object_id(review_id)
    doc = tickets.find_one({"zendesk_id": zendesk_id}, {"agent_reviews": 1})
    for review in (doc or {}).get("agent_reviews", []):
        if review.get("review_id") == review_id:
            return review
    raise KeyError(f"Review {review_id} not found on ticket {zendesk_id}")


def record_outcome(
    tickets: Collection,
    zendesk_id: str,
    review_id,
    outcome: str,
    detail: str = "",
    detection_method: str = "manual",
) -> dict:
    """Appends an outcome for one review to the ticket's `outcomes`.

    Returns:
        The stored outcome entry.
    """
    if outcome not in OUTCOMES:
        raise ValueError(f"outcome must be one of {OUTCOMES}, got {outcome!r}")
    if detection_method not in DETECTION_METHODS:
        raise ValueError(
            f"detection_method must be one of {DETECTION_METHODS}, got {detection_method!r}"
        )
    review = find_review(tickets, zendesk_id, review_id)
    entry = {
        "review_id": review["review_id"],
        "outcome": outcome,
        "detected_at": datetime.now(timezone.utc),
        "detection_method": detection_method,
        "detail": detail,
    }
    tickets.update_one({"zendesk_id": zendesk_id}, {"$push": {"outcomes": entry}})
    return entry


def log_misclassification(
    tickets: Collection,
    log: Collection,
    zendesk_id: str,
    review_id,
    correct_classification: str,
    misread_signal: str = "",
) -> ObjectId:
    """Records that a review was wrong, and marks its outcome "bad".

    Returns:
        The misclassification_log entry's _id.
    """
    if correct_classification not in CLASSIFICATIONS:
        raise ValueError(f"correct_classification must be one of {CLASSIFICATIONS}")
    review = find_review(tickets, zendesk_id, review_id)
    now = datetime.now(timezone.utc)
    log_id = log.insert_one(
        {
            "ticket_id": zendesk_id,
            "review_id": review["review_id"],
            "agent_classification": review.get("classification"),
            "correct_classification": correct_classification,
            "misread_signal": misread_signal,
            "logged_at": now,
            "reviewed": False,
            "reviewed_by": None,
            "reviewed_at": None,
            "week_of": week_of(now),
        }
    ).inserted_id
    record_outcome(
        tickets,
        zendesk_id,
        review["review_id"],
        "bad",
        detail=f"Should have been {correct_classification}"
        + (f"; misread: {misread_signal}" if misread_signal else ""),
    )
    return log_id


def mark_misclassification_reviewed(log: Collection, log_id, reviewed_by: str) -> None:
    """Marks a log entry as reviewed; only reviewed entries feed lessons_learned."""
    mark_misclassifications_reviewed(log, [log_id], reviewed_by)


def mark_misclassifications_reviewed(
    log: Collection, log_ids: Iterable, reviewed_by: str
) -> int:
    """Marks log entries as reviewed by `reviewed_by`. Entries already reviewed keep their reviewer.

    Returns:
        How many entries were newly marked.
    """
    reviewed_by = reviewed_by.strip()
    if not reviewed_by:
        raise ValueError("reviewed_by is required")
    ids = [_as_object_id(i) for i in log_ids]
    if not ids:
        return 0
    return log.update_many(
        {"_id": {"$in": ids}, "reviewed": {"$ne": True}},
        {
            "$set": {
                "reviewed": True,
                "reviewed_by": reviewed_by,
                "reviewed_at": datetime.now(timezone.utc),
            }
        },
    ).modified_count


def misclassifications_for_week(log: Collection, week: datetime) -> List[dict]:
    """The week's misclassification_log entries, oldest first."""
    return list(log.find({"week_of": week_of(week)}).sort("logged_at", 1))


def lessons_for_week(lessons: Collection, week: datetime) -> List[dict]:
    """The week's lessons_learned entries (drafts and approved), newest first."""
    return list(lessons.find({"week_of": week_of(week)}).sort("generated_at", -1))


def suggest_lessons_summary(entries: Iterable[dict]) -> str:
    """A starting summary for the week's reviewed entries, for the reviewer to edit.

    Groups the mistakes by what the agent said and what was right, with the
    signals it misread. Unreviewed entries are left out, as in draft_lessons_learned.
    """
    groups: Dict[tuple, List[str]] = defaultdict(list)
    for entry in entries:
        if entry.get("reviewed"):
            groups[
                (entry.get("agent_classification"), entry.get("correct_classification"))
            ].append((entry.get("misread_signal") or "").strip())
    if not groups:
        return ""
    lines = []
    for (said, correct), signals in sorted(
        groups.items(), key=lambda g: (-len(g[1]), str(g[0]))
    ):
        line = (
            f"- {len(signals)} ticket(s) classified {said} should have been {correct}."
        )
        misread = sorted({s for s in signals if s})
        if misread:
            line += " Misread: " + "; ".join(misread) + "."
        lines.append(line)
    return "\n".join(lines)


def draft_lessons_learned(
    log: Collection, lessons: Collection, week: datetime, summary_text: str
) -> ObjectId:
    """Stores an unapproved lessons_learned entry built on that week's reviewed log entries.

    Raises:
        ValueError: if the week has no reviewed entries.
    """
    week = week_of(week)
    if not summary_text.strip():
        raise ValueError("summary_text is required")
    log_ids = [
        entry["_id"]
        for entry in log.find({"week_of": week, "reviewed": True}, {"_id": 1})
    ]
    if not log_ids:
        raise ValueError(
            f"No reviewed misclassifications for the week of {week:%Y-%m-%d}"
        )
    return lessons.insert_one(
        {
            "week_of": week,
            "summary_text": summary_text.strip(),
            "generated_at": datetime.now(timezone.utc),
            "based_on_log_ids": log_ids,
            "approved": False,
            "approved_by": None,
            "approved_at": None,
        }
    ).inserted_id


def approve_lessons_learned(lessons: Collection, lessons_id, approved_by: str) -> bool:
    """Approves a draft. An entry already approved keeps its approver.

    Returns:
        True if the draft was approved now, False if it already was (or doesn't exist).
    """
    approved_by = approved_by.strip()
    if not approved_by:
        raise ValueError("approved_by is required")
    return (
        lessons.update_one(
            {"_id": _as_object_id(lessons_id), "approved": {"$ne": True}},
            {
                "$set": {
                    "approved": True,
                    "approved_by": approved_by,
                    "approved_at": datetime.now(timezone.utc),
                }
            },
        ).modified_count
        == 1
    )


def _latest_outcomes(doc: dict) -> Dict[ObjectId, str]:
    # Outcomes are only ever appended, so the last one per review is the latest.
    return {
        entry.get("review_id"): entry.get("outcome")
        for entry in doc.get("outcomes") or []
    }


def accuracy_report(
    tickets: Collection, group_by: str = "classification"
) -> List[dict]:
    """Accuracy of reviews, grouped by classification or by ticket topic.

    Args:
        group_by: "classification", or "topic" to group by the ticket's main
            topic from its latest topic-model run (topic.run_id/main_topic).

    Returns:
        One row per group: reviews, good, bad, pending, unrated and accuracy
        (good / (good + bad), or None when nothing is rated).
    """
    if group_by not in ("classification", "topic"):
        raise ValueError("group_by must be 'classification' or 'topic'")
    rows: Dict[str, dict] = defaultdict(
        lambda: dict(reviews=0, good=0, bad=0, pending=0, unrated=0)
    )
    projection = {"agent_reviews": 1, "outcomes": 1, "topic": 1}
    for doc in tickets.find({"agent_reviews.0": {"$exists": True}}, projection):
        latest = _latest_outcomes(doc)
        topic = doc.get("topic") or {}
        topic_key = (
            f"{topic['run_id']} / topic {topic['main_topic']}"
            if topic.get("run_id")
            else "(no topic)"
        )
        for review in doc["agent_reviews"]:
            key = (
                review.get("classification")
                if group_by == "classification"
                else topic_key
            )
            row = rows[key]
            row["reviews"] += 1
            row[latest.get(review.get("review_id"), "unrated")] += 1
    report = []
    for key, row in rows.items():
        rated = row["good"] + row["bad"]
        report.append(
            {group_by: key, **row, "accuracy": row["good"] / rated if rated else None}
        )
    order = {c: i for i, c in enumerate(CLASSIFICATIONS)}
    return sorted(
        report, key=lambda r: (order.get(r[group_by], len(order)), str(r[group_by]))
    )


def format_accuracy_report(
    report: Iterable[dict], group_by: str = "classification"
) -> str:
    lines = [f"{group_by:<28} reviews  good  bad  pending  unrated  accuracy"]
    for row in report:
        accuracy = f"{row['accuracy']:.0%}" if row["accuracy"] is not None else "-"
        lines.append(
            f"{str(row[group_by]):<28} {row['reviews']:>7} {row['good']:>5} {row['bad']:>4} "
            f"{row['pending']:>8} {row['unrated']:>8} {accuracy:>9}"
        )
    return "\n".join(lines)


def get_feedback_collections(database) -> tuple:
    """(misclassification_log, lessons_learned) in the given database."""
    return database[MISCLASSIFICATION_COLLECTION], database[LESSONS_COLLECTION]


def recent_misclassifications(
    log: Collection, limit: int = 50, since: Optional[datetime] = None
) -> List[dict]:
    query = {"logged_at": {"$gte": since}} if since else {}
    return list(log.find(query).sort("logged_at", -1).limit(limit))
