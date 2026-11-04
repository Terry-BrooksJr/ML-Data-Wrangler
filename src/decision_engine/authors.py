"""Who wrote each comment, as far as the stored IDs tell.

Ingestion only knows the requester (wrangler.ticket_to_document); every
other author's role is stored as None. The payload fills that in from IDs
we know (author_role), so the model can tell Support's comments, the Jira
integration's and the customer's apart, which triggers like "Support asked
for info" and "Jira integration note confirms the release" rely on. For
Learnosity staff it also attaches who they are (author_staff, from
staff.ROSTER), so the model can weigh a comment by department and seniority.

To find staff missing from staff.py, list candidates with

    python -m decision_engine.authors candidates

which prints only IDs and counts, never names or comment text, for authors
who have only ever written internal notes, so no customer is looked up.
"""

import sys
from collections import defaultdict
from typing import Dict, List, Optional

from pymongo.collection import Collection

from . import leadership, staff

REQUESTER = "requester"
SUPPORT = "support"
INTERNAL = "internal"
JIRA_INTEGRATION = "jira_integration"

# The Zendesk user the Jira integration posts its comments as.
JIRA_INTEGRATION_USER_ID = "1031430881"


def author_role(author, stored_role: Optional[str] = None, when=None) -> Optional[str]:
    """The role to show for a comment's author; None when unknown.

    A role stored at ingestion (the requester) wins; then the leadership
    roster (Support), the Jira integration, former Support staff (Support
    when they wrote it), and anyone else at Learnosity (internal).
    `when` is the comment's timestamp: staff who changed department are
    labelled with the one they were in then (staff.Staff.at).
    """
    if stored_role:
        return stored_role
    author = None if author is None else str(author).strip()
    if not author:
        return None
    if leadership.is_support(author):
        return SUPPORT
    if author == JIRA_INTEGRATION_USER_ID:
        return JIRA_INTEGRATION
    person = staff.lookup(author, when)
    if (
        person is not None
        and person.department == staff.SUPPORT
        and person.org == staff.LEARNOSITY
    ):
        return SUPPORT
    if staff.is_staff(author):
        return INTERNAL
    return None


def author_staff(author, when=None) -> Optional[dict]:
    """Who the author was at Learnosity on `when` (staff.Staff.describe),
    or None."""
    person = staff.lookup(author, when)
    return person.describe() if person is not None else None


def internal_only_candidates(tickets: Collection) -> List[dict]:
    """Authors with no known role who have only ever written internal notes.

    Customers can't write internal notes, so these are staff. Anyone who
    wrote even one public comment is left out: they could be a customer.

    Returns:
        [{"author", "comments", "tickets"}], most comments first.
    """
    comments: Dict[str, int] = defaultdict(int)
    ticket_ids: Dict[str, set] = defaultdict(set)
    public = set()
    for doc in tickets.find({}, {"zendesk_id": 1, "transcript": 1}):
        for entry in doc.get("transcript") or []:
            author = entry.get("author")
            if author is None or author_role(author, entry.get("role")):
                continue
            author = str(author)
            if not entry.get("is_internal_note"):
                public.add(author)
                continue
            comments[author] += 1
            ticket_ids[author].add(doc.get("zendesk_id"))
    rows = [
        {"author": a, "comments": n, "tickets": len(ticket_ids[a])}
        for a, n in comments.items()
        if a not in public
    ]
    return sorted(rows, key=lambda row: (-row["comments"], row["author"]))


if __name__ == "__main__":
    if sys.argv[1:] != ["candidates"]:
        sys.exit("usage: python -m decision_engine.authors candidates")
    from core import settings

    from .persistence import get_tickets_collection

    settings.load()
    print("author_id\tinternal_notes\ttickets")
    for row in internal_only_candidates(get_tickets_collection()):
        print(f"{row['author']}\t{row['comments']}\t{row['tickets']}")
