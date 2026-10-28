"""Who wrote each comment, as far as the stored IDs tell.

Ingestion only knows the requester (wrangler.ticket_to_document); every
other author's role is stored as None. The payload fills that in from IDs
we know (author_role), so the model can tell Support's comments, the Jira
integration's and the customer's apart, which triggers like "Support asked
for info" and "Jira integration note confirms the release" rely on.

INTERNAL_AUTHORS holds Learnosity staff outside Support (product,
engineering, CSMs). It starts empty: list candidates with

    python -m decision_engine.authors candidates

which prints only IDs and counts, never names or comment text, for authors
who have only ever written internal notes, so no customer is looked up.
"""

import sys
from collections import defaultdict
from typing import Dict, List, Optional

from pymongo.collection import Collection

from . import leadership

REQUESTER = "requester"
SUPPORT = "support"
INTERNAL = "internal"
JIRA_INTEGRATION = "jira_integration"

# The Zendesk user the Jira integration posts its comments as.
JIRA_INTEGRATION_USER_ID = "1031430881"

# Learnosity staff outside Support, by Zendesk user ID -> what they do
# (e.g. "Product"). Filled in from `candidates` once someone confirms each.
INTERNAL_AUTHORS: Dict[str, str] = {}


def author_role(author, stored_role: Optional[str] = None) -> Optional[str]:
    """The role to show for a comment's author; None when unknown.

    A role stored at ingestion (the requester) wins; then the leadership
    roster (Support), the Jira integration, and INTERNAL_AUTHORS.
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
    if author in INTERNAL_AUTHORS:
        return INTERNAL
    return None


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
