"""Links between Zendesk tickets and Jira issues, from the Zendesk Jira links export.

The export (GET /api/services/jira/links, saved as JSON) is a list of
{"id", "ticket_id", "issue_id", "issue_key", "url", "created_at",
"updated_at"}, or a page {"links": [...]} of one. Each ticket's links are
stored on its document as `jira_links`; ingestion never writes that field,
so re-ingesting a ticket keeps them.

What a link does NOT tell: the issue's status, or whether it moved. Its
created_at/updated_at are when the link was made (updated_at has never
differed from created_at in the export), so the payload passes them as
`linked_at` only. Many links have no issue_key; the issue_id is kept for those.

    python -m decision_engine.jira_links import PAGE.json [PAGE.json ...]
"""

import json
import sys
from collections import defaultdict
from typing import Dict, Iterable, List

from pymongo.collection import Collection

JIRA_LINKS_KEY = "jira_links"


def read_export(path: str) -> List[dict]:
    """The links in one saved export page."""
    with open(path) as f:
        data = json.load(f)
    links = data.get("links") if isinstance(data, dict) else data
    if not isinstance(links, list):
        raise ValueError(f"{path}: expected a list of links or {{'links': [...]}}")
    return [link for link in links if isinstance(link, dict)]


def links_by_ticket(links: Iterable[dict]) -> Dict[str, List[dict]]:
    """Zendesk ticket ID -> its links, oldest first, one per Jira issue."""
    by_ticket: Dict[str, Dict[str, dict]] = defaultdict(dict)
    for link in links:
        ticket_id, issue_id = link.get("ticket_id"), link.get("issue_id")
        if ticket_id is None or issue_id is None:
            continue
        entry = {
            "issue_id": str(issue_id),
            "issue_key": link.get("issue_key") or None,
            "linked_at": link.get("created_at"),
        }
        # Pages can overlap; a later copy of a link may carry the key.
        known = by_ticket[str(ticket_id)].get(entry["issue_id"])
        if known is None or (entry["issue_key"] and not known["issue_key"]):
            by_ticket[str(ticket_id)][entry["issue_id"]] = entry
    return {
        ticket_id: sorted(issues.values(), key=lambda e: e["linked_at"] or "")
        for ticket_id, issues in by_ticket.items()
    }


def store_links(tickets: Collection, by_ticket: Dict[str, List[dict]]) -> int:
    """Sets jira_links on each ticket that is stored; others are skipped.

    Returns:
        How many tickets were updated.
    """
    return sum(
        tickets.update_one(
            {"zendesk_id": ticket_id}, {"$set": {JIRA_LINKS_KEY: links}}
        ).modified_count
        for ticket_id, links in by_ticket.items()
    )


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] != "import":
        sys.exit("usage: python -m decision_engine.jira_links import PAGE.json [...]")
    from core import settings

    from .persistence import get_tickets_collection

    settings.load()
    links = [link for path in sys.argv[2:] for link in read_export(path)]
    by_ticket = links_by_ticket(links)
    updated = store_links(get_tickets_collection(), by_ticket)
    print(
        f"Read {len(links)} links for {len(by_ticket)} tickets; "
        f"updated {updated} stored tickets."
    )
