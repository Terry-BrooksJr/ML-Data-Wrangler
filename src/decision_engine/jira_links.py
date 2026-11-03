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

An import merges: a ticket keeps the links it already has, and gains the
export's. So pages can be imported one at a time or in any order, and an
older export never removes a newer link. The flip side: a link removed in
Zendesk stays until the field is cleared.

The export comes from outside the app and its values reach the model, so
every record is checked against the export's formats (numeric IDs, a Jira
key, an ISO 8601 time) before anything is written, and the payload shows
only values in those formats.

    python -m decision_engine.jira_links import PAGE.json [PAGE.json ...]

The app's Write Tickets to MongoDB also imports export pages, when chosen,
after writing the tickets.
"""

import json
import re
import sys
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Optional

from pymongo.collection import Collection

JIRA_LINKS_KEY = "jira_links"
_NUMERIC_ID = re.compile(r"\d{1,20}")
_ISSUE_KEY = re.compile(r"[A-Z][A-Z0-9_]{0,49}-\d{1,9}")
_TIMESTAMP = re.compile(
    r"\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}(:\d{2}(\.\d{1,9})?)?(Z|[+-]\d{2}:?\d{2})?)?"
)
# How many malformed records an error names.
_PROBLEMS_SHOWN = 5


def _is_id(value) -> bool:
    if isinstance(value, bool):
        return False
    return isinstance(value, int) and value >= 0 or (
        isinstance(value, str) and bool(_NUMERIC_ID.fullmatch(value))
    )


def is_issue_key(value) -> bool:
    return isinstance(value, str) and bool(_ISSUE_KEY.fullmatch(value))


def is_timestamp(value) -> bool:
    return isinstance(value, str) and bool(_TIMESTAMP.fullmatch(value))


def link_problem(link) -> Optional[str]:
    """Why one export record can't be imported, or None when it can."""
    if not isinstance(link, dict):
        return "not an object"
    for field in ("ticket_id", "issue_id"):
        if not _is_id(link.get(field)):
            return f"{field} {link.get(field)!r} is not a numeric ID"
    if link.get("issue_key") not in (None, "") and not is_issue_key(link["issue_key"]):
        return f"issue_key {link['issue_key']!r} is not a Jira key"
    if link.get("created_at") is not None and not is_timestamp(link["created_at"]):
        return f"created_at {link['created_at']!r} is not an ISO 8601 time"
    return None


def read_export(path: str) -> List[dict]:
    """The links in one saved export page.

    Raises:
        ValueError: the file isn't an export, or any record is malformed (so
            nothing is written from a file that is only partly usable).
    """
    with open(path) as f:
        data = json.load(f)
    links = data.get("links") if isinstance(data, dict) else data
    if not isinstance(links, list):
        raise ValueError(f"{path}: expected a list of links or {{'links': [...]}}")
    problems = [
        f"record {i}: {problem}"
        for i, problem in ((i, link_problem(link)) for i, link in enumerate(links))
        if problem
    ]
    if problems:
        shown = "; ".join(problems[:_PROBLEMS_SHOWN])
        more = f" (and {len(problems) - _PROBLEMS_SHOWN} more)" if len(problems) > _PROBLEMS_SHOWN else ""
        raise ValueError(
            f"{path}: {len(problems)} of {len(links)} links are malformed: {shown}{more}"
        )
    return links


def _entry(link: dict) -> dict:
    return {
        "issue_id": str(link["issue_id"]),
        "issue_key": link.get("issue_key") or None,
        "linked_at": link.get("created_at"),
    }


def _merge(entries: Dict[str, dict], entry: dict) -> None:
    """Adds entry, one per issue; a copy that carries the key wins."""
    known = entries.get(entry["issue_id"])
    if known is None or (entry["issue_key"] and not known.get("issue_key")):
        entries[entry["issue_id"]] = entry


def _sorted(entries: Dict[str, dict]) -> List[dict]:
    return sorted(entries.values(), key=lambda e: e.get("linked_at") or "")


def links_by_ticket(links: Iterable[dict]) -> Dict[str, List[dict]]:
    """Zendesk ticket ID -> its links, oldest first, one per Jira issue."""
    by_ticket: Dict[str, Dict[str, dict]] = defaultdict(dict)
    for link in links:
        if link.get("ticket_id") is None or link.get("issue_id") is None:
            continue
        # Pages can overlap; a later copy of a link may carry the key.
        _merge(by_ticket[str(link["ticket_id"])], _entry(link))
    return {ticket_id: _sorted(issues) for ticket_id, issues in by_ticket.items()}


def store_links(
    tickets: Collection,
    by_ticket: Dict[str, List[dict]],
    progress: Optional[Callable[[int], None]] = None,
) -> int:
    """Merges each ticket's links into the ones it has stored; tickets that
    aren't stored are skipped. Rerunning an import changes nothing more, so
    one that stopped partway is finished by running it again.

    Args:
        progress: Called with how many tickets are done after each one.

    Returns:
        How many tickets were updated.
    """
    updated = 0
    for done, (ticket_id, links) in enumerate(by_ticket.items(), start=1):
        stored = tickets.find_one({"zendesk_id": ticket_id}, {JIRA_LINKS_KEY: 1})
        if stored is not None:
            entries = {
                e["issue_id"]: e
                for e in stored.get(JIRA_LINKS_KEY) or []
                if isinstance(e, dict) and "issue_id" in e
            }
            for entry in links:
                _merge(entries, entry)
            updated += tickets.update_one(
                {"zendesk_id": ticket_id}, {"$set": {JIRA_LINKS_KEY: _sorted(entries)}}
            ).modified_count
        if progress:
            progress(done)
    return updated


def describe_import(link_count: int, ticket_count: int, updated: int) -> str:
    return (
        f"Read {link_count} Jira links for {ticket_count} tickets; "
        f"updated {updated} stored tickets."
    )


def import_links(tickets: Collection, links: List[dict]) -> str:
    """Stores the links on the tickets that are stored; returns a summary."""
    by_ticket = links_by_ticket(links)
    return describe_import(len(links), len(by_ticket), store_links(tickets, by_ticket))


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] != "import":
        sys.exit("usage: python -m decision_engine.jira_links import PAGE.json [...]")
    from core import settings

    from .persistence import get_tickets_collection

    settings.load()
    links = [link for path in sys.argv[2:] for link in read_export(path)]
    print(import_links(get_tickets_collection(), links))
