"""Step 1 of 3: lists every ticket carrying a tag being retired. Read-only.

For each old_tag in tag_changes.csv, searches Zendesk (GET /api/v2/search/export,
cursor-paginated, so there is no 1,000-result cap) and keeps the tickets
whose tags really include it. Writes tickets_to_retag.csv:
ticket_id, status, old_tag, new_tag.

Steps 2 and 3 read that file rather than searching again, so the list of
tickets to give new_tag survives step 2 removing old_tag. Re-running this
step after step 2 finds nothing for the tags already removed: keep the file.

    python 1_find_tickets.py
"""
import csv
import os
import sys
import urllib.parse

import zd

HERE = os.path.dirname(os.path.abspath(__file__))
PAGE_SIZE = 100  # Zendesk's recommendation; larger pages can time out on archived tickets


def tickets_with_tag(tag):
    """[(ticket_id, status)] for tickets tagged with tag."""
    query = urllib.parse.urlencode({"query": f"tags:{tag}", "filter[type]": "ticket", "page[size]": PAGE_SIZE})
    url = f"/api/v2/search/export.json?{query}"
    found = []
    while url:
        reply = zd.request("GET", url)
        for ticket in reply.get("results") or []:
            # Search matching isn't documented as exact; the ticket's own tags are.
            if tag in (ticket.get("tags") or []):
                found.append((ticket["id"], ticket.get("status", "")))
        url = (reply.get("links") or {}).get("next") if (reply.get("meta") or {}).get("has_more") else None
    return found


def main(argv):
    if argv:
        sys.exit("Usage: python 1_find_tickets.py")
    changes = zd.read_changes(HERE)
    rows, closed = [], 0
    for old_tag, new_tag in changes:
        found = tickets_with_tag(old_tag)
        closed += sum(status == zd.CLOSED for _, status in found)
        print(f"{old_tag} -> {new_tag}: {len(found)} ticket(s)")
        rows.extend((ticket_id, status, old_tag, new_tag) for ticket_id, status in found)
    path = os.path.join(HERE, zd.TICKETS_FILE)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ticket_id", "status", "old_tag", "new_tag"])
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} ticket/tag pair(s) to {path}.")
    if closed:
        print(f"{closed} are on closed tickets, which Zendesk won't let anyone update; steps 2 and 3 skip them.")
    print("Review it, then run: python 2_remove_old_tags.py")


if __name__ == "__main__":
    main(sys.argv[1:])
