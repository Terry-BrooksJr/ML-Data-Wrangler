"""Step 3 of 3: adds each successor tag to the tickets step 1 found.

Reads tickets_to_retag.csv (not a new search: step 2 has removed the old
tags) and, for each new_tag, sends
PUT /api/v2/tickets/update_many.json?ids=... with {"ticket": {"additional_tags": [new_tag]}}
in batches of 100. additional_tags adds to the ticket's tags rather than
replacing them. A ticket that had several old tags mapping to one new tag
is updated once. Closed tickets are skipped.

Without --apply it only prints what it would do. Results go to
3_add_results.csv.

    python 3_add_new_tags.py            # dry run
    python 3_add_new_tags.py --apply
"""

import os
import sys
from collections import defaultdict

import zd

HERE = os.path.dirname(os.path.abspath(__file__))


def main(argv):
    apply = zd.parse_apply(argv, "3_add_new_tags.py")
    by_tag = defaultdict(list)
    skipped = 0
    for row in zd.read_tickets(HERE):
        if row["status"] == zd.CLOSED:
            skipped += 1
        elif row["ticket_id"] not in by_tag[row["new_tag"]]:
            by_tag[row["new_tag"]].append(row["ticket_id"])
    for tag, ids in by_tag.items():
        print(f"add {tag!r} to {len(ids)} ticket(s)")
    if skipped:
        print(f"skipping {skipped} closed ticket/tag pair(s)")
    if not apply:
        print("\nDry run: nothing changed. Re-run with --apply to add these tags.")
        return
    log, failed = [], 0
    for tag, ids in by_tag.items():
        for batch in zd.chunks(ids):
            for ticket_id, ok, detail in zd.update_many(
                batch, {"additional_tags": [tag]}
            ):
                failed += not ok
                log.append((ticket_id, tag, ok, detail))
            print(f"  {tag}: {sum(1 for r in log if r[1] == tag)}/{len(ids)} done")
    path = zd.write_log(HERE, "3_add_results.csv", log)
    print(f"\n{len(log) - failed} added, {failed} failed. Details: {path}")


if __name__ == "__main__":
    main(sys.argv[1:])
