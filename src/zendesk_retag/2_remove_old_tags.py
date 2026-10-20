"""Step 2 of 3: removes each retired tag from the tickets step 1 found.

Reads tickets_to_retag.csv and, for each old_tag, sends
PUT /api/v2/tickets/update_many.json?ids=... with {"ticket": {"remove_tags": [old_tag]}}
in batches of 100. remove_tags removes only that tag; the ticket's other
tags are untouched. Closed tickets are skipped (Zendesk refuses to update them).

Each update is an ordinary ticket update: it changes updated_at and can fire
triggers that look at tags.

Without --apply it only prints what it would do. Results go to
2_remove_results.csv.

    python 2_remove_old_tags.py            # dry run
    python 2_remove_old_tags.py --apply
"""
import os
import sys
from collections import defaultdict

import zd

HERE = os.path.dirname(os.path.abspath(__file__))


def main(argv):
    apply = zd.parse_apply(argv, "2_remove_old_tags.py")
    by_tag = defaultdict(list)
    skipped = 0
    for row in zd.read_tickets(HERE):
        if row["status"] == zd.CLOSED:
            skipped += 1
        elif row["ticket_id"] not in by_tag[row["old_tag"]]:
            by_tag[row["old_tag"]].append(row["ticket_id"])
    for tag, ids in by_tag.items():
        print(f"remove {tag!r} from {len(ids)} ticket(s)")
    if skipped:
        print(f"skipping {skipped} closed ticket(s)")
    if not apply:
        print("\nDry run: nothing changed. Re-run with --apply to remove these tags.")
        return
    log, failed = [], 0
    for tag, ids in by_tag.items():
        for batch in zd.chunks(ids):
            for ticket_id, ok, detail in zd.update_many(batch, {"remove_tags": [tag]}):
                failed += not ok
                log.append((ticket_id, tag, ok, detail))
            print(f"  {tag}: {sum(1 for r in log if r[1] == tag)}/{len(ids)} done")
    path = zd.write_log(HERE, "2_remove_results.csv", log)
    print(f"\n{len(log) - failed} removed, {failed} failed. Details: {path}")
    print("Next: python 3_add_new_tags.py")


if __name__ == "__main__":
    main(sys.argv[1:])
