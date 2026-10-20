"""Shared Zendesk helpers for the three retag scripts. Python 3.8+, standard library only.

Credentials come from the environment, never from a file in this folder:

    export ZENDESK_SUBDOMAIN=yourcompany      # https://yourcompany.zendesk.com
    export ZENDESK_EMAIL=agent@example.com     # an agent or admin
    export ZENDESK_API_TOKEN=...               # Admin Center > Apps and integrations > Zendesk API

ZENDESK_BASE_URL overrides the subdomain (for a sandbox); it must be https://.
"""

import base64
import csv
import json
import os
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

CHANGES_FILE = "tag_changes.csv"
TICKETS_FILE = "tickets_to_retag.csv"
# update_many takes at most 100 ticket ids per request.
MAX_IDS = 100
MAX_ATTEMPTS = 6
JOB_POLL_SECONDS = 2
JOB_TIMEOUT_SECONDS = 600
# Zendesk refuses updates to closed tickets, so steps 2 and 3 skip them.
CLOSED = "closed"


def base_url():
    url = os.environ.get("ZENDESK_BASE_URL", "").rstrip("/")
    if not url:
        subdomain = os.environ.get("ZENDESK_SUBDOMAIN", "").strip()
        if not subdomain:
            sys.exit("Set ZENDESK_SUBDOMAIN (or ZENDESK_BASE_URL) first.")
        url = f"https://{subdomain}.zendesk.com"
    host = urllib.parse.urlsplit(url).hostname or ""
    if not url.startswith("https://") and host not in ("localhost", "127.0.0.1"):
        sys.exit(f"{url} must be https://: the API token is sent to it.")
    return url


def _auth_header():
    email = os.environ.get("ZENDESK_EMAIL", "").strip()
    token = os.environ.get("ZENDESK_API_TOKEN", "").strip()
    if not email or not token:
        sys.exit("Set ZENDESK_EMAIL and ZENDESK_API_TOKEN first.")
    return "Basic " + base64.b64encode(f"{email}/token:{token}".encode()).decode()


def request(method, path_or_url, body=None):
    """One API call; returns the parsed JSON reply.

    429 waits for Retry-After; 5xx and connection errors back off and retry.
    Repeating an update_many is safe: adding or removing a tag twice changes nothing.
    """
    url = path_or_url if path_or_url.startswith("http") else base_url() + path_or_url
    data = json.dumps(body).encode() if body is not None else None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        req = urllib.request.Request(
            url,
            data=data,
            method=method,
            headers={
                "Authorization": _auth_header(),
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as reply:
                return json.loads(reply.read() or b"{}")
        except urllib.error.HTTPError as e:
            if attempt < MAX_ATTEMPTS and e.code == 429:
                delay = float(e.headers.get("Retry-After") or 60)
                print(f"  Rate limited; waiting {delay:.0f}s…")
                time.sleep(delay + random.uniform(0, 1))
                continue
            if attempt < MAX_ATTEMPTS and e.code >= 500:
                time.sleep(min(2**attempt, 60))
                continue
            detail = e.read()[:500].decode("utf-8", "replace")
            sys.exit(f"{method} {url} failed: HTTP {e.code} {detail}")
        except urllib.error.URLError as e:
            if attempt < MAX_ATTEMPTS:
                time.sleep(min(2**attempt, 60))
                continue
            sys.exit(f"{method} {url} failed: {e.reason}")
    raise AssertionError("unreachable")


def read_changes(folder):
    """[(old_tag, new_tag)] from tag_changes.csv."""
    with open(os.path.join(folder, CHANGES_FILE), newline="", encoding="utf-8") as f:
        return [(row["old_tag"], row["new_tag"]) for row in csv.DictReader(f)]


def read_tickets(folder):
    """Rows of tickets_to_retag.csv (step 1's output)."""
    path = os.path.join(folder, TICKETS_FILE)
    if not os.path.exists(path):
        sys.exit(f"{path} not found: run 1_find_tickets.py first.")
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def chunks(items, size=MAX_IDS):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def wait_for_job(job):
    """Polls a job_status until it ends; returns its final state."""
    deadline = time.monotonic() + JOB_TIMEOUT_SECONDS
    while job.get("status") in ("queued", "working"):
        if time.monotonic() > deadline:
            return job
        time.sleep(JOB_POLL_SECONDS)
        job = request("GET", f"/api/v2/job_statuses/{job['id']}.json")["job_status"]
    return job


def update_many(ids, ticket_change):
    """Applies one change to up to 100 tickets; returns [(ticket_id, ok, detail)]."""
    query = urllib.parse.urlencode({"ids": ",".join(str(i) for i in ids)})
    job = wait_for_job(
        request(
            "PUT",
            f"/api/v2/tickets/update_many.json?{query}",
            {"ticket": ticket_change},
        )["job_status"]
    )
    if job.get("status") != "completed":
        return [
            (
                i,
                False,
                f"job {job.get('id')} ended {job.get('status')}: {job.get('message') or ''}",
            )
            for i in ids
        ]
    results = {str(r.get("id")): r for r in job.get("results") or []}
    outcome = []
    for i in ids:
        result = results.get(str(i))
        if result is None:
            outcome.append((i, False, "not in job results"))
        else:
            ok = result.get("success", result.get("status") != "Failed")
            outcome.append(
                (
                    i,
                    bool(ok),
                    result.get("error")
                    or result.get("details")
                    or result.get("status")
                    or "",
                )
            )
    return outcome


def write_log(folder, name, rows):
    path = os.path.join(folder, name)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ticket_id", "tag", "ok", "detail"])
        writer.writerows(rows)
    return path


def parse_apply(argv, step):
    if argv not in ([], ["--apply"]):
        sys.exit(
            f"Usage: python {step} [--apply]   (without --apply, only shows what would change)"
        )
    return argv == ["--apply"]
