"""The Tag Normalizer's Zendesk retag scripts, run against a fake Zendesk on 127.0.0.1."""

import csv
import json
import os
import subprocess
import sys
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import tag_normalizer

# Ticket id -> (status, tags) in the fake account.
TICKETS = {
    1: ("open", ["oil_change", "vip"]),
    2: ("pending", ["oil_change", "tire_rotation"]),
    3: ("closed", ["oil_change"]),
    4: ("solved", ["tire_rotation"]),
    5: ("open", ["oil_change_extra"]),  # search may return it; its tags don't match
}


class FakeZendesk(BaseHTTPRequestHandler):
    updates = []  # (ids, ticket body)
    searches = []
    jobs = {}

    def log_message(self, *args):
        pass

    def _reply(self, body, status=200):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        url = urllib.parse.urlsplit(self.path)
        params = urllib.parse.parse_qs(url.query)
        if self.headers.get("Authorization") is None:
            return self._reply({"error": "no auth"}, 401)
        if url.path == "/api/v2/search/export.json":
            assert params["filter[type]"] == ["ticket"]
            tag = params["query"][0].split(":", 1)[1]
            self.searches.append(tag)
            hits = [
                i
                for i, (_, tags) in TICKETS.items()
                if any(t.startswith(tag) for t in tags)
            ]
            page = int(params.get("page[after]", ["0"])[0])
            more = page + 2 < len(hits)
            base = f"http://{self.headers['Host']}"
            nxt = f"{base}{url.path}?{urllib.parse.urlencode({**{k: v[0] for k, v in params.items()}, 'page[after]': page + 2})}"
            return self._reply(
                {
                    "results": [
                        {"id": i, "status": TICKETS[i][0], "tags": TICKETS[i][1]}
                        for i in hits[page : page + 2]
                    ],
                    "meta": {"has_more": more},
                    "links": {"next": nxt if more else None},
                }
            )
        if url.path.startswith("/api/v2/job_statuses/"):
            job_id = url.path.rsplit("/", 1)[1].split(".")[0]
            return self._reply({"job_status": self.jobs[job_id]})
        self._reply({"error": "not found"}, 404)

    def do_PUT(self):
        url = urllib.parse.urlsplit(self.path)
        assert url.path == "/api/v2/tickets/update_many.json"
        ids = urllib.parse.parse_qs(url.query)["ids"][0].split(",")
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.updates.append((ids, body["ticket"]))
        job_id = f"job{len(self.updates)}"
        self.jobs[job_id] = {
            "id": job_id,
            "status": "completed",
            "results": [
                {"id": int(i), "action": "update", "success": True, "status": "Updated"}
                for i in ids
            ],
        }
        # The first job is still queued when the PUT returns, so it's polled.
        self._reply(
            {
                "job_status": {
                    "id": job_id,
                    "status": "queued" if job_id == "job1" else "completed",
                    **({} if job_id == "job1" else self.jobs[job_id]),
                }
            }
        )


@pytest.fixture
def zendesk():
    FakeZendesk.updates, FakeZendesk.searches, FakeZendesk.jobs = [], [], {}
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeZendesk)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


@pytest.fixture
def kit(tmp_path):
    folder = tmp_path / "kit"
    tag_normalizer.write_retag_scripts(
        {
            "oil_change": "Car Maintenance",
            "tire_rotation": "Car Maintenance",
            "car_maintenance": "Car Maintenance",
        },
        str(folder),
    )
    return folder


def run(kit, zendesk, script, *args):
    env = {
        **os.environ,
        "ZENDESK_BASE_URL": zendesk,
        "ZENDESK_EMAIL": "a@example.com",
        "ZENDESK_API_TOKEN": "t",
    }
    return subprocess.run(
        [sys.executable, script, *args],
        cwd=kit,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# --- the generator ------------------------------------------------------------------


def test_retag_changes_converts_and_skips():
    changes, warnings = tag_normalizer.retag_changes(
        {
            "oil_change": "Car Maintenance",
            "car_maintenance": "Car Maintenance",
            "Car Maintenance": "Car Maintenance",
            "refund": "refund",
        }
    )
    assert changes == [("oil_change", "car_maintenance")]
    assert any("'car_maintenance'" in w for w in warnings)
    assert any("Skipped 'Car Maintenance'" in w for w in warnings)


def test_write_retag_scripts(kit):
    assert sorted(os.listdir(kit)) == sorted(
        tag_normalizer.SCRIPT_FILES + ("tag_changes.csv",)
    )
    assert rows(kit / "tag_changes.csv") == [
        {"old_tag": "oil_change", "new_tag": "car_maintenance"},
        {"old_tag": "tire_rotation", "new_tag": "car_maintenance"},
    ]
    with pytest.raises(FileExistsError):
        tag_normalizer.write_retag_scripts({"a": "b"}, str(kit))


def test_cli_writes_scripts(tmp_path, monkeypatch):
    import numpy as np

    monkeypatch.setattr(
        tag_normalizer,
        "embedder",
        lambda backend: lambda texts: np.ones((len(texts), 2)),
    )
    source = tmp_path / "tags.txt"
    source.write_text("oil_change, oil_change, tire_rotation")
    assert (
        tag_normalizer.main(
            [str(source), "--backend", "spacy", "--scripts", str(tmp_path / "kit")]
        )
        == 0
    )
    assert rows(tmp_path / "kit" / "tag_changes.csv") == [
        {"old_tag": "tire_rotation", "new_tag": "oil_change"}
    ]


# --- the scripts --------------------------------------------------------------------


def test_full_run(kit, zendesk):
    found = run(kit, zendesk, "1_find_tickets.py")
    assert found.returncode == 0, found.stderr
    assert FakeZendesk.searches == [
        "oil_change",
        "oil_change",
        "tire_rotation",
    ]  # two pages for oil_change
    assert [
        (r["ticket_id"], r["status"], r["old_tag"])
        for r in rows(kit / "tickets_to_retag.csv")
    ] == [
        ("1", "open", "oil_change"),
        ("2", "pending", "oil_change"),
        ("3", "closed", "oil_change"),
        ("2", "pending", "tire_rotation"),
        ("4", "solved", "tire_rotation"),
    ]
    assert "closed" in found.stdout

    dry = run(kit, zendesk, "2_remove_old_tags.py")
    assert dry.returncode == 0 and "Dry run" in dry.stdout
    assert FakeZendesk.updates == []

    removed = run(kit, zendesk, "2_remove_old_tags.py", "--apply")
    assert removed.returncode == 0, removed.stderr
    assert FakeZendesk.updates == [
        (["1", "2"], {"remove_tags": ["oil_change"]}),
        (["2", "4"], {"remove_tags": ["tire_rotation"]}),
    ]
    assert all(r["ok"] == "True" for r in rows(kit / "2_remove_results.csv"))

    added = run(kit, zendesk, "3_add_new_tags.py", "--apply")
    assert added.returncode == 0, added.stderr
    # Ticket 2 had both old tags but gets the new one once; closed ticket 3 is skipped.
    assert FakeZendesk.updates[2:] == [
        (["1", "2", "4"], {"additional_tags": ["car_maintenance"]})
    ]
    assert len(rows(kit / "3_add_results.csv")) == 3


def test_scripts_refuse_plain_http_to_remote_host(kit):
    result = run(kit, "http://example.com", "1_find_tickets.py")
    assert result.returncode != 0 and "must be https" in result.stderr


def test_step_2_needs_step_1(kit, zendesk):
    result = run(kit, zendesk, "2_remove_old_tags.py", "--apply")
    assert result.returncode != 0 and "run 1_find_tickets.py first" in result.stderr
