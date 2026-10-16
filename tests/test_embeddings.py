import io
import json
import logging
import socket
import threading
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from datetime import datetime, timedelta, timezone

import mongomock
import numpy as np
import pytest

import embeddings
from decision_engine import search


@pytest.fixture
def env(monkeypatch):
    for name in ("EMBEDDING_API_URL", "EMBEDDING_API_ENDPOINT", "EMBEDDING_MODEL", "EMBEDDING_DIMENSIONS",
                 "EMBEDDING_TPM", "EMBEDDING_RPM"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("EMBEDDING_API_KEY", "test-key")
    monkeypatch.setattr(embeddings, "_limiters", {})
    monkeypatch.setattr(embeddings.random, "uniform", lambda a, b: 0.0)
    return monkeypatch


class FakeClock:
    """embeddings' clock and sleep: sleeping moves the clock on, instantly."""

    def __init__(self):
        self.t, self.sleeps = 1000.0, []

    def now(self):
        return self.t

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.t += seconds


@pytest.fixture
def clock(env):
    fake = FakeClock()
    env.setattr(embeddings, "_now", fake.now)
    env.setattr(embeddings.time, "sleep", fake.sleep)
    return fake


def _ticket(zendesk_id, subject, comments, **extra):
    return {"zendesk_id": zendesk_id, "subject": subject, "transcript": comments, "status": "open", **extra}


def _comment(body, role=None, internal=False):
    return {"author": "a", "role": role, "body": body, "timestamp": "2026-10-01", "is_internal_note": internal}


# --- config -----------------------------------------------------------------------


def test_config_defaults_to_voyage_on_mongodb(env):
    cfg = embeddings.config()
    assert cfg == embeddings.Config("https://ai.mongodb.com/v1/embeddings", "voyage-3.5", 1024, "test-key")
    assert cfg.base_url == "https://ai.mongodb.com/v1"


def test_default_endpoint_is_read_when_config_is_built(env):
    # Not at import: the CLI imports embeddings before loading .env.
    env.setenv("EMBEDDING_API_ENDPOINT", "api.voyageai.com")
    assert embeddings.config().url == "https://api.voyageai.com/v1/embeddings"


def test_config_requires_an_embeddings_url(env):
    env.setenv("EMBEDDING_API_URL", "https://ai.mongodb.com/v1")
    with pytest.raises(embeddings.EmbeddingNotConfigured, match="must end in /embeddings"):
        embeddings.config()
    env.setenv("EMBEDDING_API_URL", "https://api.voyageai.com/v1/embeddings/")
    assert embeddings.config().base_url == "https://api.voyageai.com/v1"


@pytest.mark.parametrize("name, value, message", [
    ("EMBEDDING_API_KEY", "", "EMBEDDING_API_KEY is not set"),
    ("EMBEDDING_DIMENSIONS", "1000", "must be one of 256, 512, 1024, 2048"),
    ("EMBEDDING_DIMENSIONS", "big", "must be one of"),
    ("EMBEDDING_API_URL", "http://api.voyageai.com/v1/embeddings", "not encrypted"),
])
def test_config_rejects_bad_settings(env, name, value, message):
    env.setenv(name, value)
    with pytest.raises(embeddings.EmbeddingNotConfigured, match=message):
        embeddings.config()


def test_config_allows_local_http(env):
    env.setenv("EMBEDDING_API_URL", "http://localhost:8080/v1/embeddings")
    env.setenv("EMBEDDING_DIMENSIONS", "256")
    assert embeddings.config().dimensions == 256


# --- text -------------------------------------------------------------------------


def test_opening_text_uses_subject_and_first_requester_comment():
    ticket = _ticket("1", "Login fails", [
        _comment("internal triage", internal=True),
        _comment("agent hello"),
        _comment("I can't log in", role="requester"),
        _comment("still broken", role="requester"),
    ])
    assert embeddings.opening_text(ticket) == "Login fails\n\nI can't log in"


def test_opening_text_never_uses_an_unknown_authors_comment():
    ticket = _ticket("1", "Login fails", [_comment("Hi, agent here, looking into it")])
    assert embeddings.opening_text(ticket) == "Login fails"


def test_full_text_leaves_out_internal_notes_and_is_capped(monkeypatch):
    ticket = _ticket("1", "S", [_comment("secret", internal=True), _comment("a"), _comment("b")])
    assert embeddings.full_text(ticket) == "S\n\na\n\nb"
    monkeypatch.setattr(embeddings, "MAX_CHARS", 3)
    assert embeddings.full_text(ticket) == "S\n\n"


# --- API --------------------------------------------------------------------------


def test_batches_respect_count_and_size(monkeypatch):
    monkeypatch.setattr(embeddings, "MAX_BATCH_TEXTS", 3)
    monkeypatch.setattr(embeddings, "MAX_BATCH_CHARS", 10)
    assert embeddings._batches(["a"] * 7) == [(0, 3), (3, 6), (6, 7)]
    assert embeddings._batches(["aaaaaa", "bbbbbb", "c"]) == [(0, 1), (1, 3)]
    assert embeddings._batches(["x" * 50, "y"]) == [(0, 1), (1, 2)]
    assert embeddings._batches([]) == []


# --- Voyage client, against a local server ----------------------------------------


class FakeVoyage:
    """A local Voyage API. Each request is answered by respond(body) -> (status, payload[, headers]),
    or a default reply of one vector per input."""

    def __init__(self):
        self.requests = []
        self.respond = lambda body: (200, self.reply(body["input"]))
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                server.requests.append((self.path, dict(self.headers), body))
                status, payload, *headers = server.respond(body)
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                for name, value in (headers[0] if headers else {}).items():
                    self.send_header(name, value)
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}/v1/embeddings"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    @staticmethod
    def reply(texts, dims=1024):
        data = [{"object": "embedding", "index": i, "embedding": [float(i)] * dims} for i in range(len(texts))]
        return {"object": "list", "data": data, "model": "voyage-3.5", "usage": {"total_tokens": len(texts)}}

    def paths(self):
        return [path for path, _, _ in self.requests]


@pytest.fixture
def voyage(env, clock):
    server = FakeVoyage()
    server.clock = clock
    env.setenv("EMBEDDING_API_URL", server.url)
    yield server
    server.httpd.shutdown()
    server.httpd.server_close()


def test_embed_sends_voyage_request_through_the_client(voyage):
    vectors, tokens = embeddings.embed(["a", "b"], embeddings.config(), input_type="query")

    path, headers, body = voyage.requests[0]
    assert path == "/v1/embeddings" and headers["Authorization"] == "Bearer test-key"
    assert {k: body[k] for k in ("model", "input", "input_type", "output_dimension", "truncation")} == {
        "model": "voyage-3.5", "input": ["a", "b"], "input_type": "query", "output_dimension": 1024,
        "truncation": True}
    assert vectors.shape == (2, 1024) and vectors[1, 0] == 1.0 and tokens == 2


def test_embed_splits_a_batch_over_the_token_limit(voyage):
    sizes = []

    def respond(body):
        sizes.append(len(body["input"]))
        if len(body["input"]) > 2:
            return 400, {"detail": "Request exceeds the max allowed tokens per batch"}
        return 200, voyage.reply(body["input"])

    voyage.respond = respond
    vectors, tokens = embeddings.embed(["a", "b", "c", "d", "e"], embeddings.config())
    assert sizes == [5, 2, 3, 1, 2]
    assert vectors.shape == (5, 1024) and tokens == 5


def test_other_client_errors_are_not_split(voyage):
    voyage.respond = lambda body: (400, {"detail": "Unknown model"})
    with pytest.raises(embeddings.ApiError, match="HTTP 400") as raised:
        embeddings.embed(["a", "b"], embeddings.config())
    assert not raised.value.token_limit
    assert len(voyage.requests) == 1


def test_embed_rejects_wrong_dimensions(voyage):
    voyage.respond = lambda body: (200, voyage.reply(body["input"], dims=3))
    with pytest.raises(embeddings.ApiError, match="3-dimension"):
        embeddings.embed(["a"], embeddings.config())


@pytest.mark.parametrize("status", [429, 502, 503, 504])
def test_client_retries_only_when_nothing_was_processed(voyage, status):
    outcomes = [(status, {"detail": "busy"}), (200, None)]
    voyage.respond = lambda body: (lambda o: (o[0], o[1] or voyage.reply(body["input"])))(
        outcomes[len(voyage.requests) - 1])
    vectors, _ = embeddings.embed(["a"], embeddings.config())
    assert vectors.shape == (1, 1024) and len(voyage.requests) == 2


@pytest.mark.parametrize("status", [401, 500])
def test_client_does_not_retry_failures_that_may_have_been_billed(voyage, status):
    voyage.respond = lambda body: (status, {"detail": "no"})
    with pytest.raises(embeddings.ApiError, match=f"HTTP {status}"):
        embeddings.embed(["a"], embeddings.config())
    assert len(voyage.requests) == 1


def test_client_does_not_retry_a_timeout(voyage, monkeypatch):
    monkeypatch.setattr(embeddings, "REQUEST_TIMEOUT_SECONDS", 0.2)

    def slow(body):
        threading.Event().wait(0.6)  # time.sleep is patched out by the fixture
        return 200, voyage.reply(body["input"])

    voyage.respond = slow
    with pytest.raises(embeddings.ApiError, match="timed out"):
        embeddings.embed(["a"], embeddings.config())
    assert len(voyage.requests) == 1


def test_client_retries_a_refused_connection(env):
    env.setattr(embeddings, "_now", lambda: 0.0)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]  # closed once this block ends: connections are refused
    env.setenv("EMBEDDING_API_URL", f"http://127.0.0.1:{port}/v1/embeddings")
    sleeps = []
    env.setattr(embeddings.time, "sleep", sleeps.append)
    with pytest.raises(embeddings.ApiError, match="could not connect"):
        embeddings.embed(["a"], embeddings.config())
    assert len(sleeps) == embeddings.MAX_ATTEMPTS - 1


def test_client_refuses_redirects(voyage):
    collect = voyage.url.replace("/v1/embeddings", "/collect")
    voyage.respond = lambda body: (307, {}, {"Location": collect})
    with pytest.raises(embeddings.ApiError, match="not following") as raised:
        embeddings.embed(["secret ticket text"], embeddings.config())
    assert raised.value.status == 307 and collect in str(raised.value)
    assert voyage.paths() == ["/v1/embeddings"]


# --- rate limits ------------------------------------------------------------------


def test_config_reads_rate_limits(env):
    cfg = embeddings.config()
    assert (cfg.tpm, cfg.rpm, cfg.token_budget) == (10_000, 3, 9_000)
    env.setenv("EMBEDDING_TPM", "2,000,000")
    env.setenv("EMBEDDING_RPM", "2000")
    assert (embeddings.config().tpm, embeddings.config().rpm) == (2_000_000, 2000)


@pytest.mark.parametrize("name, value", [("EMBEDDING_TPM", "0"), ("EMBEDDING_RPM", "lots")])
def test_config_rejects_bad_rate_limits(env, name, value):
    env.setenv(name, value)
    with pytest.raises(embeddings.EmbeddingNotConfigured, match=name):
        embeddings.config()


def test_batches_fit_the_token_budget(env):
    cfg = embeddings.config()
    assert embeddings.batch_chars(cfg) == int(9_000 * embeddings.CHARS_PER_TOKEN_GUESS)
    assert embeddings.batch_chars() == embeddings.MAX_BATCH_CHARS
    assert embeddings._batches(["x" * 15_000] * 3, embeddings.batch_chars(cfg)) == [(0, 1), (1, 2), (2, 3)]


def test_limiter_keeps_to_requests_per_minute(clock):
    limiter = embeddings.RateLimiter(tpm_budget=1_000_000, rpm=3)
    for _ in range(3):
        limiter.wait_for(1)
    assert clock.sleeps == []
    limiter.wait_for(1)
    assert clock.sleeps == [pytest.approx(60.5)]


def test_limiter_keeps_to_tokens_per_minute(clock):
    limiter = embeddings.RateLimiter(tpm_budget=100, rpm=100)
    limiter.wait_for(60)
    clock.sleep(10)
    limiter.wait_for(40)
    assert clock.sleeps == [10]
    limiter.wait_for(30)  # 130 > 100: wait for the first to age out
    assert clock.sleeps[1:] == [pytest.approx(50.5)]
    # Too big for any minute: sent once the window is empty.
    limiter.wait_for(500)
    assert sum(clock.sleeps) == pytest.approx(10 + 50.5 + 60.5)


def test_limiter_learns_characters_per_token(clock):
    limiter = embeddings.RateLimiter(tpm_budget=100, rpm=10)
    assert limiter.estimate(["x" * 250]) == int(250 / embeddings.CHARS_PER_TOKEN_GUESS * 1.1) + 1
    entry = limiter.wait_for(limiter.estimate(["x" * 400]))
    limiter.billed(entry, ["x" * 400], 100)
    assert entry[1] == 100 and limiter.chars_per_token() == 4
    limiter.unsent(entry)
    assert not limiter.recent


def test_requests_are_paced_to_the_rate_limit(voyage):
    vectors, _ = embeddings.embed(["x" * 20_000] * 4, embeddings.config())  # one per batch
    assert vectors.shape == (4, 1024) and len(voyage.requests) == 4
    assert voyage.clock.sleeps == [pytest.approx(60.5)]  # the 4th waits for the RPM window


def test_429_waits_for_retry_after(voyage):
    replies = iter([(429, {"detail": "slow down"}, {"Retry-After": "7"}), None])
    voyage.respond = lambda body: next(replies) or (200, voyage.reply(body["input"]))
    vectors, _ = embeddings.embed(["a"], embeddings.config())
    assert vectors.shape == (1, 1024) and len(voyage.requests) == 2
    assert voyage.clock.sleeps == [7.0]


def test_429_without_retry_after_waits_a_minute(voyage):
    replies = iter([(429, {"detail": "slow down"}), None])
    voyage.respond = lambda body: next(replies) or (200, voyage.reply(body["input"]))
    embeddings.embed(["a"], embeddings.config())
    assert voyage.clock.sleeps[0] == embeddings.RATE_WINDOW_SECONDS


def test_429_gives_up_after_its_attempts(voyage):
    voyage.respond = lambda body: (429, {"detail": "slow down"})
    with pytest.raises(embeddings.ApiError, match="HTTP 429"):
        embeddings.embed(["a"], embeddings.config())
    assert len(voyage.requests) == embeddings.MAX_RATE_LIMIT_ATTEMPTS


def test_describe_plan_estimates_time(env):
    plan = embeddings.Plan(embed=[embeddings.StaleTicket(1, "1", "x" * 520_000, "d", {})])
    assert "about 14 minute(s)" in embeddings.describe_plan(plan, embeddings.config())


@pytest.mark.parametrize("status", [400, 422, 429, 500])
def test_client_error_details_echoing_ticket_text_are_never_exposed(voyage, caplog, status):
    """The API may echo the rejected input in `detail`; it must not reach messages or logs."""
    secret = "Student Jane Doe, DOB 2012-04-01, can't see her test"
    voyage.respond = lambda body: (status, {"detail": f"Invalid input: {secret}"})
    logged = []
    handler = embeddings.logger.add(lambda m: logged.append(str(m)), level="DEBUG")
    caplog.set_level(logging.DEBUG, logger="root")
    try:
        with pytest.raises(embeddings.ApiError) as raised:
            embeddings.embed([secret], embeddings.config())
    finally:
        embeddings.logger.remove(handler)
    error = raised.value
    assert f"HTTP {status}" in str(error)
    assert "Jane" not in str(error) and all("Jane" not in str(v) for v in vars(error).values())
    assert error.__cause__ is None and error.__suppress_context__
    assert all("Jane" not in line for line in logged)
    assert "Jane" not in caplog.text


class _Reply(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _http_error(code, body=b"", headers=None):
    return urllib.error.HTTPError("https://x", code, "msg", headers or {}, io.BytesIO(body))


def _attempts(monkeypatch, outcomes):
    """Patches the opener to raise/return each outcome in turn; returns the call log."""
    calls = []

    def fake_open(request, timeout):
        calls.append(request)
        outcome = outcomes[len(calls) - 1]
        if isinstance(outcome, BaseException):
            raise outcome
        return _Reply(outcome)

    monkeypatch.setattr(embeddings, "_open", fake_open)
    monkeypatch.setattr(embeddings.time, "sleep", lambda s: None)
    return calls


@pytest.mark.parametrize("error", [
    _http_error(429), _http_error(503),
    urllib.error.URLError(ConnectionRefusedError()), urllib.error.URLError(socket.gaierror("no such host")),
])
def test_retries_only_when_nothing_was_processed(monkeypatch, error):
    calls = _attempts(monkeypatch, [error, b'{"data": []}'])
    assert embeddings.post_json("https://x", "k", {}) == {"data": []}
    assert len(calls) == 2


@pytest.mark.parametrize("error", [
    _http_error(500), _http_error(401, b'{"detail":"bad key"}'),
    TimeoutError("read timed out"), ConnectionResetError("reset"),
    urllib.error.URLError(TimeoutError("timed out")),
])
def test_does_not_retry_failures_that_may_have_been_billed(monkeypatch, error):
    calls = _attempts(monkeypatch, [error, b'{"data": []}'])
    with pytest.raises(embeddings.ApiError) as raised:
        embeddings.post_json("https://x", "k", {"input": ["secret ticket text"]})
    assert "secret ticket text" not in str(raised.value)
    assert len(calls) == 1


def test_redirects_are_refused(monkeypatch):
    calls = _attempts(monkeypatch, [_http_error(307, headers={"Location": "http://evil.example/collect"})])
    with pytest.raises(embeddings.ApiError, match="not following"):
        embeddings.post_json("https://x", "k", {"input": ["secret"]})
    assert len(calls) == 1


def test_opener_does_not_follow_redirects():
    handler = embeddings._NoRedirect()
    assert handler.redirect_request(None, None, 302, "Found", {}, "http://evil.example/") is None


# --- tickets ----------------------------------------------------------------------


@pytest.fixture
def tickets():
    collection = mongomock.MongoClient().db.tickets
    collection.insert_many([
        _ticket("1", "Login fails", [_comment("I can't log in", role="requester")]),
        _ticket("2", "Export broken", [_comment("CSV export is empty", role="requester")]),
        _ticket("3", "", []),  # no text: never embedded
    ])
    return collection


def _fake_embed(calls):
    def embed_fn(texts, cfg):
        calls.append(list(texts))
        return np.full((len(texts), cfg.dimensions), 0.5, dtype=np.float32), 10 * len(texts)
    return embed_fn


def _backfill(tickets, cfg, calls, **kwargs):
    return embeddings.run_backfill(tickets, cfg, embeddings.plan_backfill(tickets, cfg),
                                   embed_fn=_fake_embed(calls), **kwargs)


def test_backfill_stores_vectors_and_skips_current(env, tickets):
    cfg, calls = embeddings.config(), []
    summary = _backfill(tickets, cfg, calls)
    assert summary == {"embedded": 2, "changed": 0, "cleared": 0, "tokens": 20, "cancelled": 0, "complete": True}
    doc = tickets.find_one({"zendesk_id": "1"})
    assert len(doc["embedding"]) == 1024
    assert {k: doc["embedding_meta"][k] for k in ("endpoint", "model", "dimensions")} == {
        "endpoint": embeddings.default_url(), "model": "voyage-3.5", "dimensions": 1024}
    assert doc["embedding_meta"]["text_sha256"] == embeddings.text_hash("Login fails\n\nI can't log in")
    assert "embedding" not in tickets.find_one({"zendesk_id": "3"})

    assert _backfill(tickets, cfg, calls)["embedded"] == 0
    assert len(calls) == 1


def test_changed_text_endpoint_model_or_dimensions_re_embed(env, tickets):
    cfg, calls = embeddings.config(), []
    _backfill(tickets, cfg, calls)

    tickets.update_one({"zendesk_id": "1"}, {"$set": {"subject": "Login fails on Safari"}})
    assert [t.zendesk_id for t in embeddings.plan_backfill(tickets, cfg).embed] == ["1"]
    for name, value in [("EMBEDDING_DIMENSIONS", "512"), ("EMBEDDING_MODEL", "voyage-3.5-lite"),
                        ("EMBEDDING_API_URL", "https://proxy.example.com/v1/embeddings")]:
        env.setenv(name, value)
        assert len(embeddings.plan_backfill(tickets, embeddings.config()).embed) == 2
        env.delenv(name)


def test_only_the_approved_plan_is_sent(env, tickets):
    cfg, calls = embeddings.config(), []
    plan = embeddings.plan_backfill(tickets, cfg)
    tickets.insert_one(_ticket("4", "Added after the count", [_comment("new", role="requester")]))
    embeddings.run_backfill(tickets, cfg, plan, embed_fn=_fake_embed(calls))
    assert sum(len(c) for c in calls) == 2
    assert "embedding" not in tickets.find_one({"zendesk_id": "4"})


def test_ticket_edited_during_embedding_is_not_overwritten(env, tickets):
    cfg = embeddings.config()
    plan = embeddings.plan_backfill(tickets, cfg)

    def edit_while_embedding(texts, cfg):
        tickets.update_one({"zendesk_id": "1"}, {"$push": {"transcript": _comment("more detail", role="requester")}})
        return np.zeros((len(texts), cfg.dimensions), dtype=np.float32), 0

    summary = embeddings.run_backfill(tickets, cfg, plan, embed_fn=edit_while_embedding)
    assert summary["embedded"] == 1 and summary["changed"] == 1
    assert "embedding" not in tickets.find_one({"zendesk_id": "1"})


def test_tickets_whose_text_is_gone_lose_their_vectors(env, tickets):
    cfg, calls = embeddings.config(), []
    _backfill(tickets, cfg, calls)
    tickets.update_one({"zendesk_id": "2"}, {"$set": {"subject": "", "transcript": []}})
    plan = embeddings.plan_backfill(tickets, cfg)
    assert plan.embed == [] and len(plan.clear) == 1
    summary = embeddings.run_backfill(tickets, cfg, plan, embed_fn=_fake_embed(calls))
    assert summary["cleared"] == 1
    doc = tickets.find_one({"zendesk_id": "2"})
    assert "embedding" not in doc and "embedding_meta" not in doc


def test_duplicate_zendesk_ids_each_get_their_own_vector(env):
    collection = mongomock.MongoClient().db.tickets
    collection.insert_many([_ticket("7", "First copy", []), _ticket("7", "Second copy", [])])
    cfg = embeddings.config()

    def by_text(texts, cfg):
        return np.array([[float(len(t))] * cfg.dimensions for t in texts], dtype=np.float32), 0

    embeddings.run_backfill(collection, cfg, embeddings.plan_backfill(collection, cfg), embed_fn=by_text)
    for doc in collection.find():
        assert doc["embedding"][0] == float(len(doc["subject"]))
        assert doc["embedding_meta"]["text_sha256"] == embeddings.text_hash(doc["subject"])


def test_backfill_stops_between_batches_and_keeps_progress(env, tickets, monkeypatch):
    monkeypatch.setattr(embeddings, "MAX_BATCH_TEXTS", 1)
    calls, stops = [], iter([False, True])
    summary = _backfill(tickets, embeddings.config(), calls, should_stop=lambda: next(stops))
    assert (summary["embedded"], summary["cancelled"], summary["complete"]) == (1, 1, False)
    assert tickets.count_documents({"embedding": {"$exists": True}}) == 1


def test_embed_error_keeps_earlier_batches(env, tickets, monkeypatch):
    monkeypatch.setattr(embeddings, "MAX_BATCH_TEXTS", 1)
    calls = []

    def flaky(texts, cfg):
        if calls:
            raise embeddings.ApiError("HTTP 500")
        return _fake_embed(calls)(texts, cfg)

    cfg = embeddings.config()
    with pytest.raises(embeddings.ApiError):
        embeddings.run_backfill(tickets, cfg, embeddings.plan_backfill(tickets, cfg), embed_fn=flaky)
    assert tickets.count_documents({"embedding": {"$exists": True}}) == 1


def test_only_one_backfill_at_a_time(env, tickets):
    cfg = embeddings.config()
    plan = embeddings.plan_backfill(tickets, cfg)

    def nested(texts, cfg):
        with pytest.raises(embeddings.BackfillRunning):
            embeddings.run_backfill(tickets, cfg, plan, embed_fn=_fake_embed([]))
        return np.zeros((len(texts), cfg.dimensions), dtype=np.float32), 0

    embeddings.run_backfill(tickets, cfg, plan, embed_fn=nested)
    # Released afterwards.
    assert tickets.database[embeddings.LOCK_COLLECTION].count_documents({}) == 0
    embeddings.run_backfill(tickets, cfg, embeddings.plan_backfill(tickets, cfg), embed_fn=_fake_embed([]))


def test_expired_lock_is_taken_over(env, tickets):
    tickets.database[embeddings.LOCK_COLLECTION].insert_one({
        "_id": embeddings.LOCK_ID, "holder": "crashed",
        "expires_at": datetime.now(timezone.utc) - timedelta(minutes=1),
    })
    cfg = embeddings.config()
    summary = embeddings.run_backfill(tickets, cfg, embeddings.plan_backfill(tickets, cfg), embed_fn=_fake_embed([]))
    assert summary["complete"]


# --- vector index -----------------------------------------------------------------


class FakeSearchIndexes:
    """A collection stand-in recording search-index calls (mongomock has none)."""

    def __init__(self, existing=None):
        self.indexes = existing or []
        self.created = []

    def list_search_indexes(self):
        return list(self.indexes)

    def create_search_index(self, model):
        self.created.append(model.document)
        self.indexes.append({"name": model.document["name"], "type": model.document["type"],
                             "latestDefinition": model.document["definition"]})


def test_ensure_vector_index_creates_once():
    tickets = FakeSearchIndexes()
    assert search.check_vector_index(tickets, 1024) == "missing"
    assert search.ensure_vector_index(tickets, 1024) is True
    assert search.ensure_vector_index(tickets, 1024) is False
    assert search.check_vector_index(tickets, 1024) == "ok"
    [created] = tickets.created
    assert created["name"] == "tickets_vector" and created["type"] == "vectorSearch"


def _index(definition, type_="vectorSearch"):
    return FakeSearchIndexes([{"name": "tickets_vector", "type": type_, "latestDefinition": definition}])


@pytest.mark.parametrize("definition, type_, problem", [
    (search.vector_index_definition(512), "vectorSearch", "numDimensions is 512"),
    ({"fields": [{**search.vector_index_definition(1024)["fields"][0], "similarity": "euclidean"},
                 search.vector_index_definition(1024)["fields"][1]]}, "vectorSearch", "similarity"),
    ({"fields": [{**search.vector_index_definition(1024)["fields"][0], "path": "vec"},
                 search.vector_index_definition(1024)["fields"][1]]}, "vectorSearch", "no vector field on 'embedding'"),
    ({"fields": search.vector_index_definition(1024)["fields"][:1]}, "vectorSearch", "no filter field on 'status'"),
    ({"fields": search.vector_index_definition(1024)["fields"] + [{"type": "filter", "path": "tags"}]},
     "vectorSearch", "unexpected filter field on 'tags'"),
    (search.vector_index_definition(1024), "search", "not vectorSearch"),
])
def test_any_definition_difference_is_a_mismatch(definition, type_, problem):
    with pytest.raises(search.VectorIndexMismatch, match=problem):
        search.ensure_vector_index(_index(definition, type_), 1024)


def test_index_status_checks_before_embedding(env, monkeypatch):
    def unavailable(*a):
        raise search.SearchUnavailable("no")

    monkeypatch.setattr(search, "check_vector_index", unavailable)
    assert embeddings.index_status(None, embeddings.config()) == "unavailable"


@pytest.mark.parametrize("summary, status, expected", [
    ({"complete": True, "changed": 0}, "missing", "tickets_vector created"),
    ({"complete": False, "changed": 0}, "missing", "not created yet"),
    ({"complete": True, "changed": 1}, "missing", "not created yet"),
    ({"complete": True, "changed": 0}, "unavailable", "skipped"),
])
def test_index_created_only_after_a_complete_backfill(env, summary, status, expected):
    tickets = FakeSearchIndexes()
    embeddings.finish_index(tickets, embeddings.config(), summary, status)
    assert expected in summary["index"]
    assert bool(tickets.created) == (expected == "tickets_vector created")


@pytest.mark.parametrize("code", [400, 422, 500])
def test_error_bodies_echoing_ticket_text_are_never_exposed(monkeypatch, code):
    """An API or proxy may echo the rejected input; it must not reach messages or logs."""
    secret = "Student Jane Doe, DOB 2012-04-01, can't see her test"
    echo = f'{{"detail": "Invalid input: {secret}"}}'.encode()
    _attempts(monkeypatch, [_http_error(code, echo)])
    logged = []
    handler = embeddings.logger.add(lambda m: logged.append(str(m)), level="DEBUG")
    try:
        with pytest.raises(embeddings.ApiError) as raised:
            embeddings.post_json("https://x", "k", {"input": [secret]})
    finally:
        embeddings.logger.remove(handler)
    error = raised.value
    assert f"HTTP {code}" in str(error)
    assert secret not in str(error) and "Jane" not in str(error)
    assert all(secret not in str(value) for value in vars(error).values())
    assert all(secret not in line for line in logged)


def test_token_limit_is_recognised_without_keeping_the_body(monkeypatch):
    body = b'{"detail": "Request exceeds max allowed tokens per batch: <ticket text here>"}'
    _attempts(monkeypatch, [_http_error(400, body)])
    with pytest.raises(embeddings.ApiError) as raised:
        embeddings.post_json("https://x", "k", {"input": ["x"]})
    assert raised.value.token_limit is True
    assert "ticket text" not in str(raised.value)
    assert "too many tokens" in str(raised.value)


# --- CLI backfill: index preflight --------------------------------------------------


@pytest.fixture
def cli(env, tickets, monkeypatch):
    """embeddings.main wired to the mongomock tickets, with embed calls recorded."""
    import decision_engine.persistence

    monkeypatch.setattr("settings.load", lambda: None)
    monkeypatch.setattr(decision_engine.persistence, "get_tickets_collection", lambda: tickets)
    calls = []

    def fake_embed(texts, cfg, input_type="document"):
        calls.append(list(texts))
        return np.zeros((len(texts), cfg.dimensions), dtype=np.float32), 1

    monkeypatch.setattr(embeddings, "embed", fake_embed)
    return calls


def _existing_index(monkeypatch, dimensions):
    index = {"name": search.VECTOR_INDEX, "type": "vectorSearch",
             "latestDefinition": search.vector_index_definition(dimensions)}
    monkeypatch.setattr(search, "_search_indexes", lambda tickets: {search.VECTOR_INDEX: index})


def test_cli_dimension_change_aborts_before_paid_calls_or_writes(cli, tickets, monkeypatch, capsys):
    # Embedded at 1024 with a matching index, then settings change to 512.
    embeddings.run_backfill(tickets, embeddings.config(), embeddings.plan_backfill(tickets, embeddings.config()),
                            embed_fn=lambda texts, cfg: (np.ones((len(texts), 1024), dtype=np.float32), 0))
    _existing_index(monkeypatch, 1024)
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "512")
    cli.clear()

    assert embeddings.main(["backfill", "--yes"]) == 2
    assert "numDimensions is 1024, expected 512" in capsys.readouterr().err
    assert cli == []  # no paid calls
    assert {len(d["embedding"]) for d in tickets.find({"embedding": {"$exists": True}})} == {1024}


def test_cli_without_atlas_search_embeds_and_skips_the_index(cli, tickets, monkeypatch, capsys):
    def unavailable(tickets):
        raise search.SearchUnavailable("no Atlas Search")

    monkeypatch.setattr(search, "_search_indexes", unavailable)
    assert embeddings.main(["backfill", "--yes"]) == 0
    assert "skipped" in capsys.readouterr().out
    assert sum(len(c) for c in cli) == 2


def test_cli_index_changed_during_the_run_is_an_error(cli, tickets, monkeypatch, capsys):
    states = iter([{}, {search.VECTOR_INDEX: {"name": search.VECTOR_INDEX, "type": "vectorSearch",
                                              "latestDefinition": search.vector_index_definition(256)}}])
    monkeypatch.setattr(search, "_search_indexes", lambda tickets: next(states))
    assert embeddings.main(["backfill", "--yes"]) == 2
    captured = capsys.readouterr()
    assert "Embedded: 2" in captured.out and "numDimensions is 256" in captured.err


def test_cli_creates_the_index_after_a_complete_backfill(cli, tickets, monkeypatch, capsys):
    created = []
    monkeypatch.setattr(search, "_search_indexes", lambda tickets: {})
    monkeypatch.setattr(search, "ensure_vector_index", lambda t, d: created.append(d) or True)
    assert embeddings.main(["backfill", "--yes"]) == 0
    assert created == [1024]
    assert "tickets_vector created" in capsys.readouterr().out
