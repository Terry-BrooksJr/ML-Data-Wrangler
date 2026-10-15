import io
import socket
import urllib.error
from datetime import datetime, timedelta, timezone

import mongomock
import numpy as np
import pytest

import embeddings
from decision_engine import search


@pytest.fixture
def env(monkeypatch):
    for name in ("EMBEDDING_API_URL", "EMBEDDING_MODEL", "EMBEDDING_DIMENSIONS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("EMBEDDING_API_KEY", "test-key")
    return monkeypatch


def _ticket(zendesk_id, subject, comments, **extra):
    return {"zendesk_id": zendesk_id, "subject": subject, "transcript": comments, "status": "open", **extra}


def _comment(body, role=None, internal=False):
    return {"author": "a", "role": role, "body": body, "timestamp": "2026-10-01", "is_internal_note": internal}


# --- config -----------------------------------------------------------------------


def test_config_defaults_to_voyage(env):
    cfg = embeddings.config()
    assert cfg == embeddings.Config("https://api.voyageai.com/v1/embeddings", "voyage-3.5", 1024, "test-key")


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


def _reply(texts, dims=1024):
    data = [{"index": i, "embedding": [float(i)] * dims} for i in range(len(texts))]
    return {"data": list(reversed(data)), "usage": {"total_tokens": len(texts)}}


def test_embed_sends_voyage_request(env):
    sent = []

    def fake_post(url, api_key, body):
        sent.append((url, api_key, body))
        return _reply(body["input"])

    env.setattr(embeddings, "post_json", fake_post)
    vectors, tokens = embeddings.embed(["a", "b"], embeddings.config(), input_type="query")
    url, key, body = sent[0]
    assert url == embeddings.DEFAULT_URL and key == "test-key"
    assert body == {"model": "voyage-3.5", "input": ["a", "b"], "input_type": "query",
                    "output_dimension": 1024, "truncation": True}
    assert vectors.shape == (2, 1024) and vectors[1, 0] == 1.0 and tokens == 2


def test_embed_splits_a_batch_over_the_token_limit(env):
    sizes = []

    def fake_post(url, api_key, body):
        sizes.append(len(body["input"]))
        if len(body["input"]) > 2:
            raise embeddings.ApiError("HTTP 400", 400, "Request to model exceeds max allowed tokens per batch")
        return _reply(body["input"])

    env.setattr(embeddings, "post_json", fake_post)
    vectors, tokens = embeddings.embed(["a", "b", "c", "d", "e"], embeddings.config())
    assert sizes == [5, 2, 3, 1, 2]
    assert vectors.shape == (5, 1024) and tokens == 5


def test_other_client_errors_are_not_split(env):
    calls = []

    def fake_post(url, api_key, body):
        calls.append(1)
        raise embeddings.ApiError("HTTP 400", 400, "invalid model")

    env.setattr(embeddings, "post_json", fake_post)
    with pytest.raises(embeddings.ApiError):
        embeddings.embed(["a", "b"], embeddings.config())
    assert len(calls) == 1


def test_embed_rejects_wrong_dimensions(env):
    env.setattr(embeddings, "post_json", lambda *a: {"data": [{"index": 0, "embedding": [0.0] * 3}]})
    with pytest.raises(embeddings.ApiError, match="3-dimension"):
        embeddings.embed(["a"], embeddings.config())


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
        "endpoint": embeddings.DEFAULT_URL, "model": "voyage-3.5", "dimensions": 1024}
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
