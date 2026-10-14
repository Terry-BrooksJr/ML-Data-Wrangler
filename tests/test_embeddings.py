import io
import urllib.error

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
    # A single over-size text still gets its own batch.
    assert embeddings._batches(["x" * 50, "y"]) == [(0, 1), (1, 2)]
    assert embeddings._batches([]) == []


def test_embed_sends_voyage_request(env):
    sent = []

    def fake_post(url, api_key, body):
        sent.append((url, api_key, body))
        data = [{"index": i, "embedding": [float(i)] * 1024} for i in range(len(body["input"]))]
        return {"data": list(reversed(data)), "usage": {"total_tokens": 9}}

    env.setattr(embeddings, "post_json", fake_post)
    vectors, tokens = embeddings.embed(["a", "b"], embeddings.config(), input_type="query")
    url, key, body = sent[0]
    assert url == embeddings.DEFAULT_URL and key == "test-key"
    assert body == {"model": "voyage-3.5", "input": ["a", "b"], "input_type": "query",
                    "output_dimension": 1024, "truncation": True}
    assert vectors.shape == (2, 1024) and vectors[1, 0] == 1.0 and tokens == 9


def test_embed_rejects_wrong_dimensions(env):
    env.setattr(embeddings, "post_json", lambda *a: {"data": [{"index": 0, "embedding": [0.0] * 3}]})
    with pytest.raises(RuntimeError, match="3-dimension"):
        embeddings.embed(["a"], embeddings.config())


def test_post_json_does_not_retry_client_errors(monkeypatch):
    attempts = []

    def fail(request, timeout):
        attempts.append(1)
        raise urllib.error.HTTPError(request.full_url, 401, "Unauthorized", {}, io.BytesIO(b'{"detail":"bad key"}'))

    monkeypatch.setattr("urllib.request.urlopen", fail)
    with pytest.raises(RuntimeError, match="HTTP 401") as raised:
        embeddings.post_json("https://x", "k", {"input": ["secret ticket text"]})
    assert "secret ticket text" not in str(raised.value)
    assert len(attempts) == 1


def test_post_json_retries_rate_limits(monkeypatch):
    attempts = []

    class Reply(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def flaky(request, timeout):
        attempts.append(1)
        if len(attempts) < 3:
            raise urllib.error.HTTPError(request.full_url, 429, "Too Many", {}, io.BytesIO(b""))
        return Reply(b'{"data": []}')

    monkeypatch.setattr("urllib.request.urlopen", flaky)
    monkeypatch.setattr(embeddings.time, "sleep", lambda s: None)
    assert embeddings.post_json("https://x", "k", {}) == {"data": []}
    assert len(attempts) == 3


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


def test_embed_tickets_stores_vectors_and_skips_current(env, tickets):
    cfg, calls = embeddings.config(), []
    summary = embeddings.embed_tickets(tickets, cfg, embed_fn=_fake_embed(calls))
    assert summary == {"embedded": 2, "tokens": 20, "cancelled": 0}
    doc = tickets.find_one({"zendesk_id": "1"})
    assert len(doc["embedding"]) == 1024
    assert doc["embedding_meta"]["model"] == "voyage-3.5"
    assert doc["embedding_meta"]["dimensions"] == 1024
    assert doc["embedding_meta"]["text_sha256"] == embeddings.text_hash("Login fails\n\nI can't log in")
    assert "embedding" not in tickets.find_one({"zendesk_id": "3"})

    # Nothing changed: nothing re-sent.
    assert embeddings.embed_tickets(tickets, cfg, embed_fn=_fake_embed(calls))["embedded"] == 0
    assert len(calls) == 1


def test_changed_text_model_or_dimensions_re_embed(env, tickets):
    cfg, calls = embeddings.config(), []
    embeddings.embed_tickets(tickets, cfg, embed_fn=_fake_embed(calls))

    tickets.update_one({"zendesk_id": "1"}, {"$set": {"subject": "Login fails on Safari"}})
    assert [z for z, _, _ in embeddings.stale_tickets(tickets, cfg)] == ["1"]

    env.setenv("EMBEDDING_DIMENSIONS", "512")
    assert len(embeddings.stale_tickets(tickets, embeddings.config())) == 2
    env.setenv("EMBEDDING_DIMENSIONS", "1024")
    env.setenv("EMBEDDING_MODEL", "voyage-3.5-lite")
    assert len(embeddings.stale_tickets(tickets, embeddings.config())) == 2


def test_embed_tickets_stops_between_batches_and_keeps_progress(env, tickets, monkeypatch):
    monkeypatch.setattr(embeddings, "MAX_BATCH_TEXTS", 1)
    calls, stops = [], iter([False, True])
    summary = embeddings.embed_tickets(tickets, embeddings.config(), should_stop=lambda: next(stops),
                                       embed_fn=_fake_embed(calls))
    assert summary == {"embedded": 1, "tokens": 10, "cancelled": 1}
    assert tickets.count_documents({"embedding": {"$exists": True}}) == 1


def test_embed_error_keeps_earlier_batches(env, tickets, monkeypatch):
    monkeypatch.setattr(embeddings, "MAX_BATCH_TEXTS", 1)
    calls = []

    def flaky(texts, cfg):
        if calls:
            raise RuntimeError("HTTP 500")
        return _fake_embed(calls)(texts, cfg)

    with pytest.raises(RuntimeError):
        embeddings.embed_tickets(tickets, embeddings.config(), embed_fn=flaky)
    assert tickets.count_documents({"embedding": {"$exists": True}}) == 1


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
        self.indexes.append({"name": model.document["name"], "latestDefinition": model.document["definition"]})


def test_ensure_vector_index_creates_once():
    tickets = FakeSearchIndexes()
    assert search.ensure_vector_index(tickets, 1024) is True
    assert search.ensure_vector_index(tickets, 1024) is False
    [created] = tickets.created
    assert created["name"] == "tickets_vector" and created["type"] == "vectorSearch"
    assert created["definition"]["fields"][0] == {"type": "vector", "path": "embedding",
                                                  "numDimensions": 1024, "similarity": "cosine"}


def test_ensure_vector_index_refuses_dimension_change():
    tickets = FakeSearchIndexes([{"name": "tickets_vector", "latestDefinition": search.vector_index_definition(512)}])
    with pytest.raises(search.VectorIndexMismatch, match="512-dimension"):
        search.ensure_vector_index(tickets, 1024)
