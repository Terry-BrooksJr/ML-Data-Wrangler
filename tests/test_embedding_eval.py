import copy
import io
import json

import numpy as np
import pytest

import embedding_eval as ee
from fixtures import CATEGORY_FIXTURES


def _ticket(zendesk_id, subject, comments):
    return {"zendesk_id": zendesk_id, "subject": subject, "transcript": comments}


def _comment(body, role=None, internal=False):
    return {"author": "a", "role": role, "body": body, "timestamp": "2026-10-01", "is_internal_note": internal}


def test_neighbours_excludes_self_and_orders_by_similarity():
    vectors = np.array([[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9]], dtype=np.float32)
    nearest = ee.neighbours(vectors, k=2)
    assert nearest[0].tolist() == [1, 3]
    assert nearest[2].tolist() == [3, 1]
    assert all(i not in row for i, row in enumerate(nearest))


def test_neighbours_tolerates_zero_vectors():
    nearest = ee.neighbours(np.array([[0, 0], [1, 0], [1, 0.1]], dtype=np.float32), k=1)
    assert nearest.shape == (3, 1)


def test_score_perfect_and_majority_vote():
    labels = ["flag_solve", "flag_solve", "manual_triage", "manual_triage"]
    nearest = np.array([[1, 2], [0, 3], [3, 0], [2, 1]])
    result = ee.score(labels, nearest)
    assert result["top1"] == 1.0
    assert result["precision"] == 0.5
    # 1-1 ties go to the nearer neighbour, which has the right label.
    assert result["knn_accuracy"] == 1.0
    assert result["per_category"]["flag_solve"] == 0.5
    assert result["per_category"]["escalate_leadership"] is None


def test_chance_precision():
    assert ee.chance_precision(["a", "a", "b", "b"]) == pytest.approx(1 / 3)
    assert ee.chance_precision(["a"]) == 0.0


def _eval_set(tmp_path, per_label=3):
    cases = []
    for n, (name, fixture) in enumerate(CATEGORY_FIXTURES.items()):
        for i in range(per_label):
            ticket = copy.deepcopy(fixture)
            ticket["zendesk_id"] = f"{n}{i}"
            cases.append({"ticket": ticket, "expected": name})
    path = tmp_path / "eval.json"
    path.write_text(json.dumps(cases, default=str))
    return path


@pytest.fixture
def fake_candidates(monkeypatch):
    calls = []

    def make(name, sends_text):
        def embedder():
            def embed(texts):
                calls.append((name, len(texts)))
                # Same subject -> same direction, so same-category tickets cluster.
                return np.array([[hash(t.split("\n")[0]) % 97 + 1, len(t) % 13 + 1] for t in texts], dtype=np.float32), 7
            return embed
        return embedder

    def fake(voyage_model, openai_model):
        return {
            "spacy": ee.Candidate("fake spacy", "spacy", "m", None, make("spacy", False), False),
            "voyage": ee.Candidate("fake voyage", "voyage", voyage_model, "EMBEDDING_API_KEY", make("voyage", True), True),
            "openai": ee.Candidate("fake openai", "openai", openai_model, "OPENAI_API_KEY", make("openai", True), True),
        }

    monkeypatch.setattr(ee, "candidates", fake)
    monkeypatch.setattr("settings.load", lambda: None)
    return calls


def test_main_skips_providers_without_keys(tmp_path, monkeypatch, capsys, fake_candidates):
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert ee.main([str(_eval_set(tmp_path)), "--cache", str(tmp_path / "cache"), "-k", "2"]) == 0
    out = capsys.readouterr().out
    assert "skipping fake voyage: EMBEDDING_API_KEY is not set" in out
    assert "fake spacy" in out and "random neighbours (baseline)" in out
    assert [name for name, _ in fake_candidates] == ["spacy"]


def test_main_asks_before_sending_text_and_needs_yes_off_a_terminal(tmp_path, monkeypatch, capsys, fake_candidates):
    monkeypatch.setenv("EMBEDDING_API_KEY", "k")
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    assert ee.main([str(_eval_set(tmp_path)), "--providers", "voyage", "--cache", str(tmp_path / "c")]) == 1
    assert "pass --yes" in capsys.readouterr().err
    assert fake_candidates == []


def test_main_caches_embeddings(tmp_path, monkeypatch, capsys, fake_candidates):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    args = [str(_eval_set(tmp_path)), "--providers", "openai", "--cache", str(tmp_path / "c"), "--yes", "-k", "2"]
    assert ee.main(args) == 0
    assert ee.main(args + ["--show-neighbours", "2"]) == 0
    out = capsys.readouterr().out
    assert len(fake_candidates) == 1  # second run read the cache
    assert "cached" in out
    assert "-> same" in out or "-> DIFF" in out


def test_api_embedder_batches_orders_and_counts_tokens(monkeypatch):
    sent = []

    def fake_post(url, api_key, body):
        sent.append(body)
        data = [{"index": i, "embedding": [float(len(t)), 1.0]} for i, t in enumerate(body["input"])]
        return {"data": list(reversed(data)), "usage": {"total_tokens": len(body["input"])}}

    monkeypatch.setattr(ee.embeddings, "post_json", fake_post)
    monkeypatch.setattr(ee, "BATCH_SIZE", 2)
    vectors, tokens = ee._api_embedder("https://x", "k", "m", {"input_type": "document"})(["a", "bb", "ccc"])
    assert vectors[:, 0].tolist() == [1.0, 2.0, 3.0]
    assert tokens == 3
    assert [b["input"] for b in sent] == [["a", "bb"], ["ccc"]]
    assert sent[0]["input_type"] == "document" and sent[0]["model"] == "m"


def test_voyage_candidate_uses_app_settings(monkeypatch):
    monkeypatch.setenv("EMBEDDING_API_KEY", "k")
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "512")
    sent = []

    def fake_post(url, api_key, body):
        sent.append((url, api_key, body))
        return {"data": [{"index": i, "embedding": [0.0] * 512} for i in range(len(body["input"]))]}

    monkeypatch.setattr(ee.embeddings, "post_json", fake_post)
    vectors, _ = ee.voyage_embedder("voyage-3.5-lite")(["a"])
    url, key, body = sent[0]
    assert url == ee.embeddings.DEFAULT_URL and key == "k"
    assert body["model"] == "voyage-3.5-lite" and body["output_dimension"] == 512
    assert vectors.shape == (1, 512)
    # The cache key includes dimensions, so changing them doesn't reuse old vectors.
    assert ee.candidates("voyage-3.5", "x")["voyage"].model == "voyage-3.5@512"


def test_k_must_be_positive(tmp_path, capsys):
    with pytest.raises(SystemExit):
        ee.main([str(_eval_set(tmp_path)), "-k", "0"])
    assert "must be at least 1" in capsys.readouterr().err


def test_missing_spacy_model_is_skipped(tmp_path, monkeypatch, capsys, fake_candidates):
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    real_find_spec = ee.importlib.util.find_spec
    monkeypatch.setattr(ee.importlib.util, "find_spec",
                        lambda name, *a: None if name == "en_core_web_lg" else real_find_spec(name, *a))
    assert ee.main([str(_eval_set(tmp_path)), "--cache", str(tmp_path / "c"), "-k", "2"]) == 0
    assert "skipping fake spacy: the en_core_web_lg model isn't installed" in capsys.readouterr().out
    assert fake_candidates == []


def test_provider_that_cannot_load_does_not_stop_the_others(tmp_path, monkeypatch, capsys, fake_candidates):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    original = ee.candidates

    def broken_spacy(voyage_model, openai_model):
        found = original(voyage_model, openai_model)

        def fail():
            raise OSError("can't find model 'en_core_web_lg'")

        found["spacy"].make = fail
        return found

    monkeypatch.setattr(ee, "candidates", broken_spacy)
    args = [str(_eval_set(tmp_path)), "--providers", "spacy", "openai", "--cache", str(tmp_path / "c"), "--yes", "-k", "2"]
    assert ee.main(args) == 0
    captured = capsys.readouterr()
    assert "fake spacy failed: unavailable" in captured.err
    assert "fake openai" in captured.out


def test_each_view_scores_only_tickets_with_text_in_it(tmp_path, monkeypatch, capsys, fake_candidates):
    path = _eval_set(tmp_path)
    cases = json.loads(path.read_text())
    # Three tickets lose their subject and requester: no "opening" text, but
    # their comments still give them "full" text.
    for case in cases[:3]:
        case["ticket"]["subject"] = ""
        for comment in case["ticket"]["transcript"]:
            comment["role"] = None
    path.write_text(json.dumps(cases))
    assert ee.main([str(path), "--providers", "spacy", "--text", "opening", "full",
                    "--cache", str(tmp_path / "c"), "-k", "2"]) == 0
    out = capsys.readouterr().out
    assert f"Text: opening ({len(cases) - 3} tickets" in out
    assert f"Text: full ({len(cases)} tickets" in out
    assert [n for _, n in fake_candidates] == [len(cases) - 3, len(cases)]


def test_cache_key_includes_the_endpoint(tmp_path):
    calls = []

    def make():
        def embed(texts):
            calls.append(1)
            return np.ones((len(texts), 2), dtype=np.float32), 0
        return embed

    a = ee.Candidate("v", "voyage", "voyage-3.5@1024", None, make, True, "https://api.voyageai.com/v1/embeddings")
    b = ee.Candidate("v", "voyage", "voyage-3.5@1024", None, make, True, "https://proxy.example.com/v1/embeddings")
    ee.cached_embed(a, "opening", ["t"], tmp_path)
    assert ee.cached_embed(a, "opening", ["t"], tmp_path)[2] is True
    assert ee.cached_embed(b, "opening", ["t"], tmp_path)[2] is False
    assert len(calls) == 2
