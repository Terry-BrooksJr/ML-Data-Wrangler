import json
import pathlib

import mongomock
import numpy as np
import pytest

import bertopic_model
import embeddings

THEMES = [
    ["import", "csv", "upload", "error", "row", "fail"],
    ["login", "password", "reset", "sso", "account", "locked"],
    ["report", "score", "student", "export", "grade", "missing"],
]
PER_THEME = 30
DIMS = 32


def _corpus(seed=0):
    """Three well-separated themes: tokens and vectors that agree on the grouping."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(len(THEMES), DIMS)) * 5
    ids, texts, tokens, vectors = [], [], [], []
    for i in range(PER_THEME * len(THEMES)):
        theme = i % len(THEMES)
        words = list(rng.choice(THEMES[theme], 5))
        ids.append(str(1000 + i))
        texts.append(" ".join(words))
        tokens.append(words)
        vectors.append(centers[theme] + rng.normal(size=DIMS) * 0.1)
    corpus = bertopic_model.Corpus(ids, texts, np.asarray(vectors, dtype=np.float32),
                                   {"endpoint": "e", "model": "m", "dimensions": DIMS})
    return corpus, tokens


@pytest.fixture(scope="module")
def trained():
    corpus, tokens = _corpus()
    result = bertopic_model.train(corpus, tokens, bertopic_model.Settings(min_cluster_size=10))
    return corpus, tokens, result


# --- corpus -----------------------------------------------------------------------


def _ticket(zendesk_id, subject, vector=None, meta=None):
    doc = {"zendesk_id": zendesk_id, "subject": subject, "transcript": []}
    if vector is not None:
        doc["embedding"] = vector
    if meta is not None:
        doc["embedding_meta"] = meta
    return doc


def _meta(subject, model="voyage-3.5", dims=2, **changes):
    return {"endpoint": "https://api.voyageai.com/v1/embeddings", "model": model, "dimensions": dims,
            "text_sha256": embeddings.text_hash(subject), **changes}


def test_load_corpus_keeps_only_current_embeddings_from_the_main_model():
    tickets = mongomock.MongoClient().support_db.tickets
    tickets.insert_many([
        _ticket("1", "csv import fails", [0.1, 0.2], _meta("csv import fails")),
        _ticket("2", "sso login loop", [0.3, 0.4], _meta("sso login loop")),
        _ticket("3", "no vector yet"),
        _ticket("4", "edited since", [0.5, 0.6], _meta("the old text")),
        _ticket("5", "other model", [0.7, 0.8, 0.9], _meta("other model", model="voyage-3-large", dims=3)),
        _ticket("6", "", [0.1, 0.1], _meta("")),
        _ticket("7", "wrong length", [0.1], _meta("wrong length")),
    ])

    corpus = bertopic_model.load_corpus(tickets)

    assert corpus.ids == ["1", "2"]
    assert corpus.texts == ["csv import fails", "sso login loop"]
    assert corpus.vectors.shape == (2, 2)
    assert corpus.embedding["model"] == "voyage-3.5"
    assert corpus.skipped == {"no_text": 1, "no_embedding": 1, "stale": 2, "other_model": 1}


def test_load_corpus_with_no_embeddings_is_empty():
    tickets = mongomock.MongoClient().support_db.tickets
    tickets.insert_one(_ticket("1", "nothing embedded"))
    corpus = bertopic_model.load_corpus(tickets)
    assert corpus.ids == [] and corpus.skipped == {"no_embedding": 1}


# --- training ---------------------------------------------------------------------


def test_train_finds_one_topic_per_theme(trained):
    corpus, _, result = trained
    summaries = bertopic_model.topic_summaries(result)
    assert len(summaries) == len(THEMES)
    assert sorted(s["count"] for s in summaries) == [PER_THEME] * len(THEMES)
    for summary in summaries:
        members = [corpus.ids[i] for i, t in enumerate(result.topics) if t == summary["topic_id"]]
        themes = {(int(m) - 1000) % len(THEMES) for m in members}
        assert len(themes) == 1
        assert set(summary["words"][:3]) <= set(THEMES[themes.pop()])


def test_train_never_loads_pytorch(trained):
    import sys

    assert "torch" not in sys.modules and "sentence_transformers" not in sys.modules


def test_train_reassigns_outliers_by_default():
    corpus, tokens = _corpus()
    # Five near-identical tickets far from the rest: too few for a topic of 10.
    rng = np.random.default_rng(1)
    stray = (rng.normal(size=DIMS) * 20 + rng.normal(size=(5, DIMS)) * 0.1).astype(np.float32)
    corpus.ids += [str(9000 + i) for i in range(5)]
    corpus.texts += ["odd one"] * 5
    corpus.vectors = np.vstack([corpus.vectors, stray])
    tokens = tokens + [["odd", "one"]] * 5

    kept = bertopic_model.train(corpus, tokens, bertopic_model.Settings(min_cluster_size=10, n_neighbors=4, reduce_outliers=False))
    moved = bertopic_model.train(corpus, tokens, bertopic_model.Settings(min_cluster_size=10, n_neighbors=4))

    assert kept.outliers_before > 0 and kept.topics.count(bertopic_model.OUTLIER) == kept.outliers_before
    assert moved.topics.count(bertopic_model.OUTLIER) == 0


def test_train_refuses_too_few_tickets():
    corpus, tokens = _corpus()
    corpus.ids, corpus.vectors = corpus.ids[:15], corpus.vectors[:15]
    with pytest.raises(ValueError, match="too few"):
        bertopic_model.train(corpus, tokens[:15], bertopic_model.Settings())


def test_default_cluster_size_is_one_percent_with_a_floor():
    settings = bertopic_model.Settings()
    assert settings.cluster_size(300) == bertopic_model.MIN_CLUSTER_SIZE_FLOOR
    assert settings.cluster_size(5000) == 50
    assert bertopic_model.Settings(min_cluster_size=7).cluster_size(5000) == 7


# --- scoring ----------------------------------------------------------------------


def test_topic_diversity():
    assert bertopic_model.topic_diversity([["a", "b"], ["c", "d"]]) == 1.0
    assert bertopic_model.topic_diversity([["a", "b"], ["a", "b"]]) == 0.5
    assert bertopic_model.topic_diversity([]) is None


def test_coherence_prefers_real_topics_and_ignores_unknown_words():
    _, tokens = _corpus()
    real = bertopic_model.coherence([t[:4] for t in THEMES], tokens)
    mixed = bertopic_model.coherence([[THEMES[0][0], THEMES[1][0], THEMES[2][0], THEMES[0][1]]] * 3, tokens)
    assert real > mixed
    assert bertopic_model.coherence([THEMES[0][:4] + ["nowhere"]], tokens) == pytest.approx(
        bertopic_model.coherence([THEMES[0][:4]], tokens))
    assert bertopic_model.coherence([["nowhere", "never"]], tokens) is None


def test_compare_lda_scores_a_saved_run(tmp_path):
    _, tokens = _corpus()
    run_dir = tmp_path / "20261001T000000Z-abc123"
    run_dir.mkdir()
    (run_dir / "run.json").write_text(json.dumps({"topics": [{"words": THEMES[0][:4]}, {"words": THEMES[1][:4]}]}))

    comparison = bertopic_model.compare_lda("20261001T000000Z-abc123", tokens, base_dir=tmp_path)

    assert comparison["num_topics"] == 2 and comparison["topic_diversity"] == 1.0
    assert comparison["coherence_c_v"] is not None
    with pytest.raises(FileNotFoundError):
        bertopic_model.compare_lda("missing", tokens, base_dir=tmp_path)


# --- saving and assigning ---------------------------------------------------------


def test_save_run_writes_model_assignments_and_details(trained, tmp_path):
    corpus, tokens, result = trained
    run = bertopic_model.save_run(result, corpus, {"spacy_model": "en_core_web_lg"},
                                  bertopic_model.metrics(result, tokens), base_dir=tmp_path)

    run_dir = pathlib.Path(run["model_dir"])
    assert (run_dir / bertopic_model.MODEL_FILE).exists()
    assert json.loads((run_dir / bertopic_model.ASSIGNMENTS_FILE).read_text()) == dict(zip(corpus.ids, result.topics))
    saved = json.loads((run_dir / "run.json").read_text())
    assert saved["method"] == "bertopic"
    assert saved["metrics"]["num_topics"] == len(THEMES)
    assert saved["settings"]["min_cluster_size"] == 10
    assert saved["dataset"]["ticket_count"] == len(corpus.ids)
    assert saved["charts"]["barchart_html"] and pathlib.Path(saved["charts"]["barchart_html"]).exists()
    assert "BERTopic run" in bertopic_model.format_report(run)


def test_saved_runs_stay_out_of_the_lda_run_list(monkeypatch, tmp_path):
    import topic_store

    monkeypatch.setenv("TOPIC_MODEL_DIR", str(tmp_path / "topic_models"))
    monkeypatch.delenv("BERTOPIC_MODEL_DIR", raising=False)
    assert bertopic_model.model_dir() != topic_store.model_dir()
    assert bertopic_model.model_dir().name == "bertopic_models"


def test_assign_topics_writes_embedding_topic_and_leaves_topic_alone():
    tickets = mongomock.MongoClient().support_db.tickets
    tickets.insert_many([{"zendesk_id": "1", "topic": {"run_id": "lda"}}, {"zendesk_id": "2"}])
    run = {"run_id": "bt", "topics": [{"topic_id": 0, "words": ["csv", "import", "error", "row", "fail", "x"]}]}

    summary = bertopic_model.assign_topics(tickets, run, ["1", "2", "3"], [0, -1, 0])

    assert summary == {"assigned": 1, "outliers": 1, "not_in_mongo": 1}
    one = tickets.find_one({"zendesk_id": "1"})
    assert one["topic"] == {"run_id": "lda"}
    assert one["embedding_topic"]["topic"] == 0
    assert one["embedding_topic"]["words"] == ["csv", "import", "error", "row", "fail"]
    assert tickets.find_one({"zendesk_id": "2"})["embedding_topic"]["topic"] == -1
