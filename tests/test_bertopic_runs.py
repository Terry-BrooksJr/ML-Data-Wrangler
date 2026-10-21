import json
import os
import pathlib
from datetime import datetime, timezone

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import bertopic_dialog
import bertopic_model
import bertopic_runs
import mongomock
import numpy as np
import pytest
import topic_store
from PyQt5.QtWidgets import QApplication

THEMES = [
    ["import", "csv", "upload", "error", "row", "fail"],
    ["login", "password", "reset", "sso", "account", "locked"],
    ["report", "score", "student", "export", "grade", "missing"],
]
PER_THEME = 30
DIMS = 32
LDA_RUN = "20261001T000000Z-abc123"


class FakeNlp:
    meta = {"lang": "en", "name": "core_web_lg", "version": "3.7.1"}


def _corpus(seed=0):
    """Three well-separated themes: tokens and vectors that agree on the grouping."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(len(THEMES), DIMS)) * 5
    ids, texts, tokens, vectors = [], [], [], []
    for i in range(PER_THEME * len(THEMES)):
        words = list(rng.choice(THEMES[i % len(THEMES)], 5))
        ids.append(str(1000 + i))
        texts.append(" ".join(words))
        tokens.append(words)
        vectors.append(centers[i % len(THEMES)] + rng.normal(size=DIMS) * 0.1)
    corpus = bertopic_model.Corpus(
        ids,
        texts,
        np.asarray(vectors, dtype=np.float32),
        {"endpoint": "e", "model": "voyage-3.5", "dimensions": DIMS},
        {"stale": 2},
    )
    return corpus, tokens


def _patch_corpus(monkeypatch, corpus, tokens):
    monkeypatch.setattr(
        bertopic_runs.bertopic_model, "load_corpus", lambda tickets: corpus
    )
    monkeypatch.setattr(
        bertopic_runs, "tokenize", lambda nlp, stop_words, texts: tokens
    )


def _lda_dir(base):
    run_dir = base / LDA_RUN
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": LDA_RUN,
                "ran_at": "2026-10-01T00:00:00+00:00",
                "best_num_topics": 2,
                "topics": [
                    {"topic_id": 0, "share": 0.5, "words": THEMES[0][:4]},
                    {"topic_id": 1, "share": 0.5, "words": THEMES[1][:4]},
                ],
            }
        )
    )
    return base


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def saved(tmp_path_factory):
    """A real run, trained and saved through train_and_save, with assignment and an LDA comparison."""
    tmp = tmp_path_factory.mktemp("bertopic")
    corpus, tokens = _corpus()
    tickets = mongomock.MongoClient().support_db.tickets
    tickets.insert_many(
        [{"zendesk_id": i, "topic": {"run_id": "lda"}} for i in corpus.ids]
    )
    statuses = []
    with pytest.MonkeyPatch.context() as mp:
        _patch_corpus(mp, corpus, tokens)
        run = bertopic_runs.train_and_save(
            tickets,
            FakeNlp(),
            ["the"],
            bertopic_runs.TrainOptions(
                min_cluster_size=10, compare_lda=LDA_RUN, assign=True
            ),
            status=statuses.append,
            base_dir=tmp / "bertopic_models",
            lda_dir=_lda_dir(tmp / "topic_models"),
        )
    return tmp, tickets, run, statuses


# --- training ---------------------------------------------------------------------


def test_options_map_to_settings():
    assert bertopic_runs.TrainOptions().settings() == bertopic_model.Settings()
    settings = bertopic_runs.TrainOptions(
        min_cluster_size=12, nr_topics=5, keep_outliers=True
    ).settings()
    assert (
        settings.min_cluster_size,
        settings.nr_topics,
        settings.reduce_outliers,
    ) == (12, 5, False)


def test_train_and_save_saves_compares_and_reports(saved):
    tmp, _, run, statuses = saved

    assert pathlib.Path(run["model_dir"]).parent == tmp / "bertopic_models"
    assert run["metrics"]["num_topics"] == len(THEMES)
    assert (
        run["comparison"]["run_id"] == LDA_RUN and run["comparison"]["num_topics"] == 2
    )
    assert run["warnings"] == []
    assert statuses[0].startswith("Reading tickets") and any(
        "Saving" in s for s in statuses
    )


def test_train_and_save_assigns_embedding_topic_only(saved):
    _, tickets, run, _ = saved

    assert run["assignment"]["assigned"] == PER_THEME * len(THEMES)
    for doc in tickets.find():
        assert doc["topic"] == {"run_id": "lda"}
        assert doc["embedding_topic"]["run_id"] == run["run_id"]


def test_train_and_save_never_assigns_unless_asked(monkeypatch, tmp_path):
    corpus, tokens = _corpus()
    _patch_corpus(monkeypatch, corpus, tokens)
    monkeypatch.setattr(bertopic_runs.bertopic_model, "train", lambda c, t, s: "result")
    monkeypatch.setattr(bertopic_runs.bertopic_model, "metrics", lambda r, t: {})
    monkeypatch.setattr(
        bertopic_runs.bertopic_model,
        "save_run",
        lambda *args, **kwargs: {"run_id": "r", "comparison": args[4]},
    )
    assigned = []
    monkeypatch.setattr(
        bertopic_runs.bertopic_model,
        "assign_topics",
        lambda *args: assigned.append(args),
    )

    run = bertopic_runs.train_and_save(
        None,
        FakeNlp(),
        [],
        bertopic_runs.TrainOptions(compare_lda="missing"),
        lda_dir=tmp_path,
    )

    assert assigned == [] and "assignment" not in run
    # A comparison that can't be read doesn't lose the trained model.
    assert run["comparison"] is None
    assert run["warnings"] and "missing" in run["warnings"][0]


def test_train_and_save_needs_embedded_tickets(monkeypatch):
    empty = bertopic_model.Corpus([], [], np.empty((0, 0)), {}, {"no_embedding": 4})
    _patch_corpus(monkeypatch, empty, [])
    with pytest.raises(ValueError, match="embeddings backfill"):
        bertopic_runs.train_and_save(None, FakeNlp(), [], bertopic_runs.TrainOptions())


# --- saved runs -------------------------------------------------------------------


def test_list_runs_finds_saved_runs_only(saved):
    tmp, _, run, _ = saved
    base = tmp / "bertopic_models"

    listed = bertopic_runs.list_runs(base)
    assert [r["run_id"] for r in listed] == [run["run_id"]]
    assert isinstance(listed[0]["ran_at"], datetime)
    assert bertopic_runs.list_runs(tmp / "missing") == []
    # LDA runs aren't BERTopic runs.
    assert bertopic_runs.list_runs(tmp / "topic_models") == []


def test_lda_run_list_skips_bertopic_runs(saved):
    tmp, _, run, _ = saved
    mixed = tmp / "mixed"
    mixed.mkdir()
    for source in (tmp / "topic_models" / LDA_RUN, pathlib.Path(run["model_dir"])):
        (mixed / source.name).mkdir()
        (mixed / source.name / "run.json").write_text((source / "run.json").read_text())

    assert [r["run_id"] for r in topic_store.list_runs(base_dir=mixed)] == [LDA_RUN]
    assert [r["run_id"] for r in bertopic_runs.lda_runs(mixed)] == [LDA_RUN]

    runs = mongomock.MongoClient().support_db[topic_store.runs_collection()]
    runs.insert_many(
        [
            {"run_id": "old", "ran_at": 1},
            {"run_id": "lda", "method": "lda", "ran_at": 2},
            {"run_id": "bt", "method": "bertopic", "ran_at": 3},
        ]
    )
    assert [r["run_id"] for r in topic_store.list_runs(runs)] == ["lda", "old"]


def test_list_runs_newest_first(tmp_path):
    for run_id, ran_at in (
        ("a", "2026-10-01T00:00:00+00:00"),
        ("b", "2026-10-02T00:00:00+00:00"),
    ):
        (tmp_path / run_id).mkdir()
        (tmp_path / run_id / "run.json").write_text(
            json.dumps({"run_id": run_id, "ran_at": ran_at, "method": "bertopic"})
        )
    assert [r["run_id"] for r in bertopic_runs.list_runs(tmp_path)] == ["b", "a"]


def test_chart_paths_come_from_the_runs_own_folder(saved, tmp_path):
    tmp, _, run, _ = saved
    base = tmp / "bertopic_models"

    charts = bertopic_runs.chart_paths(run, base)
    assert (
        pathlib.Path(charts["barchart_html"])
        == pathlib.Path(run["model_dir"]).resolve() / "barchart.html"
    )
    # A recorded path elsewhere is ignored; a missing or undrawn chart is None.
    elsewhere = {
        **run,
        "charts": {"barchart_html": "/etc/passwd", "intertopic_html": None},
    }
    assert (
        bertopic_runs.chart_paths(elsewhere, base)["barchart_html"]
        == charts["barchart_html"]
    )
    assert bertopic_runs.chart_paths(elsewhere, base)["intertopic_html"] is None
    assert bertopic_runs.chart_paths(run, tmp_path) == {
        "barchart_html": None,
        "intertopic_html": None,
    }
    assert bertopic_runs.chart_paths({**run, "run_id": "../x"}, base) == {
        "barchart_html": None,
        "intertopic_html": None,
    }


RUN = {
    "run_id": "20261015T120000Z-def456",
    "ran_at": datetime(2026, 10, 15, 12, 0, tzinfo=timezone.utc),
    "embedding": {"model": "voyage-3.5", "dimensions": 1024},
    "dataset": {"skipped": {"no_embedding": 3}},
    "metrics": {
        "num_topics": 2,
        "coherence_c_v": 0.61234,
        "topic_diversity": 0.9,
        "outliers_before": 7,
        "outliers_after": 0,
        "ticket_count": 120,
    },
    "comparison": {
        "run_id": LDA_RUN,
        "num_topics": 8,
        "coherence_c_v": 0.4,
        "topic_diversity": None,
    },
    "topics": [
        {"topic_id": 1, "count": 80, "share": 2 / 3, "words": ["csv", "import"]},
        {"topic_id": 0, "count": 40, "share": 1 / 3, "words": ["login", "sso"]},
    ],
    "charts": {"barchart_html": None, "intertopic_html": None},
}


def test_text_helpers():
    assert bertopic_runs.describe_run(RUN) == (
        "20261015T120000Z-def456 · 2026-10-15 12:00 UTC · 2 topics · c_v 0.612 · diversity 0.900 · 120 tickets"
    )
    assert bertopic_runs.comparison_rows(RUN) == [
        ("BERTopic", "2", "0.612", "0.900"),
        (f"LDA {LDA_RUN}", "8", "0.400", "–"),
    ]
    assert bertopic_runs.comparison_rows({**RUN, "comparison": None}) == [
        ("BERTopic", "2", "0.612", "0.900")
    ]
    assert bertopic_runs.format_topics(RUN) == (
        "1. 67% of tickets (80): csv, import\n" "2. 33% of tickets (40): login, sso"
    )
    assert bertopic_runs.format_topics({}) == "No topics were found."
    summary = bertopic_runs.summary_text(RUN)
    assert (
        "120 tickets embedded with voyage-3.5" in summary
        and "3 no embedding" in summary
    )


# --- dialogs ----------------------------------------------------------------------


def test_options_dialog_defaults_and_choices(qapp):
    dialog = bertopic_dialog.BertopicOptionsDialog(
        [{"run_id": LDA_RUN, "ran_at": "2026-10-01"}]
    )
    assert dialog.options() == bertopic_runs.TrainOptions()

    dialog.min_cluster_size.setValue(15)
    dialog.nr_topics.setValue(6)
    dialog.keep_outliers.setChecked(True)
    dialog.compare_lda.setCurrentIndex(1)
    dialog.assign.setChecked(True)
    assert dialog.options() == bertopic_runs.TrainOptions(15, 6, True, LDA_RUN, True)


def test_results_dialog(qapp, monkeypatch, tmp_path):
    chart = tmp_path / "barchart.html"
    opened = []
    monkeypatch.setattr(bertopic_dialog.webbrowser, "open", opened.append)

    dialog = bertopic_dialog.BertopicResultsDialog(
        RUN, {"barchart_html": str(chart), "intertopic_html": None}
    )

    assert (
        dialog.comparison.rowCount() == 2
        and dialog.comparison.item(1, 2).text() == "0.400"
    )
    assert "csv, import" in dialog.topic_text.toPlainText()
    assert dialog.chart_buttons["barchart_html"].isEnabled()
    assert not dialog.chart_buttons["intertopic_html"].isEnabled()
    dialog.chart_buttons["barchart_html"].click()
    assert opened == [chart.resolve().as_uri()]
