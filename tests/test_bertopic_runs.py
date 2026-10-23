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
    """Stubs the corpus and tokens; returns the outcome filters load_corpus was given."""
    requested = []

    def load_corpus(tickets, outcomes=None):
        requested.append(outcomes)
        return corpus

    monkeypatch.setattr(bertopic_runs.bertopic_model, "load_corpus", load_corpus)
    monkeypatch.setattr(
        bertopic_runs, "tokenize", lambda nlp, stop_words, texts: tokens
    )
    return requested


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


# --- outcomes and the topic list --------------------------------------------------

OUTCOMES = ["client_education", "learnosity_bug", "possible_feature_enhancement"]


def _embedded_tickets(corpus):
    """mongomock tickets whose stored embeddings load_corpus accepts: subject
    text (no transcript), a matching text hash, and corpus.outcomes."""
    import embeddings

    tickets = mongomock.MongoClient().support_db.tickets
    for zid, text, vector, outcome in zip(
        corpus.ids, corpus.texts, corpus.vectors, corpus.outcomes
    ):
        tickets.insert_one(
            {
                "zendesk_id": zid,
                "subject": text,
                "transcript": [],
                "historical_outcome": outcome,
                "embedding": [float(x) for x in vector],
                "embedding_meta": {
                    "endpoint": "e",
                    "model": "voyage-3.5",
                    "dimensions": DIMS,
                    "text_sha256": embeddings.text_hash(text),
                },
            }
        )
    return tickets


def test_train_and_save_trains_only_on_the_chosen_outcomes(tmp_path, monkeypatch):
    corpus, tokens = _corpus()
    # Two of the three themes are client_education: one theme alone is a
    # single cluster, which HDBSCAN doesn't report.
    corpus.outcomes = [
        "client_education" if i % 3 < 2 else "learnosity_bug"
        for i in range(len(corpus.ids))
    ]
    tickets = _embedded_tickets(corpus)
    chosen = {
        zid for zid, o in zip(corpus.ids, corpus.outcomes) if o == "client_education"
    }
    by_text = dict(zip(corpus.texts, tokens))
    seen = []

    def tokenize(nlp, stop_words, texts):
        seen.extend(texts)
        return [by_text[t] for t in texts]

    # load_corpus is real; only lemmatizing (spaCy) is stubbed.
    monkeypatch.setattr(bertopic_runs, "tokenize", tokenize)
    run = bertopic_runs.train_and_save(
        tickets,
        FakeNlp(),
        ["the"],
        bertopic_runs.TrainOptions(min_cluster_size=10, outcomes=["client_education"]),
        base_dir=tmp_path,
    )

    assigned = json.loads(
        (pathlib.Path(run["model_dir"]) / bertopic_model.ASSIGNMENTS_FILE).read_text()
    )
    assert (
        set(assigned) == chosen
        and len(seen) == len(chosen) == run["metrics"]["ticket_count"]
    )
    assert run["dataset"]["outcomes"] == ["client_education"]
    for topic in run["topics"]:
        assert topic["outcomes"] == {"client_education": topic["count"]}
        assert set(topic["examples"]) <= chosen
        assert len(topic["examples"]) == bertopic_model.EXAMPLES_PER_TOPIC
    assert "Outcomes: client_education." in bertopic_runs.summary_text(run)


def test_assignment_blocked_by_another_process_is_recorded(tmp_path, monkeypatch):
    from datetime import timedelta

    import embeddings

    corpus, tokens = _corpus()
    _patch_corpus(monkeypatch, corpus, tokens)
    tickets = mongomock.MongoClient().support_db.tickets
    tickets.insert_many([{"zendesk_id": i} for i in corpus.ids])
    tickets.database[embeddings.LOCK_COLLECTION].insert_one(
        {
            "_id": bertopic_model.ASSIGNMENT_LOCK_ID,
            "holder": "another-process",
            "expires_at": datetime.now(timezone.utc) + timedelta(minutes=5),
        }
    )
    run = bertopic_runs.train_and_save(
        tickets,
        FakeNlp(),
        ["the"],
        bertopic_runs.TrainOptions(min_cluster_size=10, assign=True),
        base_dir=tmp_path,
    )
    assert run["assignment"]["status"] == "not_started"
    assert any("--assign-run " + run["run_id"] in w for w in run["warnings"])
    saved = json.loads((pathlib.Path(run["model_dir"]) / "run.json").read_text())
    assert saved["assignment"]["status"] == "not_started"


def test_train_and_save_explains_no_tickets_with_chosen_outcomes(tmp_path, monkeypatch):
    empty = bertopic_model.Corpus([], [], np.empty((0, 0)), {})
    _patch_corpus(monkeypatch, empty, [])
    with pytest.raises(ValueError, match="chosen outcomes"):
        bertopic_runs.train_and_save(
            None,
            FakeNlp(),
            ["the"],
            bertopic_runs.TrainOptions(outcomes=["x"]),
            base_dir=tmp_path,
        )


DETAILED_RUN = {
    **RUN,
    "dataset": {"skipped": {}, "outcomes": ["client_education", None]},
    "topics": [
        {
            "topic_id": 1,
            "count": 80,
            "share": 2 / 3,
            "words": ["csv", "import"],
            "outcomes": {"client_education": 60, "unset": 20},
            "examples": ["11", "12"],
        },
        {
            "topic_id": 0,
            "count": 40,
            "share": 1 / 3,
            "words": ["login", "sso"],
            "outcomes": {"client_education": 40},
            "examples": ["21"],
        },
    ],
}


def test_summary_and_topics_show_outcomes_and_examples():
    assert "Outcomes: client_education, unset." in bertopic_runs.summary_text(
        DETAILED_RUN
    )
    assert "Outcomes: every ticket." in bertopic_runs.summary_text(RUN)
    text = bertopic_runs.format_topics(DETAILED_RUN)
    assert "Outcomes: client_education 60, unset 20" in text
    assert "Example tickets: 11, 12" in text


def test_topic_list_csv_with_and_without_subjects():
    import csv
    import io

    tickets = mongomock.MongoClient().support_db.tickets
    tickets.insert_many(
        [
            {"zendesk_id": "11", "subject": "CSV import fails"},
            {"zendesk_id": "12", "subject": "Import row errors"},
            {"zendesk_id": "21", "subject": "SSO loop"},
            {"zendesk_id": "99", "subject": "not an example"},
        ]
    )
    subjects = bertopic_runs.example_subjects(tickets, DETAILED_RUN)
    assert subjects == {
        "11": "CSV import fails",
        "12": "Import row errors",
        "21": "SSO loop",
    }
    assert bertopic_runs.example_subjects(None, DETAILED_RUN) == {}

    rows = list(
        csv.DictReader(
            io.StringIO(bertopic_runs.topic_list_csv(DETAILED_RUN, subjects))
        )
    )
    assert [r["rank"] for r in rows] == ["1", "2"]
    first = rows[0]
    assert first["outcomes"] == "client_education 60, unset 20"
    assert first["top_words"] == "csv, import"
    assert first["example_tickets"] == "11, 12"
    assert first["example_subjects"] == "CSV import fails | Import row errors"
    assert first["proposed_name"] == "" and first["decision"] == ""

    bare = list(csv.DictReader(io.StringIO(bertopic_runs.topic_list_csv(DETAILED_RUN))))
    assert bare[0]["example_subjects"] == ""


def test_topic_list_neutralizes_formulas():
    import csv
    import io

    run = {
        **DETAILED_RUN,
        "topics": [{**DETAILED_RUN["topics"][0], "words": ["-total", "import"]}],
    }
    subjects = {"11": '=HYPERLINK("http://x","click")', "12": "@SUM(A1)"}
    row = next(csv.DictReader(io.StringIO(bertopic_runs.topic_list_csv(run, subjects))))
    assert row["example_subjects"] == '\'=HYPERLINK("http://x","click") | \'@SUM(A1)'
    assert row["top_words"] == "'-total, import"
    for text in ("+1", "-1", "=1", "@x", "\tx", "\rx"):
        assert bertopic_runs.spreadsheet_safe(text) == "'" + text
    assert bertopic_runs.spreadsheet_safe("CSV import") == "CSV import"


def test_options_dialog_outcome_choice(qapp):
    dialog = bertopic_dialog.BertopicOptionsDialog(
        [], [("learnosity_bug", 308), ("client_education", 76), (None, 42)]
    )
    labels = [box.text() for _, box in dialog.outcome_boxes]
    assert labels == [
        "learnosity_bug (308)",
        "client_education (76)",
        "No outcome set (42)",
    ]
    # All ticked: every ticket, no filter.
    assert dialog.options().outcomes is None

    dialog.outcome_boxes[0][1].setChecked(False)
    assert dialog.options().outcomes == ["client_education", None]

    for _, box in dialog.outcome_boxes:
        box.setChecked(False)
    train = dialog.buttons.button(bertopic_dialog.QDialogButtonBox.Ok)
    assert not train.isEnabled()
    dialog.outcome_boxes[1][1].setChecked(True)
    assert train.isEnabled() and dialog.options().outcomes == ["client_education"]


def _wait_for_export(qapp, dialog, timeout=5.0):
    import time

    deadline = time.monotonic() + timeout
    while dialog._export_thread is not None:
        qapp.processEvents()
        assert time.monotonic() < deadline, "subject lookup did not finish"
        time.sleep(0.01)
    qapp.processEvents()


def test_results_dialog_exports_topic_list_off_the_gui_thread(
    qapp, monkeypatch, tmp_path
):
    import threading

    out = tmp_path / "topics.csv"
    shown = []
    monkeypatch.setattr(
        bertopic_dialog.QFileDialog,
        "getSaveFileName",
        lambda *a: (str(out), "CSV (*.csv)"),
    )
    monkeypatch.setattr(
        bertopic_dialog.QMessageBox, "information", lambda *a: shown.append(a[2])
    )
    release = threading.Event()
    lookup_threads = []

    def slow_subjects(run):
        lookup_threads.append(threading.current_thread())
        release.wait(5)
        return {"11": "CSV import fails"}

    dialog = bertopic_dialog.BertopicResultsDialog(
        DETAILED_RUN, {}, subjects=slow_subjects
    )
    dialog.export_topic_list()
    # The window stays responsive while MongoDB answers.
    assert not dialog.export_button.isEnabled()
    assert dialog.export_button.text() == "Reading subjects…"
    assert not out.exists()
    release.set()
    _wait_for_export(qapp, dialog)
    assert lookup_threads and lookup_threads[0] is not threading.main_thread()
    assert "CSV import fails" in out.read_text()
    assert "Wrote 2 topic(s)" in shown[-1]
    assert dialog.export_button.isEnabled()
    assert (
        dialog.export_button.text()
        == bertopic_dialog.BertopicResultsDialog.EXPORT_LABEL
    )

    def unreachable(run):
        raise RuntimeError("no route to host")

    dialog = bertopic_dialog.BertopicResultsDialog(
        DETAILED_RUN, {}, subjects=unreachable
    )
    dialog.export_topic_list()
    _wait_for_export(qapp, dialog)
    assert "csv, import" in out.read_text()
    assert "MongoDB could not be reached" in shown[-1]


def test_closing_the_dialog_waits_for_the_lookup(qapp, monkeypatch, tmp_path):
    import threading

    monkeypatch.setattr(
        bertopic_dialog.QFileDialog,
        "getSaveFileName",
        lambda *a: (str(tmp_path / "t.csv"), "CSV (*.csv)"),
    )
    monkeypatch.setattr(bertopic_dialog.QMessageBox, "information", lambda *a: None)
    release = threading.Event()
    dialog = bertopic_dialog.BertopicResultsDialog(
        DETAILED_RUN, {}, subjects=lambda run: release.wait(5) and {}
    )
    dialog.export_topic_list()
    threading.Timer(0.2, release.set).start()
    dialog.close()  # must not destroy a running thread
    assert dialog._export_thread is None
