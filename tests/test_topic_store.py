import json
import pathlib

import mongomock
import pytest
from gensim.corpora.dictionary import Dictionary

import topic_store
from LDA_logic import LatentDirichletAllocator
from test_wrangler_mongo import data_dir, wrangler  # noqa: F401 (fixtures)

TOKENS = [
    ["import", "fail", "csv", "upload", "error"],
    ["import", "csv", "error", "row", "fail"],
    ["login", "password", "reset", "sso", "account"],
    ["password", "login", "sso", "locked", "account"],
    ["report", "score", "student", "export", "grade"],
    ["score", "report", "grade", "student", "missing"],
] * 3
IDS = [str(1000 + i) for i in range(len(TOKENS))]


@pytest.fixture(autouse=True)
def local_model_dir(tmp_path, monkeypatch):
    """Runs load only from TOPIC_MODEL_DIR; tests save their runs under tmp_path."""
    monkeypatch.setenv("TOPIC_MODEL_DIR", str(tmp_path))


@pytest.fixture
def trained(monkeypatch):
    allocator = LatentDirichletAllocator(num_of_topics=3, seed=7)
    allocator._tokens = TOKENS
    allocator.prelemma_corpus = [" ".join(t) for t in TOKENS]
    allocator.document_ids = IDS
    allocator.id2word = Dictionary(TOKENS)
    allocator.corpus = [allocator.id2word.doc2bow(t) for t in TOKENS]
    assert allocator.model_trained(iterations=20, passes=3, num_of_topics=3)

    def fake_charts(output_dir=None):
        chart, plot = pathlib.Path(output_dir) / "lda_model.html", pathlib.Path(output_dir) / "coherence.png"
        chart.write_text("<html></html>")
        plot.write_bytes(b"png")
        return str(chart), str(plot)

    monkeypatch.setattr(allocator, "visualize_results", fake_charts)
    return allocator


class FakeNlp:
    meta = {"lang": "en", "name": "core_web_lg", "version": "3.7.1"}


@pytest.fixture
def db():
    return mongomock.MongoClient().support_db


def test_save_run_writes_model_dictionary_and_details(trained, tmp_path):
    run = topic_store.save_run(trained, FakeNlp(), ["the", "a"], base_dir=tmp_path)

    run_dir = pathlib.Path(run["model_dir"])
    assert run_dir.parent == tmp_path
    for name in (topic_store.MODEL_FILE, topic_store.DICTIONARY_FILE, topic_store.RUN_FILE,
                 "lda_model.html", "coherence.png"):
        assert (run_dir / name).exists(), name
    assert run["settings"]["iterations"] == 20 and run["settings"]["seed"] == 7
    assert run["preprocessing"]["spacy_model"] == "en_core_web_lg"
    assert run["preprocessing"]["stopwords_count"] == 2
    assert [c["num_topics"] for c in run["coherence"]] == [2, 3]
    assert run["dataset"]["ticket_count"] == len(TOKENS)
    assert run["topics"] and run["best_num_topics"] in (2, 3)


def test_charts_are_per_run(trained, tmp_path):
    first = topic_store.save_run(trained, FakeNlp(), [], base_dir=tmp_path)
    second = topic_store.save_run(trained, FakeNlp(), [], base_dir=tmp_path)
    assert first["charts"]["lda_html"] != second["charts"]["lda_html"]


def test_save_run_needs_a_model(tmp_path):
    with pytest.raises(ValueError):
        topic_store.save_run(LatentDirichletAllocator(num_of_topics=2), FakeNlp(), [], base_dir=tmp_path)


def test_saved_model_reloads_and_assigns_without_retraining(trained, tmp_path, db):
    run = topic_store.save_run(trained, FakeNlp(), [], base_dir=tmp_path)
    model, dictionary = topic_store.load_run_model(run)

    bow = dictionary.doc2bow(["password", "login", "sso"])
    # Inference starts from a random point, so compare approximately.
    reloaded, original = model.get_document_topics(bow), trained.lda_model.get_document_topics(bow)
    assert [t for t, _ in reloaded] == [t for t, _ in original]
    assert [p for _, p in reloaded] == pytest.approx([p for _, p in original], abs=1e-3)


def test_run_history_in_mongo_and_on_disk(trained, tmp_path, db):
    runs = db[topic_store.RUNS_COLLECTION]
    first = topic_store.save_run(trained, FakeNlp(), [], base_dir=tmp_path)
    second = topic_store.save_run(trained, FakeNlp(), [], base_dir=tmp_path)
    topic_store.record_run(runs, first)
    topic_store.record_run(runs, second)
    topic_store.record_run(runs, second)  # re-recording replaces

    listed = topic_store.list_runs(runs)
    assert [r["run_id"] for r in listed] == [second["run_id"], first["run_id"]]
    assert listed[0]["coherence"] == second["coherence"]

    on_disk = topic_store.list_runs(base_dir=tmp_path)
    assert {r["run_id"] for r in on_disk} == {first["run_id"], second["run_id"]}
    assert "topics" in topic_store.describe_run(on_disk[0]) or on_disk[0]["best_num_topics"]


def test_list_runs_with_no_directory(tmp_path):
    assert topic_store.list_runs(base_dir=tmp_path / "missing") == []


def test_assign_trained_topics_writes_ticket_topic(trained, tmp_path, db):
    tickets = db.tickets
    tickets.insert_many([{"zendesk_id": zendesk_id} for zendesk_id in IDS[:4]])
    run = topic_store.save_run(trained, FakeNlp(), [], base_dir=tmp_path)

    summary = topic_store.assign_trained_topics(tickets, run, trained)

    assert summary == {"assigned": 4, "empty": 0, "not_in_mongo": len(IDS) - 4}
    topic = tickets.find_one({"zendesk_id": IDS[0]})["topic"]
    assert topic["run_id"] == run["run_id"]
    assert 0 <= topic["main_topic"] < trained.best_num_topics
    assert topic["mix"] and topic["words"]


def test_assign_from_saved_run_reads_tickets_from_mongo(trained, tmp_path, db, monkeypatch):
    tickets = db.tickets
    tickets.insert_many([
        {"zendesk_id": "1", "subject": "SSO login", "transcript": [{"body": "password reset for locked account"}]},
        {"zendesk_id": "2", "subject": "zzz", "transcript": []},
    ])
    run = topic_store.save_run(trained, FakeNlp(), [], base_dir=tmp_path)
    monkeypatch.setattr(topic_store, "tokenize", lambda nlp, stopwords, texts: [t.lower().split() for t in texts])

    summary = topic_store.assign_from_saved_run(tickets, run, FakeNlp(), [])

    assert summary["assigned"] == 1 and summary["empty"] == 1
    assert tickets.find_one({"zendesk_id": "1"})["topic"]["run_id"] == run["run_id"]
    assert "topic" not in tickets.find_one({"zendesk_id": "2"})


def test_topic_survives_reingestion(wrangler):  # noqa: F811
    tickets = mongomock.MongoClient().support_db.tickets
    wrangler.write_ticket_to_mongo(101, tickets)
    tickets.update_one({"zendesk_id": "101"}, {"$set": {"topic": {"run_id": "r1", "main_topic": 2}}})

    wrangler.write_ticket_to_mongo(101, tickets)

    assert tickets.find_one({"zendesk_id": "101"})["topic"] == {"run_id": "r1", "main_topic": 2}


def test_mongo_ticket_document_matches_training_document(wrangler):  # noqa: F811
    wrangler.tickets_reshaped()
    wrangler.comments_bound()
    doc = wrangler.ticket_to_document(101)

    assert wrangler.mongo_ticket_document(doc) == wrangler.ticket_documents()[0]
    assert wrangler.ticket_ids() == ["101"]


def test_dataset_fingerprint_changes_with_content():
    a = topic_store.dataset_fingerprint(["x y"], ["1"])
    assert a == topic_store.dataset_fingerprint(["x y"], ["1"])
    assert a != topic_store.dataset_fingerprint(["x z"], ["1"])


def test_run_file_is_valid_json(trained, tmp_path):
    run = topic_store.save_run(trained, FakeNlp(), [], base_dir=tmp_path)
    data = json.loads((pathlib.Path(run["model_dir"]) / topic_store.RUN_FILE).read_text())
    assert data["run_id"] == run["run_id"]


# --- reuse before retraining, and retrieval from S3 -------------------------------


def _match_args(allocator, iterations=20, passes=3, max_topics=3, seed=None):
    if seed is not None:
        allocator.seed = seed
    return (
        topic_store.dataset_fingerprint(allocator.prelemma_corpus, allocator.document_ids),
        allocator.training_settings(max_topics, iterations, passes),
        topic_store.preprocessing_info(FakeNlp(), ["a", "the"]),
    )


def test_matching_run_found_regardless_of_seed(trained, tmp_path):
    run = topic_store.save_run(trained, FakeNlp(), ["a", "the"], base_dir=tmp_path)
    runs = topic_store.list_runs(base_dir=tmp_path)
    match = topic_store.find_matching_run(runs, *_match_args(trained, seed=12345))
    assert match["run_id"] == run["run_id"]


@pytest.mark.parametrize("change", ["iterations", "passes", "max_topics", "tickets", "stopwords"])
def test_no_match_when_anything_but_the_seed_differs(trained, tmp_path, change):
    topic_store.save_run(trained, FakeNlp(), ["a", "the"], base_dir=tmp_path)
    runs = topic_store.list_runs(base_dir=tmp_path)
    dataset, settings, preprocessing = _match_args(trained)
    if change in ("iterations", "passes", "max_topics"):
        settings[change] += 1
    elif change == "tickets":
        dataset = topic_store.dataset_fingerprint(trained.prelemma_corpus[:-1], trained.document_ids[:-1])
    else:
        preprocessing = topic_store.preprocessing_info(FakeNlp(), ["a", "the", "an"])
    assert topic_store.find_matching_run(runs, dataset, settings, preprocessing) is None


def test_newest_matching_run_wins(trained, tmp_path):
    from datetime import datetime, timezone

    old = topic_store.save_run(trained, FakeNlp(), ["a", "the"], base_dir=tmp_path,
                               now=datetime(2026, 1, 1, tzinfo=timezone.utc))
    new = topic_store.save_run(trained, FakeNlp(), ["a", "the"], base_dir=tmp_path,
                               now=datetime(2026, 6, 1, tzinfo=timezone.utc))
    match = topic_store.find_matching_run(topic_store.list_runs(base_dir=tmp_path), *_match_args(trained))
    assert match["run_id"] == new["run_id"] != old["run_id"]


def test_known_runs_merges_mongo_and_disk(trained, tmp_path, db):
    local_only = topic_store.save_run(trained, FakeNlp(), ["a"], base_dir=tmp_path)
    local_only["storage"] = {"url": "s3://b/p/"}
    topic_store.write_run_file(local_only)
    shared = topic_store.save_run(trained, FakeNlp(), ["a"], base_dir=tmp_path)
    runs = db[topic_store.RUNS_COLLECTION]
    topic_store.record_run(runs, shared)
    elsewhere = {**shared, "run_id": "20250101T000000Z-remote", "model_dir": "/other/machine/x"}
    topic_store.record_run(runs, elsewhere)
    # Mongo's copy of local_only missed its upload record.
    topic_store.record_run(runs, {**local_only, "storage": None})

    by_id = {r["run_id"]: r for r in topic_store.known_runs(runs, base_dir=tmp_path)}
    assert set(by_id) == {local_only["run_id"], shared["run_id"], "20250101T000000Z-remote"}
    assert by_id[local_only["run_id"]]["storage"] == {"url": "s3://b/p/"}


def test_ensure_local_keeps_local_runs(trained, tmp_path):
    run = topic_store.save_run(trained, FakeNlp(), ["a"], base_dir=tmp_path)
    assert topic_store.ensure_local(dict(run), base_dir=tmp_path)["model_dir"] == run["model_dir"]


def test_ensure_local_finds_a_run_in_this_computers_folder(trained, tmp_path):
    run = topic_store.save_run(trained, FakeNlp(), ["a"], base_dir=tmp_path)
    moved = {**run, "model_dir": "/another/computer/" + run["run_id"]}
    local = topic_store.ensure_local(moved, base_dir=tmp_path)
    assert local["model_dir"] == run["model_dir"]
    assert local["charts"]["lda_html"] == run["charts"]["lda_html"]


def test_ensure_local_without_upload_explains(trained, tmp_path):
    run = {"run_id": "gone", "model_dir": str(tmp_path / "nope"), "charts": {}}
    with pytest.raises(FileNotFoundError, match="never uploaded"):
        topic_store.ensure_local(run, base_dir=tmp_path)


def test_uploaded_run_downloads_loads_and_relocates_charts(trained, tmp_path, monkeypatch):
    import boto3
    from moto import mock_aws

    import model_storage

    for name, value in {"MODEL_BUCKET": "models", "B2_KEY_ID": "k", "B2_APPLICATION_KEY": "s",
                        "MODEL_STORAGE_REGION": "us-east-1", "MODEL_SIGNING_KEY": "z" * 40}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("B2_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("MODEL_PREFIX", raising=False)
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1", aws_access_key_id="k", aws_secret_access_key="s")
        s3.create_bucket(Bucket="models")
        monkeypatch.setattr(model_storage, "make_client", lambda cfg: s3)
        run = topic_store.save_run(trained, FakeNlp(), ["a"], base_dir=tmp_path / "machine_a")
        model_storage.upload_and_record(run, model_storage.config())

        # Another computer: only the run's details (e.g. from Mongo), no files.
        elsewhere = tmp_path / "machine_b"
        remote = json.loads(json.dumps(run, default=str))
        remote["model_dir"] = "/Users/someone-else/topic_models/" + run["run_id"]
        model, dictionary = topic_store.load_run_model(remote, base_dir=elsewhere)

    assert remote["model_dir"] == str((elsewhere / run["run_id"]).resolve())
    assert pathlib.Path(remote["charts"]["lda_html"]).read_text() == "<html></html>"
    assert model.num_topics == run["best_num_topics"] and len(dictionary) == len(trained.id2word)


def test_recorded_model_dir_is_never_loaded(trained, tmp_path):
    """A run record (e.g. from Mongo) pointing at a folder elsewhere isn't trusted."""
    planted = topic_store.save_run(trained, FakeNlp(), ["a"], base_dir=tmp_path / "downloads")
    record = {**planted, "model_dir": planted["model_dir"]}  # files exist there, but not in the model folder
    with pytest.raises(FileNotFoundError, match="aren't on this computer"):
        topic_store.load_run_model(record, base_dir=tmp_path / "topic_models")
    assert not topic_store.has_local_files(record, base_dir=tmp_path / "topic_models")


@pytest.mark.parametrize("run_id", ["../../etc", "..", "a/b", "/abs"])
def test_traversal_run_ids_are_refused_before_any_download(tmp_path, monkeypatch, run_id):
    import model_storage

    called = []
    monkeypatch.setattr(model_storage, "download_run", lambda *a, **k: called.append(1))
    run = {"run_id": run_id, "storage": {"bucket": "b", "prefix": f"p/{run_id}/"}, "charts": {}}
    with pytest.raises(ValueError, match="plain folder name"):
        topic_store.ensure_local(run, base_dir=tmp_path)
    assert called == []


def test_missing_dictionary_triggers_a_download(trained, tmp_path, monkeypatch):
    import shutil

    import model_storage

    run = topic_store.save_run(trained, FakeNlp(), ["a"], base_dir=tmp_path)
    backup = tmp_path / "backup"
    shutil.copytree(run["model_dir"], backup)
    (pathlib.Path(run["model_dir"]) / topic_store.DICTIONARY_FILE).unlink()
    run["storage"] = {"bucket": "b", "prefix": f"p/{run['run_id']}/"}
    downloads = []

    def fake_download(run_, cfg, dest):
        downloads.append(dest)
        shutil.copy(backup / topic_store.DICTIONARY_FILE, dest / topic_store.DICTIONARY_FILE)
        return dest

    monkeypatch.setattr(model_storage, "config", lambda: object())
    monkeypatch.setattr(model_storage, "download_run", fake_download)
    model, dictionary = topic_store.load_run_model(run, base_dir=tmp_path)
    assert downloads == [pathlib.Path(run["model_dir"])] and len(dictionary) == len(trained.id2word)


def test_chart_paths_outside_the_run_are_dropped(trained, tmp_path):
    run = topic_store.save_run(trained, FakeNlp(), ["a"], base_dir=tmp_path)
    run["charts"]["lda_html"] = str(tmp_path / ".." / "evil.html")
    topic_store.ensure_local(run, base_dir=tmp_path)
    assert run["charts"]["lda_html"] is None or pathlib.Path(run["charts"]["lda_html"]).parent == pathlib.Path(run["model_dir"])


def test_run_file_write_is_atomic(trained, tmp_path, monkeypatch):
    import os

    run = topic_store.save_run(trained, FakeNlp(), ["a"], base_dir=tmp_path)
    before = (pathlib.Path(run["model_dir"]) / topic_store.RUN_FILE).read_text()

    def disk_full(*args):
        raise OSError("No space left on device")

    monkeypatch.setattr(os, "fsync", disk_full)
    run["storage"] = {"url": "s3://b/p/"}
    with pytest.raises(OSError):
        topic_store.write_run_file(run)
    assert (pathlib.Path(run["model_dir"]) / topic_store.RUN_FILE).read_text() == before
    assert [r["run_id"] for r in topic_store.list_runs(base_dir=tmp_path)] == [run["run_id"]]
    assert not list(pathlib.Path(run["model_dir"]).glob(".run.json.*"))
