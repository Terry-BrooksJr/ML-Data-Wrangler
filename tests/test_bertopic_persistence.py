"""BERTopic runs are kept like LDA runs: on disk, uploaded with a signed manifest, recorded in Mongo."""
import json

import mongomock
import pytest

import bertopic_model
import bertopic_runs
import model_storage
import topic_store
from test_model_storage import BUCKET, env, s3  # noqa: F401  (fixtures)

RUN_ID = "20261015T120000Z-bt0001"


@pytest.fixture
def bert_dir(env, tmp_path):
    env.setenv("BERTOPIC_MODEL_DIR", str(tmp_path / "bertopic_models"))
    return tmp_path / "bertopic_models"


def _bertopic_run(base, run_id=RUN_ID, **extra):
    run_dir = base / run_id
    run_dir.mkdir(parents=True)
    (run_dir / bertopic_model.MODEL_FILE).write_bytes(b"pickled model")
    (run_dir / bertopic_model.ASSIGNMENTS_FILE).write_text('{"101": 0}')
    (run_dir / "barchart.html").write_text("<html>bars</html>")
    run = {"run_id": run_id, "ran_at": "2026-10-15T12:00:00+00:00", "method": "bertopic", "model_dir": str(run_dir),
           "metrics": {"num_topics": 1, "coherence_c_v": 0.5, "topic_diversity": 1.0, "ticket_count": 1},
           "topics": [{"topic_id": 0, "count": 1, "share": 1.0, "words": ["csv"]}],
           "charts": {"barchart_html": str(run_dir / "barchart.html"), "intertopic_html": None}, **extra}
    topic_store.write_run_file(run)
    return run


def _keys(s3_client, prefix=""):
    return {o["Key"] for o in s3_client.list_objects_v2(Bucket=BUCKET, Prefix=prefix).get("Contents", [])}


# --- upload -----------------------------------------------------------------------


def test_upload_goes_to_its_own_folder_with_a_signed_manifest(s3, bert_dir):
    run = _bertopic_run(bert_dir)
    bertopic_runs.upload(run, model_storage.config)

    prefix = f"topic-models/bertopic/{RUN_ID}/"
    assert _keys(s3) == {prefix + name for name in (bertopic_model.MODEL_FILE, bertopic_model.ASSIGNMENTS_FILE,
                                                    "barchart.html", "run.json", "manifest.json")}
    assert run["storage"]["prefix"] == prefix and run["warnings"] == []
    saved = json.loads((bert_dir / RUN_ID / "run.json").read_text())
    assert saved["storage"]["url"] == f"s3://{BUCKET}/{prefix}"


def test_upload_problems_become_warnings(bert_dir, monkeypatch):
    run = _bertopic_run(bert_dir)
    bertopic_runs.upload(run, lambda: None)
    assert run["warnings"] == [] and "storage" not in run

    def unconfigured():
        raise model_storage.StorageNotConfigured("no keys")

    bertopic_runs.upload(run, unconfigured)
    assert run["warnings"] == ["Not uploaded to S3: no keys"]

    run["warnings"] = []
    monkeypatch.setattr(model_storage, "upload_and_record", lambda *a: (_ for _ in ()).throw(OSError("offline")))
    bertopic_runs.upload(run, model_storage.config)
    assert "offline" in run["warnings"][0]
    assert f"python -m model_storage upload --bertopic {RUN_ID}" in run["warnings"][0]


# --- Mongo ------------------------------------------------------------------------


def test_recorded_runs_are_listed_as_bertopic_never_as_lda(bert_dir):
    runs = mongomock.MongoClient().db[topic_store.RUNS_COLLECTION]
    run = _bertopic_run(bert_dir, warnings=["local note"], assignment={"assigned": 1})
    runs.insert_one({"run_id": "lda-run", "ran_at": "2026-10-01T00:00:00+00:00", "topics": []})

    bertopic_runs.record_run(runs, run)
    bertopic_runs.record_run(runs, run)  # re-recording replaces

    assert [r["run_id"] for r in bertopic_runs.list_runs(runs=runs)] == [RUN_ID]
    assert [r["run_id"] for r in topic_store.list_runs(runs)] == ["lda-run"]
    record = runs.find_one({"run_id": RUN_ID})
    assert record["method"] == "bertopic" and "warnings" not in record and "assignment" not in record


def test_known_runs_merges_mongo_and_this_computer(bert_dir):
    runs = mongomock.MongoClient().db[topic_store.RUNS_COLLECTION]
    local = _bertopic_run(bert_dir, storage={"url": "s3://b/p/", "bucket": "b", "prefix": f"p/{RUN_ID}/"})
    elsewhere = {**local, "run_id": "20261016T000000Z-other1", "ran_at": "2026-10-16T00:00:00+00:00"}
    elsewhere.pop("storage")
    bertopic_runs.record_run(runs, elsewhere)
    bertopic_runs.record_run(runs, {k: v for k, v in local.items() if k != "storage"})

    known = bertopic_runs.known_runs(runs, bert_dir)
    assert [r["run_id"] for r in known] == ["20261016T000000Z-other1", RUN_ID]
    assert known[1]["storage"]["url"] == "s3://b/p/"  # Mongo missed the upload; the local record keeps it
    assert bertopic_runs.where(known[1], bert_dir) == "on this computer"
    assert bertopic_runs.where(known[0], bert_dir).startswith("in MongoDB only")


# --- download ---------------------------------------------------------------------


def _uploaded_then_seen_elsewhere(bert_dir, tmp_path, env):
    run = _bertopic_run(bert_dir)
    bertopic_runs.upload(run, model_storage.config)
    other = tmp_path / "other_computer"
    env.setenv("BERTOPIC_MODEL_DIR", str(other))
    remote = json.loads(json.dumps(run, default=str))
    remote["model_dir"] = "/Users/someone-else/bertopic_models/" + RUN_ID
    return remote, other


def test_a_run_from_another_computer_is_downloaded_and_verified(s3, bert_dir, tmp_path, env):
    remote, other = _uploaded_then_seen_elsewhere(bert_dir, tmp_path, env)
    assert not bertopic_runs.has_local_files(remote)

    local = bertopic_runs.ensure_local(remote)

    assert local["model_dir"] == str(other.resolve() / RUN_ID)
    assert (other / RUN_ID / bertopic_model.MODEL_FILE).read_bytes() == b"pickled model"
    assert bertopic_runs.has_local_files(local)
    assert bertopic_runs.chart_paths(local)["barchart_html"] == str((other / RUN_ID / "barchart.html").resolve())


def test_a_tampered_upload_is_refused_and_nothing_installed(s3, bert_dir, tmp_path, env):
    remote, other = _uploaded_then_seen_elsewhere(bert_dir, tmp_path, env)
    s3.put_object(Bucket=BUCKET, Key=remote["storage"]["prefix"] + bertopic_model.MODEL_FILE, Body=b"evil pickle")

    with pytest.raises(model_storage.UntrustedRun):
        bertopic_runs.ensure_local(remote)
    assert not (other / RUN_ID).exists()


def test_a_run_that_cant_be_fetched_says_why(bert_dir, tmp_path, env):
    remote = {"run_id": RUN_ID, "model_dir": "/elsewhere", "charts": {}}
    with pytest.raises(FileNotFoundError, match="never uploaded"):
        bertopic_runs.ensure_local(remote)
    with pytest.raises(FileNotFoundError, match="set the model storage settings"):
        bertopic_runs.ensure_local({**remote, "storage": {"url": "s3://b/p/"}}, storage=lambda: None)


# --- training and the CLI -----------------------------------------------------------


def test_train_and_save_uploads_and_records(bert_dir, monkeypatch):
    import numpy as np

    corpus = bertopic_model.Corpus(["101"], ["csv import"], np.zeros((1, 2), np.float32), {"model": "m"})
    monkeypatch.setattr(bertopic_runs.bertopic_model, "load_corpus", lambda tickets: corpus)
    monkeypatch.setattr(bertopic_runs, "tokenize", lambda nlp, sw, texts: [["csv", "import"]])
    monkeypatch.setattr(bertopic_runs.bertopic_model, "train", lambda c, t, s: "result")
    monkeypatch.setattr(bertopic_runs.bertopic_model, "metrics", lambda r, t: {})
    monkeypatch.setattr(bertopic_runs.topic_store, "preprocessing_info", lambda nlp, sw: {})
    monkeypatch.setattr(bertopic_runs.bertopic_model, "save_run", lambda *a, **k: _bertopic_run(bert_dir))
    calls = []
    monkeypatch.setattr(bertopic_runs, "upload", lambda run, storage, status: calls.append("upload"))
    runs = mongomock.MongoClient().db[topic_store.RUNS_COLLECTION]

    run = bertopic_runs.train_and_save(None, None, [], bertopic_runs.TrainOptions(), runs=runs)
    assert calls == ["upload"] and runs.find_one({"run_id": RUN_ID})["method"] == "bertopic"

    class Down:
        def replace_one(self, *a, **k):
            raise ConnectionError("mongo down")

    monkeypatch.setattr(bertopic_runs.bertopic_model, "save_run",
                        lambda *a, **k: _bertopic_run(bert_dir, "20261015T130000Z-bt0002"))
    run = bertopic_runs.train_and_save(None, None, [], bertopic_runs.TrainOptions(), runs=Down())
    assert any("MongoDB could not be updated" in w for w in run["warnings"])


def test_cli_uploads_bertopic_runs(s3, bert_dir, monkeypatch, capsys):
    monkeypatch.setattr("settings.load", lambda: None)
    run = _bertopic_run(bert_dir)

    assert model_storage.main(["upload", "--bertopic"]) == 0
    assert RUN_ID in capsys.readouterr().out
    assert f"topic-models/bertopic/{RUN_ID}/manifest.json" in _keys(s3)
    assert model_storage.main(["upload", "--bertopic"]) == 0
    assert "already uploaded" in capsys.readouterr().out
