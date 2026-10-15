import json

import boto3
import pytest
from moto import mock_aws

import model_storage
import topic_store

BUCKET = "lasso-models"


@pytest.fixture
def env(monkeypatch):
    for name in ("MODEL_BUCKET", "B2_KEY_ID", "B2_APPLICATION_KEY", "B2_ENDPOINT_URL",
                 "MODEL_STORAGE_REGION", "MODEL_PREFIX", "MONGO_URI"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MODEL_BUCKET", BUCKET)
    monkeypatch.setenv("B2_KEY_ID", "key-id")
    monkeypatch.setenv("B2_APPLICATION_KEY", "secret")
    monkeypatch.setenv("MODEL_STORAGE_REGION", "us-east-1")
    return monkeypatch


@pytest.fixture
def s3(env):
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1", aws_access_key_id="key-id",
                              aws_secret_access_key="secret")
        client.create_bucket(Bucket=BUCKET)
        yield client


def _run(tmp_path, run_id="20261014T120000Z-abc123", **extra):
    run_dir = tmp_path / "topic_models" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "lda.model").write_bytes(b"model")
    (run_dir / "lda.model.expElogbeta.npy").write_bytes(b"sidecar")
    (run_dir / "dictionary.dict").write_bytes(b"dict")
    (run_dir / "charts").mkdir()
    (run_dir / "charts" / "lda_model.html").write_text("<html>")
    run = {"run_id": run_id, "ran_at": "2026-10-14T12:00:00+00:00", "model_dir": str(run_dir), "warnings": [], **extra}
    topic_store.write_run_file(run)
    return run


# --- config -----------------------------------------------------------------------


def test_uploads_are_off_without_a_bucket(env):
    env.delenv("MODEL_BUCKET")
    assert model_storage.config() is None


def test_bucket_without_keys_is_an_error(env):
    env.delenv("B2_APPLICATION_KEY")
    with pytest.raises(model_storage.StorageNotConfigured, match="key ID and secret"):
        model_storage.config()


@pytest.mark.parametrize("endpoint", ["http://s3.us-west-004.backblazeb2.com", "s3.example.com"])
def test_insecure_endpoint_is_an_error(env, endpoint):
    env.setenv("B2_ENDPOINT_URL", endpoint)
    with pytest.raises(model_storage.StorageNotConfigured, match="https://"):
        model_storage.config()


def test_backblaze_region_comes_from_endpoint(env):
    env.delenv("MODEL_STORAGE_REGION")
    env.setenv("B2_ENDPOINT_URL", "https://s3.us-west-004.backblazeb2.com")
    cfg = model_storage.config()
    assert cfg.region == "us-west-004" and cfg.endpoint_url == "https://s3.us-west-004.backblazeb2.com"


@pytest.mark.parametrize("prefix, expected", [("", ""), ("models", "models/"), ("/a/b/", "a/b/")])
def test_prefix_is_normalized(env, prefix, expected):
    env.setenv("MODEL_PREFIX", prefix)
    assert model_storage.config().prefix == expected


def test_default_prefix(env):
    assert model_storage.config().prefix == "topic-models/"


# --- upload -----------------------------------------------------------------------


def test_upload_run_copies_every_file_with_run_json_last(s3, tmp_path):
    run = _run(tmp_path)
    cfg = model_storage.config()
    uploaded = []

    class Recording:
        def upload_file(self, path, bucket, key):
            uploaded.append(key)
            s3.upload_file(path, bucket, key)

    storage = model_storage.upload_run(run, cfg, Recording())
    prefix = f"topic-models/{run['run_id']}/"
    assert uploaded[-1] == prefix + "run.json"
    keys = {o["Key"] for o in s3.list_objects_v2(Bucket=BUCKET)["Contents"]}
    assert keys == {prefix + name for name in
                    ("lda.model", "lda.model.expElogbeta.npy", "dictionary.dict", "charts/lda_model.html", "run.json")}
    assert s3.get_object(Bucket=BUCKET, Key=prefix + "lda.model")["Body"].read() == b"model"
    assert storage["url"] == f"s3://{BUCKET}/{prefix}" and storage["files"] == 5


def test_upload_and_record_adds_storage_to_run_json(s3, tmp_path):
    run = _run(tmp_path)
    storage = model_storage.upload_and_record(run, model_storage.config(), s3)
    assert run["storage"] is storage
    saved = json.loads((tmp_path / "topic_models" / run["run_id"] / "run.json").read_text())
    assert saved["storage"]["url"] == storage["url"]


def test_missing_run_directory(s3, tmp_path):
    run = {"run_id": "gone", "model_dir": str(tmp_path / "nope")}
    with pytest.raises(FileNotFoundError):
        model_storage.upload_run(run, model_storage.config(), s3)


def test_failed_upload_leaves_no_run_json(s3, tmp_path):
    run = _run(tmp_path)
    calls = []

    class FailsSecond:
        def upload_file(self, path, bucket, key):
            calls.append(key)
            if len(calls) == 2:
                raise OSError("connection reset")
            s3.upload_file(path, bucket, key)

    with pytest.raises(OSError):
        model_storage.upload_run(run, model_storage.config(), FailsSecond())
    keys = {o["Key"] for o in s3.list_objects_v2(Bucket=BUCKET).get("Contents", [])}
    assert not any(k.endswith("run.json") for k in keys)


# --- CLI --------------------------------------------------------------------------


def test_cli_uploads_only_runs_not_yet_uploaded(s3, tmp_path, env, monkeypatch, capsys):
    env.setenv("TOPIC_MODEL_DIR", str(tmp_path / "topic_models"))
    monkeypatch.setattr("settings.load", lambda: None)
    monkeypatch.setattr(model_storage, "make_client", lambda cfg: s3)
    _run(tmp_path, "20261001T000000Z-old111", storage={"url": "s3://elsewhere/"})
    new = _run(tmp_path, "20261014T000000Z-new222")

    assert model_storage.main(["upload"]) == 0
    out = capsys.readouterr().out
    assert new["run_id"] in out and "old111" not in out
    assert model_storage.main(["upload"]) == 0
    assert "Every local run is already uploaded." in capsys.readouterr().out


def test_cli_needs_a_bucket(env, monkeypatch, capsys):
    env.delenv("MODEL_BUCKET")
    monkeypatch.setattr("settings.load", lambda: None)
    assert model_storage.main(["upload"]) == 2
    assert "MODEL_BUCKET is not set" in capsys.readouterr().err
