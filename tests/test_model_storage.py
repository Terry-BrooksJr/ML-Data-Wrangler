import json
import os

import boto3
import model_storage
import mongomock
import pytest
import topic_store
from moto import mock_aws

BUCKET = "lasso-models"
KEY = "k" * 40


@pytest.fixture
def env(monkeypatch, tmp_path):
    for name in (
        "MODEL_BUCKET",
        "B2_KEY_ID",
        "B2_APPLICATION_KEY",
        "B2_ENDPOINT_URL",
        "MODEL_STORAGE_REGION",
        "MODEL_PREFIX",
        "MONGO_URI",
        "MODEL_SIGNING_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MODEL_BUCKET", BUCKET)
    monkeypatch.setenv("B2_KEY_ID", "key-id")
    monkeypatch.setenv("B2_APPLICATION_KEY", "secret")
    monkeypatch.setenv("MODEL_STORAGE_REGION", "us-east-1")
    monkeypatch.setenv("MODEL_SIGNING_KEY", KEY)
    monkeypatch.setenv("TOPIC_MODEL_DIR", str(tmp_path / "topic_models"))
    return monkeypatch


@pytest.fixture
def s3(env, monkeypatch):
    with mock_aws():
        client = boto3.client(
            "s3",
            region_name="us-east-1",
            aws_access_key_id="key-id",
            aws_secret_access_key="secret",
        )
        client.create_bucket(Bucket=BUCKET)
        monkeypatch.setattr(model_storage, "make_client", lambda cfg: client)
        yield client


def _run(tmp_path, run_id="20261014T120000Z-abc123", folder="topic_models", **extra):
    run_dir = tmp_path / folder / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "lda.model").write_bytes(b"model")
    (run_dir / "lda.model.expElogbeta.npy").write_bytes(b"sidecar")
    (run_dir / "dictionary.dict").write_bytes(b"dict")
    (run_dir / "charts").mkdir()
    (run_dir / "charts" / "lda_model.html").write_text("<html>")
    run = {
        "run_id": run_id,
        "ran_at": "2026-10-14T12:00:00+00:00",
        "model_dir": str(run_dir),
        "warnings": [],
        "charts": {"lda_html": str(run_dir / "charts" / "lda_model.html")},
        **extra,
    }
    topic_store.write_run_file(run)
    return run


def _keys(s3, prefix=""):
    return {
        o["Key"]
        for o in s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix).get("Contents", [])
    }


def _manifest(s3, run):
    body = s3.get_object(Bucket=BUCKET, Key=run["storage"]["prefix"] + "manifest.json")[
        "Body"
    ].read()
    return json.loads(body)


def _as_seen_elsewhere(run):
    """The run's details as another computer gets them from Mongo: no local files."""
    remote = json.loads(json.dumps(run, default=str))
    remote["model_dir"] = "/Users/someone-else/topic_models/" + run["run_id"]
    return remote


# --- config -----------------------------------------------------------------------


def test_uploads_are_off_without_a_bucket(env):
    env.delenv("MODEL_BUCKET")
    assert model_storage.config() is None


def test_bucket_without_keys_is_an_error(env):
    env.delenv("B2_APPLICATION_KEY")
    with pytest.raises(model_storage.StorageNotConfigured, match="key ID and secret"):
        model_storage.config()


@pytest.mark.parametrize("value", [None, "short"])
def test_bucket_needs_a_signing_key(env, value):
    if value is None:
        env.delenv("MODEL_SIGNING_KEY")
    else:
        env.setenv("MODEL_SIGNING_KEY", value)
    with pytest.raises(model_storage.StorageNotConfigured, match="MODEL_SIGNING_KEY"):
        model_storage.config()


@pytest.mark.parametrize(
    "endpoint", ["http://s3.us-west-004.backblazeb2.com", "s3.example.com"]
)
def test_insecure_endpoint_is_an_error(env, endpoint):
    env.setenv("B2_ENDPOINT_URL", endpoint)
    with pytest.raises(model_storage.StorageNotConfigured, match="https://"):
        model_storage.config()


def test_backblaze_region_comes_from_endpoint(env):
    env.delenv("MODEL_STORAGE_REGION")
    env.setenv("B2_ENDPOINT_URL", "https://s3.us-west-004.backblazeb2.com")
    cfg = model_storage.config()
    assert (
        cfg.region == "us-west-004"
        and cfg.endpoint_url == "https://s3.us-west-004.backblazeb2.com"
    )


@pytest.mark.parametrize(
    "prefix, expected", [("", ""), ("models", "models/"), ("/a/b/", "a/b/")]
)
def test_prefix_is_normalized(env, prefix, expected):
    env.setenv("MODEL_PREFIX", prefix)
    assert model_storage.config().prefix == expected


def test_new_key_command(capsys):
    assert model_storage.main(["new-key"]) == 0
    assert len(capsys.readouterr().out.strip()) >= model_storage.MIN_SIGNING_KEY_CHARS


@pytest.mark.parametrize(
    "run_id", ["../evil", "a/b", "..", ".hidden", "", "x" * 101, None, "a\\b"]
)
def test_unsafe_run_ids_are_refused(run_id):
    with pytest.raises(ValueError, match="plain folder name"):
        model_storage.check_run_id(run_id)


# --- upload -----------------------------------------------------------------------


def test_upload_copies_every_file_then_a_signed_manifest(s3, tmp_path):
    run = _run(tmp_path)
    uploaded = []
    real = s3.upload_fileobj

    class Recording:
        def delete_object(self, **kw):
            return s3.delete_object(**kw)

        def upload_fileobj(self, data, bucket, key):
            uploaded.append(key)
            real(data, bucket, key)

    storage = model_storage.upload_run(run, model_storage.config(), Recording())
    prefix = f"topic-models/{run['run_id']}/"
    assert uploaded[-2:] == [prefix + "run.json", prefix + "manifest.json"]
    assert _keys(s3) == {
        prefix + name
        for name in (
            "lda.model",
            "lda.model.expElogbeta.npy",
            "dictionary.dict",
            "charts/lda_model.html",
            "run.json",
            "manifest.json",
        )
    }
    document = json.loads(
        s3.get_object(Bucket=BUCKET, Key=prefix + "manifest.json")["Body"].read()
    )
    assert (
        model_storage.verify(document, run["run_id"], KEY.encode())["files"][
            "lda.model"
        ]["bytes"]
        == 5
    )
    assert storage["region"] == "us-east-1" and storage["files"] == 5


def test_symlinks_and_outside_files_are_not_uploaded(s3, tmp_path):
    run = _run(tmp_path)
    secret = tmp_path / "id_rsa"
    secret.write_text("PRIVATE KEY")
    os.symlink(secret, pathlib_path(run) / "innocent.txt")
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "notes.txt").write_text("private")
    os.symlink(outside, pathlib_path(run) / "linked_dir")

    model_storage.upload_run(run, model_storage.config(), s3)
    keys = _keys(s3)
    assert not any("innocent" in k or "linked_dir" in k for k in keys)
    assert all(
        b"PRIVATE" not in s3.get_object(Bucket=BUCKET, Key=k)["Body"].read()
        for k in keys
    )


def pathlib_path(run):
    import pathlib

    return pathlib.Path(run["model_dir"])


def test_reupload_removes_the_old_manifest_first(s3, tmp_path):
    run = _run(tmp_path)
    model_storage.upload_and_record(run, model_storage.config(), s3)
    (pathlib_path(run) / "lda.model").write_bytes(b"retrained")

    class FailsMidway:
        calls = 0

        def delete_object(self, **kw):
            return s3.delete_object(**kw)

        def upload_fileobj(self, data, bucket, key):
            FailsMidway.calls += 1
            if FailsMidway.calls == 2:
                raise OSError("connection reset")
            s3.upload_fileobj(data, bucket, key)

    with pytest.raises(OSError):
        model_storage.upload_run(run, model_storage.config(), FailsMidway())
    assert run["storage"]["prefix"] + "manifest.json" not in _keys(s3)
    with pytest.raises(FileNotFoundError, match="didn't finish"):
        model_storage.download_run(
            _as_seen_elsewhere(run),
            model_storage.config(),
            tmp_path / "other" / run["run_id"],
        )


def test_upload_and_record_adds_storage_to_run_json(s3, tmp_path):
    run = _run(tmp_path)
    storage = model_storage.upload_and_record(run, model_storage.config(), s3)
    saved = json.loads((pathlib_path(run) / "run.json").read_text())
    assert saved["storage"]["url"] == storage["url"]


def test_missing_run_directory(s3, tmp_path):
    run = {"run_id": "gone", "model_dir": str(tmp_path / "nope")}
    with pytest.raises(FileNotFoundError):
        model_storage.upload_run(run, model_storage.config(), s3)


# --- download ---------------------------------------------------------------------


def test_download_verifies_and_installs(s3, tmp_path):
    run = _run(tmp_path, folder="machine_a")
    model_storage.upload_and_record(run, model_storage.config(), s3)
    dest = tmp_path / "machine_b" / run["run_id"]
    model_storage.download_run(_as_seen_elsewhere(run), model_storage.config(), dest)
    assert (dest / "lda.model").read_bytes() == b"model"
    assert (dest / "charts" / "lda_model.html").read_text() == "<html>"
    assert not list(dest.parent.glob(".*"))  # no staging folder left


def test_wrong_signing_key_is_refused(s3, tmp_path, env):
    run = _run(tmp_path, folder="machine_a")
    model_storage.upload_and_record(run, model_storage.config(), s3)
    env.setenv("MODEL_SIGNING_KEY", "x" * 40)
    dest = tmp_path / "machine_b" / run["run_id"]
    with pytest.raises(model_storage.UntrustedRun, match="signature"):
        model_storage.download_run(
            _as_seen_elsewhere(run), model_storage.config(), dest
        )
    assert not dest.exists()


def test_swapped_model_file_is_refused(s3, tmp_path):
    """Someone with bucket access but not the key replaces the pickle."""
    run = _run(tmp_path, folder="machine_a")
    model_storage.upload_and_record(run, model_storage.config(), s3)
    # Same size as the real file, so only the hash check can catch it.
    s3.put_object(
        Bucket=BUCKET, Key=run["storage"]["prefix"] + "lda.model", Body=b"EVIL!"
    )
    dest = tmp_path / "machine_b" / run["run_id"]
    with pytest.raises(model_storage.UntrustedRun, match="doesn't match its manifest"):
        model_storage.download_run(
            _as_seen_elsewhere(run), model_storage.config(), dest
        )
    assert not dest.exists()


def test_swapped_chart_is_refused(s3, tmp_path):
    run = _run(tmp_path, folder="machine_a")
    model_storage.upload_and_record(run, model_storage.config(), s3)
    s3.put_object(
        Bucket=BUCKET,
        Key=run["storage"]["prefix"] + "charts/lda_model.html",
        Body=b"<script>alert(1)</script>",
    )
    with pytest.raises(model_storage.UntrustedRun):
        model_storage.download_run(
            _as_seen_elsewhere(run),
            model_storage.config(),
            tmp_path / "b" / run["run_id"],
        )


def test_forged_manifest_without_the_key_is_refused(s3, tmp_path):
    run = _run(tmp_path, folder="machine_a")
    model_storage.upload_and_record(run, model_storage.config(), s3)
    document = _manifest(s3, run)
    document["manifest"]["files"]["lda.model"] = {"sha256": "0" * 64, "bytes": 4}
    s3.put_object(
        Bucket=BUCKET,
        Key=run["storage"]["prefix"] + "manifest.json",
        Body=json.dumps(document),
    )
    with pytest.raises(model_storage.UntrustedRun, match="signature"):
        model_storage.download_run(
            _as_seen_elsewhere(run),
            model_storage.config(),
            tmp_path / "b" / run["run_id"],
        )


def test_only_listed_files_are_fetched(s3, tmp_path):
    run = _run(tmp_path, folder="machine_a")
    model_storage.upload_and_record(run, model_storage.config(), s3)
    s3.put_object(
        Bucket=BUCKET, Key=run["storage"]["prefix"] + "huge.bin", Body=b"x" * 10_000
    )
    dest = tmp_path / "b" / run["run_id"]
    model_storage.download_run(_as_seen_elsewhere(run), model_storage.config(), dest)
    assert not (dest / "huge.bin").exists()


def test_prefix_must_be_the_runs_own_folder(s3, tmp_path):
    run = _run(tmp_path, folder="machine_a")
    model_storage.upload_and_record(run, model_storage.config(), s3)
    remote = _as_seen_elsewhere(run)
    remote["storage"]["prefix"] = "topic-models/"  # broad prefix
    with pytest.raises(model_storage.UntrustedRun, match="isn't this run's folder"):
        model_storage.download_run(
            remote, model_storage.config(), tmp_path / "b" / run["run_id"]
        )


def _signed(manifest):
    return {"manifest": manifest, "hmac": model_storage.sign(manifest, KEY.encode())}


@pytest.mark.parametrize(
    "files, problem",
    [
        ({f"f{i}": {"sha256": "0" * 64, "bytes": 1} for i in range(201)}, "201 files"),
        ({"big": {"sha256": "0" * 64, "bytes": 3 * 1024**3}}, "GB"),
        ({"../escape": {"sha256": "0" * 64, "bytes": 1}}, "unsafe path"),
        ({"/abs": {"sha256": "0" * 64, "bytes": 1}}, "unsafe path"),
        ({"a/../b": {"sha256": "0" * 64, "bytes": 1}}, "unsafe path"),
        ({"x": {"sha256": "0" * 64, "bytes": -1}}, "malformed"),
        ({}, "0 files"),
    ],
)
def test_even_signed_manifests_are_bounded(files, problem):
    manifest = {"version": 1, "run_id": "r1", "files": files}
    with pytest.raises(model_storage.UntrustedRun, match=problem):
        model_storage.verify(_signed(manifest), "r1", KEY.encode())


def test_manifest_for_another_run_is_refused():
    manifest = {
        "version": 1,
        "run_id": "other",
        "files": {"a": {"sha256": "0" * 64, "bytes": 1}},
    }
    with pytest.raises(model_storage.UntrustedRun, match="different run"):
        model_storage.verify(_signed(manifest), "r1", KEY.encode())


def test_oversized_object_stops_streaming(s3, tmp_path):
    run = _run(tmp_path, folder="machine_a")
    model_storage.upload_and_record(run, model_storage.config(), s3)
    s3.put_object(
        Bucket=BUCKET, Key=run["storage"]["prefix"] + "lda.model", Body=b"x" * 5_000_000
    )
    with pytest.raises(model_storage.UntrustedRun, match="larger than its manifest"):
        model_storage.download_run(
            _as_seen_elsewhere(run),
            model_storage.config(),
            tmp_path / "b" / run["run_id"],
        )
    assert not list((tmp_path / "b").glob("**/*.model"))


def test_download_uses_the_endpoint_recorded_with_the_run(env, tmp_path, monkeypatch):
    made = []
    monkeypatch.setattr(
        model_storage, "make_client", lambda cfg: made.append(cfg) or object()
    )
    storage = {
        "bucket": "b",
        "prefix": "topic-models/r1/",
        "endpoint_url": "https://s3.eu-central-003.backblazeb2.com",
    }
    model_storage._client_for(storage, model_storage.config())
    assert (
        made[0].endpoint_url == storage["endpoint_url"]
        and made[0].region == "eu-central-003"
    )

    with pytest.raises(model_storage.UntrustedRun, match="isn't https"):
        model_storage._client_for(
            {**storage, "endpoint_url": "http://evil.example"}, model_storage.config()
        )


def test_concurrent_download_reuses_the_winners_copy(s3, tmp_path, monkeypatch):
    run = _run(tmp_path, folder="machine_a")
    model_storage.upload_and_record(run, model_storage.config(), s3)
    remote = _as_seen_elsewhere(run)
    dest = tmp_path / "machine_b" / run["run_id"]
    real_install = model_storage._install

    def other_window_wins(staging, dest_, manifest):
        # Another app window installs its own verified copy just before us.
        monkeypatch.setattr(model_storage, "_install", real_install)
        model_storage.download_run(dict(remote), model_storage.config(), dest_)
        real_install(staging, dest_, manifest)

    monkeypatch.setattr(model_storage, "_install", other_window_wins)
    assert model_storage.download_run(remote, model_storage.config(), dest) == dest
    assert (dest / "lda.model").read_bytes() == b"model"


def test_incomplete_local_copy_is_moved_aside(s3, tmp_path):
    run = _run(tmp_path, folder="machine_a")
    model_storage.upload_and_record(run, model_storage.config(), s3)
    dest = tmp_path / "machine_b" / run["run_id"]
    dest.mkdir(parents=True)
    (dest / "lda.model").write_bytes(b"model")  # no dictionary
    model_storage.download_run(_as_seen_elsewhere(run), model_storage.config(), dest)
    assert (dest / "dictionary.dict").read_bytes() == b"dict"
    assert len(list(dest.parent.glob(f".{run['run_id']}.incomplete-*"))) == 1


# --- CLI --------------------------------------------------------------------------


def test_cli_uploads_runs_not_yet_uploaded_and_records_them(
    s3, tmp_path, env, monkeypatch, capsys
):
    monkeypatch.setattr("settings.load", lambda: None)
    env.setenv("MONGO_URI", "mongodb://test")
    runs = mongomock.MongoClient().db[topic_store.RUNS_COLLECTION]
    import decision_engine.persistence

    monkeypatch.setattr(
        decision_engine.persistence,
        "get_database",
        lambda: {topic_store.RUNS_COLLECTION: runs},
    )
    _run(tmp_path, "20261001T000000Z-old111", storage={"url": "s3://elsewhere/"})
    new = _run(tmp_path, "20261014T000000Z-new222")

    assert model_storage.main(["upload"]) == 0
    out = capsys.readouterr().out
    assert new["run_id"] in out and "old111" not in out
    prefix = f"topic-models/{new['run_id']}/"
    assert {
        prefix + "lda.model",
        prefix + "dictionary.dict",
        prefix + "run.json",
        prefix + "manifest.json",
    } <= _keys(s3)
    # Not in Mongo before: now it is, with its upload, so other computers find it.
    record = runs.find_one({"run_id": new["run_id"]})
    assert (
        record["storage"]["prefix"] == prefix
        and record["model_dir"] == new["model_dir"]
    )

    assert model_storage.main(["upload"]) == 0
    assert "Every local run is already uploaded." in capsys.readouterr().out


def test_cli_keeps_an_existing_mongo_record(s3, tmp_path, monkeypatch):
    runs = mongomock.MongoClient().db.runs
    run = _run(tmp_path)
    runs.insert_one({"run_id": run["run_id"], "assignment": {"assigned": 7}})
    model_storage.record_storage(runs, run, {"url": "s3://b/p/"})
    record = runs.find_one({"run_id": run["run_id"]})
    assert record["assignment"] == {"assigned": 7} and record["storage"] == {
        "url": "s3://b/p/"
    }


def test_cli_needs_a_bucket(env, monkeypatch, capsys):
    env.delenv("MODEL_BUCKET")
    monkeypatch.setattr("settings.load", lambda: None)
    assert model_storage.main(["upload"]) == 2
    assert "MODEL_BUCKET is not set" in capsys.readouterr().err


# --- a downloaded run's local details ------------------------------------------------


def test_downloaded_run_json_is_rewritten_for_this_computer(
    s3, tmp_path, env, monkeypatch, capsys
):
    run = _run(tmp_path, folder="machine_a")
    model_storage.upload_and_record(run, model_storage.config(), s3)
    machine_b = tmp_path / "machine_b"
    env.setenv("TOPIC_MODEL_DIR", str(machine_b))

    topic_store.ensure_local(_as_seen_elsewhere(run))

    local = (machine_b / run["run_id"]).resolve()
    saved = json.loads((local / "run.json").read_text())
    assert saved["model_dir"] == str(local)
    assert saved["storage"]["prefix"] == run["storage"]["prefix"]
    assert saved["charts"]["lda_html"] == str(local / "charts" / "lda_model.html")

    # The sequence from the review: a default upload must not pick the downloaded run.
    monkeypatch.setattr("settings.load", lambda: None)
    assert model_storage.main(["upload"]) == 0
    assert "Every local run is already uploaded." in capsys.readouterr().out
    [listed] = topic_store.list_runs()
    assert listed["model_dir"] == str(local) and listed["storage"]


def test_rewritten_run_json_still_counts_as_a_complete_download(s3, tmp_path, env):
    run = _run(tmp_path, folder="machine_a")
    model_storage.upload_and_record(run, model_storage.config(), s3)
    env.setenv("TOPIC_MODEL_DIR", str(tmp_path / "machine_b"))
    topic_store.ensure_local(_as_seen_elsewhere(run))
    local = tmp_path / "machine_b" / run["run_id"]

    # A second window downloading the same run reuses this copy rather than moving it aside.
    model_storage.download_run(_as_seen_elsewhere(run), model_storage.config(), local)
    assert not list(local.parent.glob(f".{run['run_id']}.incomplete-*"))
    assert json.loads((local / "run.json").read_text())["storage"]


def test_unchanged_local_run_json_is_not_rewritten(s3, tmp_path, env):
    run = _run(tmp_path)
    model_storage.upload_and_record(run, model_storage.config(), s3)
    path = pathlib_path(run) / "run.json"
    before = path.stat().st_mtime_ns
    topic_store.ensure_local(json.loads(path.read_text()))
    assert path.stat().st_mtime_ns == before
