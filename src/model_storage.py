"""Uploads saved topic-model runs to S3-compatible storage (AWS S3 or Backblaze B2).

A run's directory (model, dictionary, gensim's sidecar files, charts,
run.json) is copied to s3://MODEL_BUCKET/MODEL_PREFIX<run_id>/, followed by a
signed manifest.json listing every file's SHA-256 and size. The manifest goes
last, so a prefix with a valid manifest holds the whole run. The upload
location is added to the run as `storage` and to the local run.json, so it
reaches Mongo with the rest of the run's details.

download_run() brings an uploaded run back when its files aren't on this
computer. Model files are pickles (gensim), which run code when loaded, so a
download is trusted only if its manifest's HMAC verifies with
MODEL_SIGNING_KEY: a key every computer sharing models holds, and that
someone able to edit MongoDB or the bucket doesn't. Only the files the
manifest lists are fetched, each checked against its size and hash before the
run is moved into place.

Settings (see settings.SETTINGS): MODEL_BUCKET turns this on; B2_KEY_ID,
B2_APPLICATION_KEY, MODEL_SIGNING_KEY, and B2_ENDPOINT_URL (blank for AWS S3);
optional MODEL_PREFIX and MODEL_STORAGE_REGION.

    cd src && python -m model_storage upload [RUN_ID ...]   # default: local runs not yet uploaded
    cd src && python -m model_storage upload --bertopic [RUN_ID ...]   # BERTopic runs (bertopic_runs)
    cd src && python -m model_storage new-key               # prints a new MODEL_SIGNING_KEY
"""
import argparse
import dataclasses
import hashlib
import hmac
import io
import json
import os
import pathlib
import re
import secrets
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

from loguru import logger

from decision_engine.classify import is_loopback_url

DEFAULT_PREFIX = "topic-models/"
RUN_FILE = "run.json"  # same as topic_store.RUN_FILE
MANIFEST_FILE = "manifest.json"  # uploaded last: its presence marks a complete upload
MANIFEST_VERSION = 1
# A run is a model, a dictionary, a few gensim sidecar files and two charts.
MAX_RUN_FILES = 200
MAX_RUN_BYTES = 2 * 1024 ** 3
MAX_MANIFEST_BYTES = 1024 * 1024
MIN_SIGNING_KEY_CHARS = 32
CHUNK_BYTES = 1024 * 1024
# Any single safe path component (topic_store.new_run_id's format fits): a run
# ID names a local folder and an S3 prefix.
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
# s3.<region>.backblazeb2.com: B2's S3 endpoints carry their region.
_B2_ENDPOINT = re.compile(r"^https://s3\.([a-z0-9-]+)\.backblazeb2\.com/?$")


class StorageNotConfigured(RuntimeError):
    """MODEL_BUCKET is set but the other storage settings are missing or invalid."""


class UntrustedRun(ValueError):
    """An upload failed verification: unsigned, wrongly signed, or not what its manifest lists."""


@dataclass(frozen=True)
class Config:
    bucket: str
    prefix: str
    key_id: str
    secret: str
    endpoint_url: Optional[str]
    region: Optional[str]
    signing_key: bytes = b""


def check_run_id(run_id) -> str:
    """run_id if it is a single safe path component.

    Raises:
        ValueError: it could escape the model folder or the run's S3 prefix.
    """
    if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id) or ".." in run_id:
        raise ValueError(f"Refusing run ID {run_id!r}: it isn't a plain folder name.")
    return run_id


def _endpoint_ok(endpoint: Optional[str]) -> bool:
    if not endpoint:
        return True
    scheme = endpoint.split(":", 1)[0].lower()
    return scheme == "https" or (scheme == "http" and is_loopback_url(endpoint))


def _region_for(endpoint: Optional[str]) -> Optional[str]:
    match = _B2_ENDPOINT.match(endpoint or "")
    return match.group(1) if match else None


def config() -> Optional[Config]:
    """The storage settings, or None when uploads are off (MODEL_BUCKET unset).

    Raises:
        StorageNotConfigured: the bucket is set but the keys or the signing
            key are missing, or the endpoint isn't https:// (the keys sign
            requests to it).
    """
    bucket = os.environ.get("MODEL_BUCKET", "").strip()
    if not bucket:
        return None
    key_id = os.environ.get("B2_KEY_ID", "").strip()
    secret = os.environ.get("B2_APPLICATION_KEY", "").strip()
    if not key_id or not secret:
        raise StorageNotConfigured("MODEL_BUCKET is set, but the storage key ID and secret aren't. Add them in Settings.")
    signing_key = os.environ.get("MODEL_SIGNING_KEY", "").strip()
    if len(signing_key) < MIN_SIGNING_KEY_CHARS:
        raise StorageNotConfigured(
            f"MODEL_SIGNING_KEY must be at least {MIN_SIGNING_KEY_CHARS} characters, and the same on every computer "
            "that shares models. Generate one with: python -m model_storage new-key"
        )
    endpoint = os.environ.get("B2_ENDPOINT_URL", "").strip() or None
    if not _endpoint_ok(endpoint):
        raise StorageNotConfigured(f"B2_ENDPOINT_URL must be https:// (http:// only for localhost), got {endpoint}.")
    region = os.environ.get("MODEL_STORAGE_REGION", "").strip() or _region_for(endpoint)
    prefix = os.environ.get("MODEL_PREFIX", DEFAULT_PREFIX).strip().lstrip("/")
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    return Config(bucket=bucket, prefix=prefix, key_id=key_id, secret=secret, endpoint_url=endpoint, region=region,
                  signing_key=signing_key.encode("utf-8"))


def make_client(cfg: Config):
    import boto3
    from botocore.config import Config as BotoConfig

    return boto3.client(
        "s3",
        endpoint_url=cfg.endpoint_url,
        region_name=cfg.region,
        aws_access_key_id=cfg.key_id,
        aws_secret_access_key=cfg.secret,
        config=BotoConfig(retries={"max_attempts": 5, "mode": "standard"}),
    )


# --- manifest ---------------------------------------------------------------------


def _canonical(manifest: dict) -> bytes:
    return json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sign(manifest: dict, key: bytes) -> str:
    return hmac.new(key, _canonical(manifest), hashlib.sha256).hexdigest()


def _check_relative(relative) -> None:
    parts = relative.split("/") if isinstance(relative, str) else []
    if not parts or relative.startswith("/") or any(p in ("", ".", "..") or "\\" in p for p in parts):
        raise UntrustedRun(f"its manifest names an unsafe path {relative!r}")


def verify(document: dict, run_id: str, key: bytes) -> dict:
    """The manifest inside a signed manifest.json document, if it checks out.

    Raises:
        UntrustedRun: the signature, run ID, file count, total size or a path is wrong.
    """
    manifest, signature = document.get("manifest"), document.get("hmac")
    if not isinstance(manifest, dict) or not isinstance(signature, str):
        raise UntrustedRun("the upload has no signed manifest")
    if not hmac.compare_digest(sign(manifest, key), signature):
        raise UntrustedRun("its manifest signature doesn't match MODEL_SIGNING_KEY (a different key, or tampered with)")
    if manifest.get("version") != MANIFEST_VERSION or manifest.get("run_id") != run_id:
        raise UntrustedRun("its manifest is for a different run")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files or len(files) > MAX_RUN_FILES:
        raise UntrustedRun(f"its manifest lists {len(files) if isinstance(files, dict) else 0} files "
                           f"(1-{MAX_RUN_FILES} allowed)")
    total = 0
    for relative, entry in files.items():
        _check_relative(relative)
        size = entry.get("bytes") if isinstance(entry, dict) else None
        if not isinstance(size, int) or isinstance(size, bool) or size < 0 or not isinstance(entry.get("sha256"), str):
            raise UntrustedRun(f"its manifest entry for {relative!r} is malformed")
        total += size
    if total > MAX_RUN_BYTES:
        raise UntrustedRun(f"it is {total / 1e9:.1f} GB (limit {MAX_RUN_BYTES / 1e9:.0f} GB)")
    return manifest


# --- upload -----------------------------------------------------------------------


def run_files(run_dir: pathlib.Path) -> List[pathlib.Path]:
    """Every regular file in the run, inside it, run.json last.

    Symlinks (and anything under a symlinked folder) are skipped, and so is
    anything whose real path is outside the run's folder, so a planted link
    can't upload another file's contents.
    """
    root = run_dir.resolve()
    files = []
    for path in sorted(run_dir.rglob("*")):
        relative = path.relative_to(run_dir)
        if any((run_dir / p).is_symlink() for p in [relative, *list(relative.parents)[:-1]]):
            logger.warning(f"Not uploading {path}: it is or is under a symlink.")
            continue
        if not path.is_file() or path.name == MANIFEST_FILE:
            continue
        if root not in path.resolve().parents:
            logger.warning(f"Not uploading {path}: it is outside {root}.")
            continue
        files.append(path)
    return [p for p in files if p.name != RUN_FILE] + [p for p in files if p.name == RUN_FILE]


def upload_run(run: dict, cfg: Config, client=None) -> dict:
    """Uploads a run's directory and returns its `storage` entry.

    A previous upload's manifest is deleted first, and the new one is written
    only after every file, so an interrupted re-upload never looks complete.
    Each file is read once, and the bytes uploaded are the bytes hashed.

    Raises:
        FileNotFoundError: the run's directory is missing.
        botocore errors: the upload failed (it then has no manifest).
    """
    run_id = check_run_id(run["run_id"])
    run_dir = pathlib.Path(run["model_dir"])
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory {run_dir} doesn't exist.")
    client = client or make_client(cfg)
    prefix = f"{cfg.prefix}{run_id}/"
    client.delete_object(Bucket=cfg.bucket, Key=prefix + MANIFEST_FILE)
    files: Dict[str, dict] = {}
    for path in run_files(run_dir):
        relative = path.relative_to(run_dir).as_posix()
        data = path.read_bytes()
        client.upload_fileobj(io.BytesIO(data), cfg.bucket, prefix + relative)
        files[relative] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    manifest = {"version": MANIFEST_VERSION, "run_id": run_id, "files": files}
    document = json.dumps({"manifest": manifest, "hmac": sign(manifest, cfg.signing_key)}).encode("utf-8")
    client.upload_fileobj(io.BytesIO(document), cfg.bucket, prefix + MANIFEST_FILE)
    total = sum(entry["bytes"] for entry in files.values())
    location = {
        "url": f"s3://{cfg.bucket}/{prefix}",
        "bucket": cfg.bucket,
        "prefix": prefix,
        "endpoint_url": cfg.endpoint_url,
        "region": cfg.region,
        "files": len(files),
        "bytes": total,
        "uploaded_at": datetime.now(timezone.utc),
    }
    logger.success(f"Uploaded run {run_id} ({len(files)} files, {total / 1e6:.1f} MB) to {location['url']}")
    return location


def upload_and_record(run: dict, cfg: Config, client=None) -> dict:
    """Uploads the run, then adds `storage` to the run and its local run.json."""
    import topic_store

    run["storage"] = upload_run(run, cfg, client)
    topic_store.write_run_file(run)
    return run["storage"]


# --- download ---------------------------------------------------------------------


def _client_for(storage: dict, cfg: Config):
    """A client for where the run was uploaded, which may differ from the current settings.

    Raises:
        UntrustedRun: the recorded endpoint isn't https:// (our keys sign the request).
    """
    endpoint = storage.get("endpoint_url", cfg.endpoint_url)
    if not _endpoint_ok(endpoint):
        raise UntrustedRun(f"its recorded endpoint {endpoint} isn't https://")
    if endpoint == cfg.endpoint_url:
        region = storage.get("region") or cfg.region
    else:
        region = storage.get("region") or _region_for(endpoint)
    return make_client(dataclasses.replace(cfg, endpoint_url=endpoint, region=region))


def _fetch(client, bucket: str, key: str, target: pathlib.Path, expected: dict) -> None:
    """Streams an object to target, stopping past its expected size, then checks its hash."""
    body = client.get_object(Bucket=bucket, Key=key)["Body"]
    digest, written = hashlib.sha256(), 0
    with open(target, "wb") as f:
        while True:
            chunk = body.read(CHUNK_BYTES)
            if not chunk:
                break
            written += len(chunk)
            if written > expected["bytes"]:
                raise UntrustedRun(f"s3://{bucket}/{key} is larger than its manifest says")
            digest.update(chunk)
            f.write(chunk)
    if written != expected["bytes"] or not hmac.compare_digest(digest.hexdigest(), expected["sha256"]):
        raise UntrustedRun(f"s3://{bucket}/{key} doesn't match its manifest")


def _is_complete(folder: pathlib.Path, manifest: dict) -> bool:
    """True if folder holds every file the manifest lists, with matching sizes and hashes.

    run.json is skipped: it is verified when downloaded, then rewritten
    locally with the run's local paths and upload (topic_store.ensure_local).
    """
    for relative, entry in manifest["files"].items():
        if relative == RUN_FILE:
            continue
        path = folder / relative
        if path.is_symlink() or not path.is_file() or path.stat().st_size != entry["bytes"]:
            return False
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            return False
    return True


def _read_manifest(client, bucket: str, prefix: str) -> bytes:
    try:
        body = client.get_object(Bucket=bucket, Key=prefix + MANIFEST_FILE)["Body"].read(MAX_MANIFEST_BYTES + 1)
    except Exception as e:
        code = getattr(e, "response", {}).get("Error", {}).get("Code") if hasattr(e, "response") else None
        if code in ("NoSuchKey", "404"):
            raise FileNotFoundError(f"s3://{bucket}/{prefix} has no {MANIFEST_FILE}; the upload didn't finish.") from None
        raise
    if len(body) > MAX_MANIFEST_BYTES:
        raise UntrustedRun(f"its {MANIFEST_FILE} is over {MAX_MANIFEST_BYTES // 1024} KB")
    return body


def download_run(run: dict, cfg: Config, dest: pathlib.Path, client=None) -> pathlib.Path:
    """Downloads and verifies an uploaded run into dest.

    The run's recorded prefix must end with its run ID. Its manifest must be
    signed with MODEL_SIGNING_KEY and list at most MAX_RUN_FILES files totalling
    at most MAX_RUN_BYTES; every listed file must match its size and hash, and
    nothing else is fetched. Files go to a staging folder next to dest, which
    replaces dest only when complete. An incomplete dest is moved aside, not
    deleted; if another download finished first, its verified copy is used.

    Raises:
        FileNotFoundError: the run was never uploaded, or its upload has no
            manifest (it didn't finish).
        UntrustedRun: the upload failed verification. Nothing is installed.
        ValueError: the run ID isn't a plain folder name.
    """
    run_id = check_run_id(run["run_id"])
    storage = run.get("storage") or {}
    bucket, prefix = storage.get("bucket"), storage.get("prefix")
    if not bucket or not isinstance(prefix, str):
        raise FileNotFoundError(f"Run {run_id} was never uploaded.")
    if not prefix.endswith(f"{run_id}/") or ".." in prefix.split("/"):
        raise UntrustedRun(f"its recorded location s3://{bucket}/{prefix} isn't this run's folder")
    client = client or _client_for(storage, cfg)
    try:
        document = json.loads(_read_manifest(client, bucket, prefix))
    except ValueError:
        raise UntrustedRun(f"its {MANIFEST_FILE} isn't valid JSON") from None
    manifest = verify(document if isinstance(document, dict) else {}, run_id, cfg.signing_key)

    if dest.exists() and _is_complete(dest, manifest):
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    staging = pathlib.Path(tempfile.mkdtemp(dir=dest.parent, prefix=f".{dest.name}."))
    try:
        for relative, entry in manifest["files"].items():
            target = (staging / relative).resolve()
            if staging.resolve() not in target.parents:
                raise UntrustedRun(f"its manifest names a path outside the run's folder: {relative!r}")
            target.parent.mkdir(parents=True, exist_ok=True)
            _fetch(client, bucket, prefix + relative, target, entry)
        _install(staging, dest, manifest)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    logger.success(f"Downloaded and verified run {run_id} from s3://{bucket}/{prefix} to {dest}")
    return dest


def _install(staging: pathlib.Path, dest: pathlib.Path, manifest: dict) -> None:
    """Moves a verified staging folder to dest."""
    if dest.exists() and not _is_complete(dest, manifest):
        aside = dest.with_name(f".{dest.name}.incomplete-{datetime.now():%Y%m%d-%H%M%S}")
        dest.rename(aside)
        logger.warning(f"Moved the incomplete {dest} aside to {aside}.")
    try:
        staging.rename(dest)
    except OSError:
        # Another download (a second app window) finished first.
        if dest.exists() and _is_complete(dest, manifest):
            shutil.rmtree(staging, ignore_errors=True)
            return
        raise


# --- CLI --------------------------------------------------------------------------


def record_storage(runs_collection, run: dict, storage: dict) -> None:
    """Adds the upload to the run's Mongo record, creating the record if it has none."""
    details = {k: v for k, v in run.items() if k not in ("_id", "storage")}
    runs_collection.update_one({"run_id": run["run_id"]},
                               {"$set": {"storage": storage}, "$setOnInsert": details}, upsert=True)


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m model_storage", description="Upload topic-model runs to S3.")
    commands = parser.add_subparsers(dest="command", required=True)
    upload = commands.add_parser("upload", help="upload saved runs (default: local runs not uploaded yet)")
    upload.add_argument("run_ids", nargs="*")
    upload.add_argument("--bertopic", action="store_true",
                        help="upload BERTopic runs (BERTOPIC_MODEL_DIR) instead of LDA runs")
    commands.add_parser("new-key", help="print a new random MODEL_SIGNING_KEY")
    args = parser.parse_args(argv)
    if args.command == "new-key":
        print(secrets.token_urlsafe(48))
        return 0

    import settings

    settings.load()
    import topic_store

    try:
        cfg = config()
    except StorageNotConfigured as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if cfg is None:
        print("error: MODEL_BUCKET is not set. Set the bucket in Settings.", file=sys.stderr)
        return 2
    if args.bertopic:
        import bertopic_model
        import bertopic_runs

        runs, run_dir, cfg = bertopic_runs.list_runs(), bertopic_model.model_dir(), bertopic_runs.storage_config(cfg)
    else:
        runs, run_dir = topic_store.list_runs(), topic_store.model_dir()
    if args.run_ids:
        by_id = {run["run_id"]: run for run in runs}
        missing = [r for r in args.run_ids if r not in by_id]
        if missing:
            print(f"error: no local run {', '.join(missing)} under {run_dir}", file=sys.stderr)
            return 2
        chosen = [by_id[r] for r in args.run_ids]
    else:
        chosen = [run for run in runs if not run.get("storage")]
    if not chosen:
        print("Every local run is already uploaded.")
        return 0
    client = make_client(cfg)
    runs_collection = None
    if os.environ.get("MONGO_URI"):
        from decision_engine.persistence import get_database

        runs_collection = get_database()[topic_store.RUNS_COLLECTION]
    for run in chosen:
        storage = upload_and_record(run, cfg, client)
        if runs_collection is not None:
            record_storage(runs_collection, run, storage)
        print(f"{run['run_id']}: {storage['url']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
