"""Uploads saved topic-model runs to S3-compatible storage (AWS S3 or Backblaze B2).

A run's directory (model, dictionary, gensim's sidecar files, charts,
run.json) is copied to s3://MODEL_BUCKET/MODEL_PREFIX<run_id>/. run.json goes
last, so a prefix holding run.json has the whole run. The upload location is
added to the run as `storage` and to the local run.json, so it reaches
Mongo with the rest of the run's details. download_run() brings an uploaded
run back when its files aren't on this computer (trained elsewhere, or the
local folder was cleared).

Settings (see settings.SETTINGS): MODEL_BUCKET turns this on; B2_KEY_ID,
B2_APPLICATION_KEY, and B2_ENDPOINT_URL (blank for AWS S3); optional
MODEL_PREFIX and MODEL_STORAGE_REGION.

    cd src && python -m model_storage upload [RUN_ID ...]   # default: local runs not yet uploaded
"""
import argparse
import os
import pathlib
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

from loguru import logger

from decision_engine.classify import is_loopback_url

DEFAULT_PREFIX = "topic-models/"
RUN_FILE = "run.json"  # same as topic_store.RUN_FILE; uploaded last
# s3.<region>.backblazeb2.com: B2's S3 endpoints carry their region.
_B2_ENDPOINT = re.compile(r"^https://s3\.([a-z0-9-]+)\.backblazeb2\.com/?$")


class StorageNotConfigured(RuntimeError):
    """MODEL_BUCKET is set but the other storage settings are missing or invalid."""


@dataclass(frozen=True)
class Config:
    bucket: str
    prefix: str
    key_id: str
    secret: str
    endpoint_url: Optional[str]
    region: Optional[str]


def config() -> Optional[Config]:
    """The storage settings, or None when uploads are off (MODEL_BUCKET unset).

    Raises:
        StorageNotConfigured: the bucket is set but the keys are missing, or
            the endpoint isn't https:// (the keys are sent to it).
    """
    bucket = os.environ.get("MODEL_BUCKET", "").strip()
    if not bucket:
        return None
    key_id = os.environ.get("B2_KEY_ID", "").strip()
    secret = os.environ.get("B2_APPLICATION_KEY", "").strip()
    if not key_id or not secret:
        raise StorageNotConfigured("MODEL_BUCKET is set, but the storage key ID and secret aren't. Add them in Settings.")
    endpoint = os.environ.get("B2_ENDPOINT_URL", "").strip() or None
    if endpoint and not (endpoint.startswith("https://") or (endpoint.startswith("http://") and is_loopback_url(endpoint))):
        raise StorageNotConfigured(f"B2_ENDPOINT_URL must be https:// (http:// only for localhost), got {endpoint}.")
    region = os.environ.get("MODEL_STORAGE_REGION", "").strip() or None
    if not region and endpoint:
        match = _B2_ENDPOINT.match(endpoint)
        region = match.group(1) if match else None
    prefix = os.environ.get("MODEL_PREFIX", DEFAULT_PREFIX).strip().lstrip("/")
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    return Config(bucket=bucket, prefix=prefix, key_id=key_id, secret=secret, endpoint_url=endpoint, region=region)


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


def run_files(run_dir: pathlib.Path) -> List[pathlib.Path]:
    """Every file in the run, run.json last."""
    files = sorted(p for p in run_dir.rglob("*") if p.is_file())
    return [p for p in files if p.name != RUN_FILE] + [p for p in files if p.name == RUN_FILE]


def upload_run(run: dict, cfg: Config, client=None) -> dict:
    """Uploads a run's directory and returns its `storage` entry.

    Raises:
        FileNotFoundError: the run's directory is missing.
        botocore errors: the upload failed (a partial upload has no run.json).
    """
    run_dir = pathlib.Path(run["model_dir"])
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory {run_dir} doesn't exist.")
    client = client or make_client(cfg)
    prefix = f"{cfg.prefix}{run['run_id']}/"
    files = run_files(run_dir)
    total = 0
    for path in files:
        key = prefix + path.relative_to(run_dir).as_posix()
        client.upload_file(str(path), cfg.bucket, key)
        total += path.stat().st_size
    location = {
        "url": f"s3://{cfg.bucket}/{prefix}",
        "bucket": cfg.bucket,
        "prefix": prefix,
        "endpoint_url": cfg.endpoint_url,
        "files": len(files),
        "bytes": total,
        "uploaded_at": datetime.now(timezone.utc),
    }
    logger.success(f"Uploaded run {run['run_id']} ({len(files)} files, {total / 1e6:.1f} MB) to {location['url']}")
    return location


def upload_and_record(run: dict, cfg: Config, client=None) -> dict:
    """Uploads the run, then adds `storage` to the run and its local run.json."""
    import topic_store

    run["storage"] = upload_run(run, cfg, client)
    topic_store.write_run_file(run)
    return run["storage"]


def download_run(run: dict, cfg: Config, dest: pathlib.Path, client=None) -> pathlib.Path:
    """Downloads an uploaded run into dest (which must not exist yet).

    Uses the bucket and prefix recorded on the run, with the current
    credentials. Files are fetched into a temporary folder next to dest and
    moved into place only once all of them arrived.

    Raises:
        FileNotFoundError: the run has no upload, or the upload has no run.json
            (it never finished).
        ValueError: a key would land outside dest.
    """
    storage = run.get("storage") or {}
    if not storage.get("bucket") or storage.get("prefix") is None:
        raise FileNotFoundError(f"Run {run['run_id']} was never uploaded.")
    client = client or make_client(cfg)
    bucket, prefix = storage["bucket"], storage["prefix"]
    keys = []
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        keys.extend(item["Key"] for item in page.get("Contents", []))
    if prefix + RUN_FILE not in keys:
        raise FileNotFoundError(f"s3://{bucket}/{prefix} has no {RUN_FILE}; the upload didn't finish.")

    dest.parent.mkdir(parents=True, exist_ok=True)
    staging = pathlib.Path(tempfile.mkdtemp(dir=dest.parent, prefix=f".{dest.name}."))
    try:
        for key in keys:
            relative = key[len(prefix):]
            target = (staging / relative).resolve()
            # A key like "../x" must not write outside the run's folder.
            if not relative or staging.resolve() not in target.parents:
                raise ValueError(f"Refusing to download s3://{bucket}/{key}: it's outside the run's folder.")
            target.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(bucket, key, str(target))
        staging.rename(dest)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    logger.success(f"Downloaded run {run['run_id']} from s3://{bucket}/{prefix} to {dest}")
    return dest


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m model_storage", description="Upload topic-model runs to S3.")
    commands = parser.add_subparsers(dest="command", required=True)
    upload = commands.add_parser("upload", help="upload saved runs (default: local runs not uploaded yet)")
    upload.add_argument("run_ids", nargs="*")
    args = parser.parse_args(argv)

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
    runs = topic_store.list_runs()
    if args.run_ids:
        by_id = {run["run_id"]: run for run in runs}
        missing = [r for r in args.run_ids if r not in by_id]
        if missing:
            print(f"error: no local run {', '.join(missing)} under {topic_store.model_dir()}", file=sys.stderr)
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
            runs_collection.update_one({"run_id": run["run_id"]}, {"$set": {"storage": storage}})
        print(f"{run['run_id']}: {storage['url']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
