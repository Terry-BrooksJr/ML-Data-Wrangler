"""Saving Tag Normalizer runs and the tag decisions they make, without any Qt.

A run is kept the way LDA and BERTopic runs are (see topic_store,
bertopic_runs and model_storage): its folder on this computer,
<TAG_RUN_DIR>/<run_id> (default tag_normalizer_runs/), is the source of
truth; when MODEL_BUCKET is set it is uploaded, with a signed manifest, to
<MODEL_PREFIX>tag_normalizer/<run_id>/; and its details (not its files) go to
the tag_normalization_run_log collection (TAG_RUNS_COLLECTION), so other
computers can list it. The folder holds the tags and their vectors, so a
saved run reopens exactly as it was without embedding anything again.

Saving a run also records its decisions in normalized_tags
(NORMALIZED_TAGS_COLLECTION), one document per successor tag:

    {
      "tag": "password_reset",        # the successor, as written
      "text": "password reset",       # cleaned (tag_normalizer.clean_tag); unique
      "replaces": [                   # every tag merged into it; tag is unique
        {"tag": "pw_reset", "text": "pw reset", "run_id": "...", "replaced_at": ...},
      ],
      "run_ids": ["..."],             # runs that changed this document
      "created_at": ..., "updated_at": ...,
    }

A deprecated tag belongs to one successor only (a unique index on
replaces.tag), and a successor is never itself deprecated: when a later run
merges successor A into B, A and everything A replaced move to B (keeping
when and by which run each was first deprecated) and A's document goes. The latest run wins: a tag deprecated earlier but kept as
itself (unticked) or chosen as a successor now leaves its old successor. A
successor left replacing nothing is removed.

New runs read every document back (previous_decisions): successors join the
grouping as anchors and deprecated tags go straight to their successor (see
tag_normalizer.Previous).
"""

import dataclasses
import getpass
import json
import os
import pathlib
from collections import Counter
from datetime import datetime, timezone
from typing import Callable, Dict, FrozenSet, Iterable, List, Optional, Tuple

import numpy as np
import tag_normalizer
import topic_store
from loguru import logger
from pymongo.collection import Collection
from tag_normalizer import Group, Previous, Tag

METHOD = "tag_normalizer"
RUNS_COLLECTION = os.environ.get("TAG_RUNS_COLLECTION", "tag_normalization_run_log")
NORMALIZED_COLLECTION = os.environ.get("NORMALIZED_TAGS_COLLECTION", "normalized_tags")
# Under MODEL_PREFIX in the bucket, apart from the topic-model runs.
STORAGE_SUBFOLDER = "tag_normalizer/"
TAGS_FILE = "tags.json"
VECTORS_FILE = "vectors.npy"
MAPPING_CSV = "mapping.csv"
MAPPING_JSON = "mapping.json"


def run_dir() -> pathlib.Path:
    return pathlib.Path(
        os.environ.get("TAG_RUN_DIR", "tag_normalizer_runs")
    ).resolve()


def _user() -> str:
    try:
        return getpass.getuser()
    except Exception:  # no login name (some containers)
        return ""


def embedding_info(backend: str) -> dict:
    """Which model made the vectors, so runs on different models aren't confused."""
    if backend == tag_normalizer.SPACY:
        return {"backend": backend, "model": "en_core_web_lg"}
    try:
        import embeddings

        cfg = embeddings.config()
        return {"backend": backend, "model": cfg.model, "dimensions": cfg.dimensions}
    except Exception:  # not configured; the vectors came from somewhere, though
        return {"backend": backend}


# --- the run's files ------------------------------------------------------------------


def _tags_document(tags: List[Tag]) -> List[dict]:
    return [
        {"text": t.text, "spellings": dict(t.spellings), "successor": t.successor}
        for t in tags
    ]


def save_run(
    tags: List[Tag],
    vectors: np.ndarray,
    groups: List[Group],
    mapped: Dict[str, str],
    settings: dict,
    edits: dict,
    previous: Optional[Previous],
    source: str = "",
    base_dir: Optional[pathlib.Path] = None,
    now: Optional[datetime] = None,
) -> dict:
    """Writes the run's folder (tags, vectors, mapping, run.json); returns its details.

    Args:
        settings: backend, embedding (embedding_info), threshold, merge_structured.
        edits: renamed ([{"tags": [cleaned texts], "canonical": name}]) and
            excluded ([cleaned texts]), the user's changes to the groups.
        previous: The earlier decisions this run was grouped with.
    """
    now = now or datetime.now(timezone.utc)
    run_id = topic_store.new_run_id(now)
    folder = (base_dir or run_dir()) / run_id
    folder.mkdir(parents=True, exist_ok=False)
    (folder / TAGS_FILE).write_text(
        json.dumps(_tags_document(tags), ensure_ascii=False), encoding="utf-8"
    )
    np.save(folder / VECTORS_FILE, np.asarray(vectors, dtype=np.float32))
    (folder / MAPPING_CSV).write_text(
        tag_normalizer.to_csv(groups, mapped), encoding="utf-8"
    )
    (folder / MAPPING_JSON).write_text(
        tag_normalizer.to_json(groups, mapped), encoding="utf-8"
    )
    changes = {old: new for old, new in mapped.items() if old != new}
    run = {
        "run_id": run_id,
        "ran_at": now,
        "method": METHOD,
        "saved_by": _user(),
        "source": source,
        "settings": settings,
        "edits": edits,
        "previous": dataclasses.asdict(previous or Previous()),
        "counts": {
            "tags": sum(1 for t in tags if not t.anchor_only),
            "groups_merged": len(set(changes.values())),
            "tags_changed": len(changes),
        },
        "changes": changes,
        "model_dir": str(folder),
    }
    topic_store.write_run_file(run)
    logger.success(f"Saved Tag Normalizer run {run_id} to {folder}")
    return run


def load_tags(
    run: dict, base_dir: Optional[pathlib.Path] = None
) -> Tuple[List[Tag], np.ndarray]:
    """The run's tags (anchors included) and their vectors, from its local folder.

    Raises:
        FileNotFoundError, ValueError: the files are missing or don't match.
    """
    folder = local_dir(run, base_dir)
    rows = json.loads((folder / TAGS_FILE).read_text(encoding="utf-8"))
    # allow_pickle=False: a downloaded .npy is data, never code.
    vectors = np.load(folder / VECTORS_FILE, allow_pickle=False)
    if not isinstance(rows, list) or len(rows) != len(vectors):
        raise ValueError(f"Run {run['run_id']}'s tags and vectors don't match.")
    tags = [
        Tag(
            str(row["text"]),
            Counter({str(k): int(v) for k, v in row["spellings"].items()}),
            row.get("successor"),
        )
        for row in rows
    ]
    return tags, vectors


def previous_of(run: dict) -> Previous:
    previous = run.get("previous") or {}
    return Previous(dict(previous.get("successors") or {}), dict(previous.get("replaced") or {}))


def renamed_of(run: dict) -> Dict[FrozenSet[str], str]:
    return {
        frozenset(item["tags"]): item["canonical"]
        for item in (run.get("edits") or {}).get("renamed", [])
    }


# --- normalized_tags ------------------------------------------------------------------


def ensure_indexes(normalized: Collection) -> None:
    normalized.create_index("text", unique=True)
    normalized.create_index(
        "replaces.tag",
        unique=True,
        partialFilterExpression={"replaces.tag": {"$exists": True}},
    )


def previous_decisions(normalized: Collection) -> Previous:
    """Every saved successor and the tags it replaces, by cleaned text."""
    previous = Previous()
    for doc in normalized.find({}, {"_id": 0, "tag": 1, "text": 1, "replaces": 1}):
        previous.successors[doc["text"]] = doc["tag"]
        for entry in doc.get("replaces") or []:
            if entry["text"] != doc["text"]:
                previous.replaced[entry["text"]] = doc["text"]
    return previous


def record_decisions(
    normalized: Collection,
    mapped: Dict[str, str],
    run_id: str,
    now: Optional[datetime] = None,
) -> dict:
    """Applies a run's mapping ({tag: tag to use}) to normalized_tags; returns counts.

    Not one transaction (a standalone Mongo has none): each step keeps the
    rules above, and the run log keeps the mapping, so a run can be saved again.
    """
    now = now or datetime.now(timezone.utc)
    ensure_indexes(normalized)
    clean = tag_normalizer.clean_tag
    changes = {old: new for old, new in mapped.items() if old != new}
    successors = set(changes.values())
    counts = Counter()

    def release(tag: str) -> int:
        """Takes tag out of whichever successor replaces it."""
        result = normalized.update_many(
            {"replaces.tag": tag},
            {"$pull": {"replaces": {"tag": tag}}, "$set": {"updated_at": now},
             "$addToSet": {"run_ids": run_id}},
        )
        return result.modified_count

    # A tag kept as itself, or now a successor, is no longer deprecated.
    kept = {tag for tag, target in mapped.items() if tag == target}
    for tag in sorted(kept | successors):
        counts["released"] += release(tag)

    for old, new in changes.items():
        text = clean(new)
        entries = [{"tag": old, "text": clean(old), "run_id": run_id, "replaced_at": now}]
        # A successor merged into another: its tags move with it (no chains).
        if clean(old) != text:
            folded = normalized.find_one_and_delete({"text": clean(old)})
            if folded:
                counts["folded"] += 1
                # Each keeps the run that first deprecated it.
                entries += [e for e in folded.get("replaces") or [] if e["tag"] != old]
        for entry in entries:
            release(entry["tag"])
        normalized.update_one(
            {"text": text},
            {
                "$set": {"tag": new, "updated_at": now},
                "$setOnInsert": {"text": text, "created_at": now},
                "$addToSet": {"run_ids": run_id},
                "$push": {"replaces": {"$each": entries}},
            },
            upsert=True,
        )
        counts["deprecated"] += 1
    # A successor replacing nothing is no successor.
    counts["removed"] = normalized.delete_many(
        {"$or": [{"replaces": {"$size": 0}}, {"replaces": {"$exists": False}}]}
    ).deleted_count
    counts["successors"] = len(successors)
    return dict(counts)


# --- saving a run ---------------------------------------------------------------------


def storage_config(cfg):
    """The LDA storage settings with the Tag Normalizer's own folder in the bucket."""
    return dataclasses.replace(cfg, prefix=f"{cfg.prefix}{STORAGE_SUBFOLDER}")


def upload(run: dict, storage: Callable[[], Optional[object]], client=None) -> None:
    """Uploads the run when storage is configured; a failure becomes a warning on the run."""
    import model_storage

    run.setdefault("warnings", [])
    try:
        cfg = storage()
    except model_storage.StorageNotConfigured as e:
        run["warnings"].append(f"Not uploaded to S3: {e}")
        return
    if cfg is None:
        return
    cfg = storage_config(cfg)
    try:
        model_storage.upload_and_record(run, cfg, client)
    except Exception as e:
        logger.exception(f"Could not upload Tag Normalizer run {run['run_id']} to S3: {e}")
        run["warnings"].append(
            f"Saved locally, but the upload to s3://{cfg.bucket} failed: {e}. "
            f"Retry with: python -m model_storage upload --tags {run['run_id']}"
        )


def _document(run: dict) -> dict:
    """What Mongo keeps of a run: its details, not this computer's notes."""
    return {k: v for k, v in run.items() if k not in ("_id", "warnings")}


def record_run(runs: Collection, run: dict) -> None:
    runs.replace_one({"run_id": run["run_id"]}, _document(run), upsert=True)


def save(
    tags: List[Tag],
    vectors: np.ndarray,
    groups: List[Group],
    mapped: Dict[str, str],
    settings: dict,
    edits: dict,
    previous: Optional[Previous],
    source: str = "",
    database=None,
    storage: Callable[[], Optional[object]] = lambda: None,
    base_dir: Optional[pathlib.Path] = None,
    client=None,
) -> dict:
    """Saves the run here, uploads it, records its decisions and the run in Mongo.

    Each step after saving the folder that fails becomes a warning on the
    returned run (as in bertopic_runs), so a saved run is never lost. Without
    a database (no MONGO_URI) only the folder and upload happen.
    """
    run = save_run(
        tags, vectors, groups, mapped, settings, edits, previous, source, base_dir
    )
    run["warnings"] = []
    upload(run, storage, client)
    if database is None:
        run["warnings"].append(
            "MongoDB isn't set up, so normalized_tags and the run log weren't updated."
        )
        return run
    try:
        run["normalized"] = record_decisions(
            database[NORMALIZED_COLLECTION], mapped, run["run_id"]
        )
    except Exception as e:
        logger.exception(f"Could not update {NORMALIZED_COLLECTION}: {e}")
        run["warnings"].append(f"Saved, but {NORMALIZED_COLLECTION} could not be updated: {e}")
    else:
        topic_store.write_run_file(run)
    try:
        record_run(database[RUNS_COLLECTION], run)
    except Exception as e:
        logger.exception(f"Could not record Tag Normalizer run {run['run_id']}: {e}")
        run["warnings"].append(f"Saved locally, but MongoDB could not be updated: {e}")
    return run


# --- saved runs -----------------------------------------------------------------------


def list_runs(
    base_dir: Optional[pathlib.Path] = None, runs: Optional[Collection] = None
) -> List[dict]:
    """Tag Normalizer runs, newest first: from Mongo when given, else those on this computer."""
    if runs is not None:
        return list(runs.find({"method": METHOD}, {"_id": 0}).sort("ran_at", -1))
    base_dir = base_dir or run_dir()
    if not base_dir.exists():
        return []
    found = (
        topic_store._read_run_file(p) for p in base_dir.glob(f"*/{topic_store.RUN_FILE}")
    )
    runs_found = [r for r in found if r and r.get("method") == METHOD]
    return sorted(runs_found, key=lambda r: str(r.get("ran_at")), reverse=True)


def known_runs(
    runs: Optional[Collection] = None, base_dir: Optional[pathlib.Path] = None
) -> List[dict]:
    """Runs from Mongo and this computer, by run_id, newest first (Mongo's record wins,
    but a local upload record survives if Mongo missed it)."""
    local = {run["run_id"]: run for run in list_runs(base_dir)}
    merged = dict(local)
    if runs is not None:
        for run in list_runs(runs=runs):
            if not run.get("storage") and (local.get(run["run_id"]) or {}).get("storage"):
                run["storage"] = local[run["run_id"]]["storage"]
            merged[run["run_id"]] = run
    return sorted(merged.values(), key=lambda r: str(r.get("ran_at")), reverse=True)


def describe_run(run: dict) -> str:
    ran_at = run.get("ran_at")
    ran_at = ran_at.strftime("%Y-%m-%d %H:%M") if isinstance(ran_at, datetime) else str(ran_at)
    counts = run.get("counts") or {}
    backend = (run.get("settings") or {}).get("backend", "?")
    return (
        f"{ran_at} · {counts.get('tags', '?')} tags, {counts.get('tags_changed', '?')} changed"
        f" · {backend} · {run.get('saved_by') or 'unknown'}"
    )


def local_dir(run: dict, base_dir: Optional[pathlib.Path] = None) -> pathlib.Path:
    """<TAG_RUN_DIR>/<run_id>; the run's recorded model_dir is never trusted.

    Raises:
        ValueError: the run ID isn't a plain folder name.
    """
    import model_storage

    base = (base_dir or run_dir()).resolve()
    folder = (base / model_storage.check_run_id(run["run_id"])).resolve()
    if folder.parent != base:
        raise ValueError(f"Refusing run {run['run_id']!r}: it would be outside {base}.")
    return folder


def _complete(folder: pathlib.Path) -> bool:
    return all((folder / name).is_file() for name in (TAGS_FILE, VECTORS_FILE, topic_store.RUN_FILE))


def has_local_files(run: dict, base_dir: Optional[pathlib.Path] = None) -> bool:
    try:
        return _complete(local_dir(run, base_dir))
    except ValueError:
        return False


def where(run: dict, base_dir: Optional[pathlib.Path] = None) -> str:
    if has_local_files(run, base_dir):
        return "on this computer"
    return f"in {run['storage']['url']}" if run.get("storage") else "in MongoDB only"


def ensure_local(
    run: dict,
    base_dir: Optional[pathlib.Path] = None,
    storage: Callable[[], Optional[object]] = None,
    client=None,
) -> dict:
    """Makes the run's files available in <TAG_RUN_DIR>/<run_id>, downloading them if needed.

    The upload is verified against MODEL_SIGNING_KEY before anything is installed.

    Raises:
        FileNotFoundError: the files aren't here and can't be downloaded.
        ValueError: the run ID isn't a plain folder name.
        model_storage.UntrustedRun: the upload failed verification.
    """
    import model_storage

    local = local_dir(run, base_dir)
    if not _complete(local):
        if not run.get("storage"):
            raise FileNotFoundError(
                f"Tag Normalizer run {run['run_id']}'s files aren't on this computer and it was "
                "never uploaded."
            )
        cfg = (storage or model_storage.config)()
        if cfg is None:
            raise FileNotFoundError(
                f"Tag Normalizer run {run['run_id']} isn't on this computer. It was uploaded to "
                f"{run['storage'].get('url')}; set the model storage settings to download it."
            )
        model_storage.download_run(run, storage_config(cfg), local, client)
        if not _complete(local):
            raise FileNotFoundError(f"Tag Normalizer run {run['run_id']}'s upload is incomplete.")
    run["model_dir"] = str(local)
    return run
