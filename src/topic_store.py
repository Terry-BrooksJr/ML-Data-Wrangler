"""Saves topic-model runs and links them to tickets.

Each training run gets a directory under TOPIC_MODEL_DIR holding the LDA
model and its dictionary (always saved together), its charts, and run.json.
Model files stay on local disk (and, when MODEL_BUCKET is set, are copied to
S3 by model_storage): Atlas free tier caps storage at 512 MB, so Mongo keeps
only each run's details and a pointer to its directory, in the
`topic_modeling_run_log` collection (TOPIC_RUNS_COLLECTION). Each ticket's main topic is written to its
`tickets` document under `topic`, tagged with the run that assigned it;
re-ingestion never touches that field.
"""

import hashlib
import json
import os
import pathlib
import re
import tempfile
import uuid
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import gensim
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
from LDA_logic import LatentDirichletAllocator, REMOVED_POS, tokenize
from loguru import logger
from pymongo.collection import Collection

def runs_collection() -> str:
    """The LDA runs collection's name, read when used so a saved setting applies at once."""
    return os.environ.get("TOPIC_RUNS_COLLECTION", "topic_modeling_run_log")

MODEL_FILE = "lda.model"
DICTIONARY_FILE = "dictionary.dict"
RUN_FILE = "run.json"
TOPIC_MIX_MIN = 0.05  # topics below this share of a ticket are left out of its mix


def model_dir() -> pathlib.Path:
    return pathlib.Path(os.environ.get("TOPIC_MODEL_DIR", "topic_models")).resolve()


def new_run_id(now: Optional[datetime] = None) -> str:
    now = now or datetime.now(timezone.utc)
    return f"{now:%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}"


def dataset_fingerprint(documents: Sequence[str], ids: Sequence[str]) -> dict:
    """Identifies the data a run was trained on: ticket count plus a content hash."""
    digest = hashlib.sha256()
    for ticket_id, document in zip(ids or [""] * len(documents), documents):
        digest.update(f"{ticket_id}\x1f{document}\x1e".encode("utf-8"))
    return {"ticket_count": len(documents), "hash": digest.hexdigest()[:16]}


def preprocessing_info(nlp, stopwords: Sequence[str]) -> dict:
    """What's needed to reproduce a run's preprocessing."""
    import spacy

    meta = getattr(nlp, "meta", {}) or {}
    return {
        "spacy_version": spacy.__version__,
        "spacy_model": f"{meta.get('lang', '')}_{meta.get('name', '')}".strip("_")
        or None,
        "spacy_model_version": meta.get("version"),
        "stopwords_count": len(stopwords),
        "stopwords_hash": hashlib.sha256(
            "\n".join(sorted(stopwords)).encode("utf-8")
        ).hexdigest()[:16],
        "removed_pos": list(REMOVED_POS),
        "gensim_version": gensim.__version__,
    }


def save_run(
    allocator: LatentDirichletAllocator,
    nlp,
    stopwords: Sequence[str],
    base_dir: Optional[pathlib.Path] = None,
    now: Optional[datetime] = None,
) -> dict:
    """Saves the best model, its dictionary and charts, and returns the run document.

    Raises:
        ValueError: if the allocator has no trained model.
    """
    if allocator.lda_model is None or allocator.id2word is None:
        raise ValueError("There is no trained model to save.")
    now = now or datetime.now(timezone.utc)
    run_id = new_run_id(now)
    run_dir = (base_dir or model_dir()) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    allocator.lda_model.save(str(run_dir / MODEL_FILE))
    allocator.id2word.save(str(run_dir / DICTIONARY_FILE))
    chart_path, plot_path = allocator.visualize_results(run_dir)

    run = {
        "run_id": run_id,
        "ran_at": now,
        "settings": allocator.settings,
        "preprocessing": preprocessing_info(nlp, stopwords),
        "coherence": [
            {"num_topics": n, "coherence": c}
            for n, c in zip(allocator.topic_counts, allocator.coherence_values)
        ],
        "best_num_topics": allocator.best_num_topics,
        "topics": [
            {"topic_id": topic_id, "share": share, "words": words}
            for topic_id, share, words in allocator.get_topics()
        ],
        "dataset": dataset_fingerprint(
            allocator.prelemma_corpus or [], allocator.document_ids
        ),
        "model_dir": str(run_dir),
        "charts": {"lda_html": chart_path, "coherence_png": plot_path},
        "warnings": [],
    }
    write_run_file(run)
    logger.success(f"Saved topic-model run {run_id} to {run_dir}")
    return run


def write_run_file(run: dict) -> None:
    """Writes the run's details to run.json in its directory, atomically.

    The text is written to a temporary file beside it and swapped in only
    once complete: a full disk or a crash mid-write (e.g. while recording an
    upload) must not leave an unreadable run.json, which would hide the run.
    """
    run_dir = pathlib.Path(run["model_dir"])
    text = json.dumps(
        {k: v for k, v in run.items() if k != "_id"}, indent=2, default=str
    )
    fd, tmp = tempfile.mkstemp(dir=run_dir, prefix=f".{RUN_FILE}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, run_dir / RUN_FILE)
    except BaseException:
        os.unlink(tmp)
        raise


def record_run(runs: Collection, run: dict) -> None:
    """Stores the run's details (not its model files) in the LDA runs collection."""
    runs.replace_one(
        {"run_id": run["run_id"]},
        {k: v for k, v in run.items() if k != "_id"},
        upsert=True,
    )


def _read_run_file(path: pathlib.Path) -> Optional[dict]:
    try:
        with open(path) as f:
            run = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(f"Skipping unreadable run file {path}: {e}")
        return None
    if isinstance(run.get("ran_at"), str):
        run["ran_at"] = datetime.fromisoformat(run["ran_at"])
    return run


# run.json's `method` for runs that aren't LDA models (bertopic_model saves "bertopic").
# A run with no method is LDA: runs saved before the field existed.
LDA_METHODS = (None, "lda")


def list_runs(
    runs: Optional[Collection] = None, base_dir: Optional[pathlib.Path] = None
) -> List[dict]:
    """Previous LDA runs, newest first: from Mongo when given, else from the run.json files on disk.

    Runs of another method (e.g. a BERTopic run.json copied into
    TOPIC_MODEL_DIR) are skipped: everything listed here is loaded as an LDA model.
    """
    if runs is not None:
        return list(
            runs.find({"method": {"$in": list(LDA_METHODS)}}, {"_id": 0}).sort(
                "ran_at", -1
            )
        )
    base_dir = base_dir or model_dir()
    found = (
        [_read_run_file(p) for p in base_dir.glob(f"*/{RUN_FILE}")]
        if base_dir.exists()
        else []
    )
    return sorted(
        (r for r in found if r and r.get("method") in LDA_METHODS),
        key=lambda r: str(r.get("ran_at")),
        reverse=True,
    )


def describe_run(run: dict) -> str:
    ran_at = run.get("ran_at")
    ran_at = (
        ran_at.strftime("%Y-%m-%d %H:%M")
        if isinstance(ran_at, datetime)
        else str(ran_at)
    )
    best = max((c["coherence"] for c in run.get("coherence") or []), default=None)
    dataset = run.get("dataset") or {}
    return (
        f"{run['run_id']} · {ran_at} UTC · {run.get('best_num_topics')} topics"
        + (f" · coherence {best:.3f}" if best is not None else "")
        + f" · {dataset.get('ticket_count', '?')} tickets"
    )


def local_dir(run: dict, base_dir: Optional[pathlib.Path] = None) -> pathlib.Path:
    """Where this computer keeps the run: <TOPIC_MODEL_DIR>/<run_id>.

    The run's recorded model_dir is never used for loading: a record from
    Mongo could point anywhere, and model files are pickles that run code
    when loaded. Only this folder, holding runs trained here or downloaded
    and verified, is trusted.

    Raises:
        ValueError: the run ID isn't a plain folder name.
    """
    import model_storage

    base = (base_dir or model_dir()).resolve()
    folder = (base / model_storage.check_run_id(run["run_id"])).resolve()
    if folder.parent != base:
        raise ValueError(f"Refusing run {run['run_id']!r}: it would be outside {base}.")
    return folder


def _complete(folder: pathlib.Path) -> bool:
    return (folder / MODEL_FILE).is_file() and (folder / DICTIONARY_FILE).is_file()


def has_local_files(run: dict, base_dir: Optional[pathlib.Path] = None) -> bool:
    """True if the run's model and dictionary are both in this computer's model folder."""
    try:
        return _complete(local_dir(run, base_dir))
    except ValueError:
        return False


def _recorded_path(path: str) -> pathlib.PurePath:
    """A path recorded on whichever computer trained the run, parsed for that platform.

    A Windows path ("C:\\Users\\...\\charts\\lda_model.html") read on macOS
    or Linux would otherwise be one long file name.
    """
    if (
        re.match(r"^[A-Za-z]:[\\/]", path)
        or path.startswith("\\\\")
        or ("\\" in path and "/" not in path)
    ):
        return pathlib.PureWindowsPath(path)
    return pathlib.PurePosixPath(path)


def _relocate(run: dict, new_dir: pathlib.Path) -> None:
    """Points the run's model_dir and chart paths at new_dir.

    Recorded paths may come from another computer (and platform), so each
    chart keeps only its part inside the run's folder, e.g. charts/x.html.
    """
    old_dir = _recorded_path(str(run.get("model_dir") or ""))
    charts = dict(run.get("charts") or {})
    for name, path in charts.items():
        if path:
            recorded = _recorded_path(str(path))
            old = type(recorded)(str(old_dir)) if old_dir.parts else None
            try:
                relative_parts = recorded.relative_to(old).parts if old else None
            except ValueError:
                relative_parts = None
            if relative_parts is None:
                # Recorded under another layout: keep what follows the run's
                # own folder, else just the file name.
                parts = recorded.parts
                after = max(
                    (i for i, part in enumerate(parts) if part == run["run_id"]),
                    default=None,
                )
                relative_parts = (
                    parts[after + 1 :]
                    if after is not None and after + 1 < len(parts)
                    else (recorded.name,)
                )
            local = new_dir.joinpath(*relative_parts)
            # A recorded chart path can't point outside the run's folder either.
            charts[name] = (
                str(local) if new_dir.resolve() in local.resolve().parents else None
            )
    run["charts"] = charts
    run["model_dir"] = str(new_dir)


def ensure_local(run: dict, base_dir: Optional[pathlib.Path] = None) -> dict:
    """Makes the run's files available in this computer's model folder, downloading them if needed.

    Uses <TOPIC_MODEL_DIR>/<run_id> when it has both the model and the
    dictionary; otherwise downloads and verifies the run's upload (see
    model_storage.download_run). Updates the run's paths to that folder.

    Raises:
        FileNotFoundError: the files aren't here and can't be downloaded.
        ValueError: the run ID isn't a plain folder name.
        model_storage.UntrustedRun: the upload failed verification.
    """
    local = local_dir(run, base_dir)
    if not _complete(local):
        if not run.get("storage"):
            raise FileNotFoundError(
                f"Run {run['run_id']}'s model files aren't on this computer and it was never uploaded."
            )
        import model_storage

        cfg = model_storage.config()
        if cfg is None:
            raise FileNotFoundError(
                f"Run {run['run_id']} isn't on this computer. It was uploaded to {run['storage'].get('url')}; "
                "set the model storage settings to download it."
            )
        model_storage.download_run(run, cfg, local)
        if not _complete(local):
            raise FileNotFoundError(
                f"Run {run['run_id']}'s upload has no {MODEL_FILE} or {DICTIONARY_FILE}."
            )
    _relocate(run, local)
    _persist_local_details(run, local)
    return run


def _persist_local_details(run: dict, local: pathlib.Path) -> None:
    """Rewrites the local run.json if it doesn't match the relocated run.

    A downloaded run.json was written on the computer that trained the run:
    its model_dir points there, and it predates the upload, so it has no
    `storage`. Left as is, list_runs would show the wrong paths and a default
    `model_storage upload` would pick the run as never uploaded.
    """
    current = _read_run_file(local / RUN_FILE) or {}
    if current.get("model_dir") != run["model_dir"] or current.get(
        "storage"
    ) != _jsonable(run.get("storage")):
        write_run_file(run)


def _jsonable(value):
    """value as run.json stores it (dates become strings), for comparison."""
    return json.loads(json.dumps(value, default=str))


def load_run_model(
    run: dict, base_dir: Optional[pathlib.Path] = None
) -> Tuple[LdaModel, Dictionary]:
    """Loads a saved run's model and dictionary, without retraining (downloading it if needed).

    Only files in this computer's model folder are loaded (see local_dir).
    """
    run_dir = pathlib.Path(ensure_local(run, base_dir)["model_dir"])
    return LdaModel.load(str(run_dir / MODEL_FILE)), Dictionary.load(
        str(run_dir / DICTIONARY_FILE)
    )


def _without_seed(settings: Optional[dict]) -> dict:
    return {k: v for k, v in (settings or {}).items() if k != "seed"}


def find_matching_run(
    runs: Iterable[dict], dataset: dict, settings: dict, preprocessing: dict
) -> Optional[dict]:
    """The newest run trained on the same tickets with the same settings and preprocessing.

    The seed is ignored: each training run picks its own, so retraining with
    identical inputs would only differ by chance.
    """
    wanted = (dataset, _without_seed(settings), preprocessing)
    matches = [
        run
        for run in runs
        if (
            run.get("dataset"),
            _without_seed(run.get("settings")),
            run.get("preprocessing"),
        )
        == wanted
    ]
    return max(matches, key=lambda run: str(run.get("ran_at")), default=None)


def known_runs(
    runs: Optional[Collection] = None, base_dir: Optional[pathlib.Path] = None
) -> List[dict]:
    """Runs from Mongo (shared between computers) and this computer's disk, by run_id, newest first."""
    local = {run["run_id"]: run for run in list_runs(base_dir=base_dir)}
    merged = dict(local)
    if runs is not None:
        # Mongo wins: it has uploads recorded from any computer. A local upload
        # record survives if Mongo missed it (its update failed).
        for run in list_runs(runs):
            if not run.get("storage") and (local.get(run["run_id"]) or {}).get(
                "storage"
            ):
                run["storage"] = local[run["run_id"]]["storage"]
            merged[run["run_id"]] = run
    return sorted(merged.values(), key=lambda r: str(r.get("ran_at")), reverse=True)


def topic_assignment(
    model: LdaModel, bow, run_id: str, now: datetime
) -> Optional[dict]:
    """A ticket's `topic` field: its main topic and topic mix under one run."""
    mix = sorted(
        model.get_document_topics(bow, minimum_probability=0.0),
        key=lambda t: t[1],
        reverse=True,
    )
    if not bow or not mix:
        return None
    main_topic, probability = mix[0]
    return {
        "run_id": run_id,
        "main_topic": int(main_topic),
        "probability": float(probability),
        "words": [word for word, _ in model.show_topic(main_topic, topn=5)],
        "mix": [
            {"topic": int(t), "probability": float(p)}
            for t, p in mix
            if p >= TOPIC_MIX_MIN
        ],
        "assigned_at": now,
    }


def assign_topics(
    tickets: Collection, run_id: str, model: LdaModel, bows: Dict[str, list]
) -> dict:
    """Writes each ticket's topic under `run_id` to its tickets document.

    Only existing tickets are updated (no upsert). Tickets whose text has no
    words in the model's dictionary are left unassigned.

    Returns:
        Counts: assigned, empty (no usable words), not_in_mongo.
    """
    now = datetime.now(timezone.utc)
    summary = {"assigned": 0, "empty": 0, "not_in_mongo": 0}
    for zendesk_id, bow in bows.items():
        assignment = topic_assignment(model, bow, run_id, now)
        if assignment is None:
            summary["empty"] += 1
            continue
        matched = tickets.update_one(
            {"zendesk_id": str(zendesk_id)}, {"$set": {"topic": assignment}}
        ).matched_count
        summary["assigned" if matched else "not_in_mongo"] += 1
    logger.info(f"Topic assignment for run {run_id}: {summary}")
    return summary


def assign_trained_topics(
    tickets: Collection, run: dict, allocator: LatentDirichletAllocator
) -> dict:
    """Assigns topics to the tickets a run was just trained on, reusing its corpus."""
    if len(allocator.document_ids) != len(allocator.corpus):
        raise ValueError("The corpus has no ticket IDs to assign topics to.")
    return assign_topics(
        tickets,
        run["run_id"],
        allocator.lda_model,
        dict(zip(allocator.document_ids, allocator.corpus)),
    )


def assign_from_saved_run(
    tickets: Collection,
    run: dict,
    nlp,
    stopwords: Sequence[str],
    query: Optional[dict] = None,
) -> dict:
    """Loads a saved run and assigns its topics to tickets read from Mongo.

    Args:
        query: Which tickets to assign; defaults to all of them.
    """
    from wrangler import DataWrangler

    model, dictionary = load_run_model(run)
    docs = list(
        tickets.find(query or {}, {"zendesk_id": 1, "subject": 1, "transcript.body": 1})
    )
    texts = [DataWrangler.mongo_ticket_document(doc) for doc in docs]
    bows = {
        doc["zendesk_id"]: dictionary.doc2bow(tokens)
        for doc, tokens in zip(docs, tokenize(nlp, stopwords, texts))
    }
    return assign_topics(tickets, run["run_id"], model, bows)


def format_run(run: dict) -> Iterable[str]:
    """Markdown lines for a run's topics, most prevalent first."""
    for rank, topic in enumerate(run.get("topics") or [], start=1):
        yield f"{rank}. **{topic['share']:.0%} of tickets** (topic {topic['topic_id']}): {', '.join(topic['words'])}"
