"""Saves topic-model runs and links them to tickets.

Each training run gets a directory under TOPIC_MODEL_DIR holding the LDA
model and its dictionary (always saved together), its charts, and run.json.
Model files stay on local disk (and, when MODEL_BUCKET is set, are copied to
S3 by model_storage): Atlas free tier caps storage at 512 MB, so Mongo keeps
only each run's details and a pointer to its directory, in the
`topic_model_runs` collection. Each ticket's main topic is written to its
`tickets` document under `topic`, tagged with the run that assigned it;
re-ingestion never touches that field.
"""
import hashlib
import json
import os
import pathlib
import uuid
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import gensim
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
from loguru import logger
from pymongo.collection import Collection

from LDA_logic import REMOVED_POS, LatentDirichletAllocator, tokenize

RUNS_COLLECTION = os.environ.get("TOPIC_RUNS_COLLECTION", "topic_model_runs")
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
        "spacy_model": f"{meta.get('lang', '')}_{meta.get('name', '')}".strip("_") or None,
        "spacy_model_version": meta.get("version"),
        "stopwords_count": len(stopwords),
        "stopwords_hash": hashlib.sha256("\n".join(sorted(stopwords)).encode("utf-8")).hexdigest()[:16],
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
            {"num_topics": n, "coherence": c} for n, c in zip(allocator.topic_counts, allocator.coherence_values)
        ],
        "best_num_topics": allocator.best_num_topics,
        "topics": [
            {"topic_id": topic_id, "share": share, "words": words}
            for topic_id, share, words in allocator.get_topics()
        ],
        "dataset": dataset_fingerprint(allocator.prelemma_corpus or [], allocator.document_ids),
        "model_dir": str(run_dir),
        "charts": {"lda_html": chart_path, "coherence_png": plot_path},
        "warnings": [],
    }
    write_run_file(run)
    logger.success(f"Saved topic-model run {run_id} to {run_dir}")
    return run


def write_run_file(run: dict) -> None:
    """Writes the run's details to run.json in its directory."""
    with open(pathlib.Path(run["model_dir"]) / RUN_FILE, "w") as f:
        json.dump({k: v for k, v in run.items() if k != "_id"}, f, indent=2, default=str)


def record_run(runs: Collection, run: dict) -> None:
    """Stores the run's details (not its model files) in topic_model_runs."""
    runs.replace_one({"run_id": run["run_id"]}, {k: v for k, v in run.items() if k != "_id"}, upsert=True)


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


def list_runs(runs: Optional[Collection] = None, base_dir: Optional[pathlib.Path] = None) -> List[dict]:
    """Previous runs, newest first: from Mongo when given, else from the run.json files on disk."""
    if runs is not None:
        return list(runs.find({}, {"_id": 0}).sort("ran_at", -1))
    base_dir = base_dir or model_dir()
    found = [_read_run_file(p) for p in base_dir.glob(f"*/{RUN_FILE}")] if base_dir.exists() else []
    return sorted((r for r in found if r), key=lambda r: str(r.get("ran_at")), reverse=True)


def describe_run(run: dict) -> str:
    ran_at = run.get("ran_at")
    ran_at = ran_at.strftime("%Y-%m-%d %H:%M") if isinstance(ran_at, datetime) else str(ran_at)
    best = max((c["coherence"] for c in run.get("coherence") or []), default=None)
    dataset = run.get("dataset") or {}
    return (
        f"{run['run_id']} · {ran_at} UTC · {run.get('best_num_topics')} topics"
        + (f" · coherence {best:.3f}" if best is not None else "")
        + f" · {dataset.get('ticket_count', '?')} tickets"
    )


def has_local_files(run: dict) -> bool:
    """True if the run's model and dictionary are on this computer."""
    run_dir = pathlib.Path(run.get("model_dir") or "")
    return (run_dir / MODEL_FILE).is_file() and (run_dir / DICTIONARY_FILE).is_file()


def _relocate(run: dict, new_dir: pathlib.Path) -> None:
    """Points the run's model_dir and chart paths at new_dir."""
    old_dir = pathlib.Path(run.get("model_dir") or "")
    charts = dict(run.get("charts") or {})
    for name, path in charts.items():
        if path:
            path = pathlib.Path(path)
            try:
                charts[name] = str(new_dir / path.relative_to(old_dir))
            except ValueError:
                charts[name] = str(new_dir / path.name)
    run["charts"] = charts
    run["model_dir"] = str(new_dir)


def ensure_local(run: dict, base_dir: Optional[pathlib.Path] = None) -> dict:
    """Makes the run's files available locally, downloading them from S3 if needed.

    Looks in the run's recorded folder, then in this computer's TOPIC_MODEL_DIR
    (a run trained elsewhere), then downloads its upload. Updates the run's
    paths to where the files are.

    Raises:
        FileNotFoundError: the files aren't here and can't be downloaded.
    """
    if has_local_files(run):
        return run
    local = (base_dir or model_dir()) / run["run_id"]
    if not (local / MODEL_FILE).is_file() and run.get("storage"):
        import model_storage

        cfg = model_storage.config()
        if cfg is None:
            raise FileNotFoundError(
                f"Run {run['run_id']} isn't on this computer. It was uploaded to {run['storage'].get('url')}; "
                "set the model storage settings to download it."
            )
        if local.exists():
            raise FileNotFoundError(f"{local} exists but holds no complete model; move it aside to download the run again.")
        model_storage.download_run(run, cfg, local)
    if not (local / MODEL_FILE).is_file():
        raise FileNotFoundError(f"Run {run['run_id']}'s model files aren't on this computer and it was never uploaded.")
    _relocate(run, local)
    return run


def load_run_model(run: dict) -> Tuple[LdaModel, Dictionary]:
    """Loads a saved run's model and dictionary, without retraining (downloading it if needed)."""
    run_dir = pathlib.Path(ensure_local(run)["model_dir"])
    return LdaModel.load(str(run_dir / MODEL_FILE)), Dictionary.load(str(run_dir / DICTIONARY_FILE))


def _without_seed(settings: Optional[dict]) -> dict:
    return {k: v for k, v in (settings or {}).items() if k != "seed"}


def find_matching_run(runs: Iterable[dict], dataset: dict, settings: dict, preprocessing: dict) -> Optional[dict]:
    """The newest run trained on the same tickets with the same settings and preprocessing.

    The seed is ignored: each training run picks its own, so retraining with
    identical inputs would only differ by chance.
    """
    wanted = (dataset, _without_seed(settings), preprocessing)
    matches = [run for run in runs
               if (run.get("dataset"), _without_seed(run.get("settings")), run.get("preprocessing")) == wanted]
    return max(matches, key=lambda run: str(run.get("ran_at")), default=None)


def known_runs(runs: Optional[Collection] = None, base_dir: Optional[pathlib.Path] = None) -> List[dict]:
    """Runs from Mongo (shared between computers) and this computer's disk, by run_id, newest first."""
    local = {run["run_id"]: run for run in list_runs(base_dir=base_dir)}
    merged = dict(local)
    if runs is not None:
        # Mongo wins: it has uploads recorded from any computer. A local upload
        # record survives if Mongo missed it (its update failed).
        for run in list_runs(runs):
            if not run.get("storage") and (local.get(run["run_id"]) or {}).get("storage"):
                run["storage"] = local[run["run_id"]]["storage"]
            merged[run["run_id"]] = run
    return sorted(merged.values(), key=lambda r: str(r.get("ran_at")), reverse=True)


def topic_assignment(model: LdaModel, bow, run_id: str, now: datetime) -> Optional[dict]:
    """A ticket's `topic` field: its main topic and topic mix under one run."""
    mix = sorted(model.get_document_topics(bow, minimum_probability=0.0), key=lambda t: t[1], reverse=True)
    if not bow or not mix:
        return None
    main_topic, probability = mix[0]
    return {
        "run_id": run_id,
        "main_topic": int(main_topic),
        "probability": float(probability),
        "words": [word for word, _ in model.show_topic(main_topic, topn=5)],
        "mix": [{"topic": int(t), "probability": float(p)} for t, p in mix if p >= TOPIC_MIX_MIN],
        "assigned_at": now,
    }


def assign_topics(tickets: Collection, run_id: str, model: LdaModel, bows: Dict[str, list]) -> dict:
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
        matched = tickets.update_one({"zendesk_id": str(zendesk_id)}, {"$set": {"topic": assignment}}).matched_count
        summary["assigned" if matched else "not_in_mongo"] += 1
    logger.info(f"Topic assignment for run {run_id}: {summary}")
    return summary


def assign_trained_topics(tickets: Collection, run: dict, allocator: LatentDirichletAllocator) -> dict:
    """Assigns topics to the tickets a run was just trained on, reusing its corpus."""
    if len(allocator.document_ids) != len(allocator.corpus):
        raise ValueError("The corpus has no ticket IDs to assign topics to.")
    return assign_topics(tickets, run["run_id"], allocator.lda_model, dict(zip(allocator.document_ids, allocator.corpus)))


def assign_from_saved_run(
    tickets: Collection, run: dict, nlp, stopwords: Sequence[str], query: Optional[dict] = None
) -> dict:
    """Loads a saved run and assigns its topics to tickets read from Mongo.

    Args:
        query: Which tickets to assign; defaults to all of them.
    """
    from wrangler import DataWrangler

    model, dictionary = load_run_model(run)
    docs = list(tickets.find(query or {}, {"zendesk_id": 1, "subject": 1, "transcript.body": 1}))
    texts = [DataWrangler.mongo_ticket_document(doc) for doc in docs]
    bows = {doc["zendesk_id"]: dictionary.doc2bow(tokens) for doc, tokens in zip(docs, tokenize(nlp, stopwords, texts))}
    return assign_topics(tickets, run["run_id"], model, bows)


def format_run(run: dict) -> Iterable[str]:
    """Markdown lines for a run's topics, most prevalent first."""
    for rank, topic in enumerate(run.get("topics") or [], start=1):
        yield f"{rank}. **{topic['share']:.0%} of tickets** (topic {topic['topic_id']}): {', '.join(topic['words'])}"
