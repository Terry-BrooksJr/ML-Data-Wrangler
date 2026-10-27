"""Training and listing BERTopic runs for the app, without any Qt.

bertopic_model does the work; this module strings its steps together for
the app's worker thread and reads back the runs it saved. BERTopic runs live
in BERTOPIC_MODEL_DIR, never TOPIC_MODEL_DIR, and their run.json says
`"method": "bertopic"`, which topic_store.list_runs skips, so the LDA lists
never try to load one as an LDA model.

Writing each ticket's topic to Mongo is opt-in (TrainOptions.assign) and
only ever touches `embedding_topic`, never the LDA `topic` field.

Runs are kept the way LDA runs are (see topic_store and model_storage): the
folder on this computer is the source of truth; when MODEL_BUCKET is set it
is uploaded, with a signed manifest, to <MODEL_PREFIX>bertopic/<run_id>/; and
the run's details (not its files) are recorded in the bertopic_run_log
collection (BERTOPIC_RUNS_COLLECTION), apart from the LDA runs, so other
computers can list it. A run whose files aren't here is
downloaded and verified before it is opened, into <BERTOPIC_MODEL_DIR>/<run_id>
only: its model is a pickle, which runs code when loaded.
"""

import csv
import dataclasses
import io
import os
import pathlib
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from topics import bertopic_model
from topics import topic_store
from topics.LDA_logic import tokenize
from loguru import logger
from pymongo.collection import Collection

METHOD = "bertopic"
# Under MODEL_PREFIX in the bucket, apart from the LDA runs.
STORAGE_SUBFOLDER = "bertopic/"
CHARTS = ("barchart_html", "intertopic_html")


def runs_collection() -> str:
    """The BERTopic runs collection's name, read when used so a saved setting applies at once."""
    return os.environ.get("BERTOPIC_RUNS_COLLECTION", "bertopic_run_log")


@dataclass
class TrainOptions:
    """What the user picked in the app; None means bertopic_model's default."""

    min_cluster_size: Optional[int] = None
    nr_topics: Optional[int] = None
    keep_outliers: bool = False
    # A saved LDA run (in TOPIC_MODEL_DIR) to score on the same tickets.
    compare_lda: Optional[str] = None
    # Write each ticket's topic to `embedding_topic` in Mongo.
    assign: bool = False
    # Only tickets with these Zendesk outcomes (None for unset); None: every ticket.
    outcomes: Optional[List[Optional[str]]] = None

    def settings(self) -> bertopic_model.Settings:
        return bertopic_model.Settings(
            min_cluster_size=self.min_cluster_size,
            nr_topics=self.nr_topics,
            reduce_outliers=not self.keep_outliers,
        )


def train_and_save(
    tickets: Collection,
    nlp,
    stop_words: Sequence[str],
    options: TrainOptions,
    status: Callable[[str], None] = lambda message: None,
    base_dir: Optional[pathlib.Path] = None,
    lda_dir: Optional[pathlib.Path] = None,
    runs: Optional[Collection] = None,
    storage: Callable[[], Optional[object]] = lambda: None,
) -> dict:
    """Trains BERTopic on the tickets' stored embeddings and saves the run.

    The run is saved here first, then uploaded (storage() gives the
    model_storage.Config, or None when uploads are off) and recorded in
    runs (bertopic_run_log). Each of those that fails becomes a warning on
    the run, as does a comparison LDA run that can't be read, so a trained
    model is never lost. The returned run has `warnings` and, when
    options.assign is set, `assignment`.

    Raises:
        ValueError: no tickets have a current embedding, or too few to find topics.
    """
    status("Reading tickets and their stored embeddings...")
    corpus = bertopic_model.load_corpus(tickets, options.outcomes)
    if not corpus.ids:
        if options.outcomes:
            raise ValueError(
                "No tickets with the chosen outcomes "
                f"({bertopic_model.describe_outcomes(options.outcomes)}) have an embedding "
                "made from their current text."
            )
        raise ValueError(
            "No tickets have an embedding made from their current text. "
            "Embed them first: cd src && python -m embedding.embeddings backfill"
        )
    status(f"Lemmatizing {len(corpus.ids)} tickets...")
    tokens = tokenize(nlp, stop_words, corpus.texts)
    status(f"Clustering {len(corpus.ids)} tickets...")
    result = bertopic_model.train(corpus, tokens, options.settings())

    status("Scoring the topics...")
    warnings: List[str] = []
    comparison = None
    if options.compare_lda:
        try:
            comparison = bertopic_model.compare_lda(
                options.compare_lda, tokens, lda_dir
            )
        except (OSError, ValueError, KeyError) as e:
            warnings.append(f"LDA run {options.compare_lda} could not be compared: {e}")
    scores = bertopic_model.metrics(result, tokens)

    status("Saving the model and its charts...")
    run = bertopic_model.save_run(
        result,
        corpus,
        topic_store.preprocessing_info(nlp, stop_words),
        scores,
        comparison,
        base_dir=base_dir,
        outcomes=options.outcomes,
    )
    run["warnings"] = warnings
    upload(run, storage, status)
    if runs is not None:
        status("Recording the run in MongoDB...")
        try:
            record_run(runs, run)
        except Exception as e:
            logger.exception(
                f"Could not record BERTopic run {run['run_id']} in MongoDB: {e}"
            )
            run["warnings"].append(
                f"Saved locally, but MongoDB could not be updated: {e}"
            )
    if options.assign:
        status(
            f"Writing `{bertopic_model.ASSIGNMENT_FIELD}` to {len(corpus.ids)} tickets..."
        )
        run["assignment"] = _assign(tickets, run, corpus.ids, result.topics, runs)
    return run


def _assign(tickets, run: dict, ids, topics, runs: Optional[Collection]) -> dict:
    """Assigns topics and records how it went (run.json, and the run log when
    there is one), adding a warning to the run unless it completed."""
    try:
        summary = bertopic_model.assign_topics(tickets, run, ids, topics)
    except bertopic_model.AssignmentRunning as e:
        summary = {
            "status": "not_started",
            "total": len(ids),
            "written": 0,
            "assigned": 0,
            "outliers": 0,
            "error": str(e),
        }
    if summary["status"] != "complete":
        run["warnings"].append(
            bertopic_model.describe_assignment(summary, run["run_id"])
        )
    run["assignment"] = summary
    try:
        topic_store.write_run_file(run)
        if runs is not None:
            runs.update_one(
                {"run_id": run["run_id"]}, {"$set": {"assignment": summary}}
            )
    except Exception as e:  # the assignment itself is done; only the record failed
        logger.warning(f"Could not record the assignment of run {run['run_id']}: {e}")
    return summary


# --- persistence ------------------------------------------------------------------


def storage_config(cfg):
    """The LDA storage settings with BERTopic's own folder in the bucket."""
    return dataclasses.replace(cfg, prefix=f"{cfg.prefix}{STORAGE_SUBFOLDER}")


def upload(
    run: dict,
    storage: Callable[[], Optional[object]],
    status: Callable[[str], None] = lambda m: None,
    client=None,
) -> None:
    """Uploads the run when storage is configured; a failure becomes a warning on the run.

    Like the app's LDA upload: the model is already saved here, and
    `python -m core.model_storage upload --bertopic RUN_ID` retries it.
    """
    from core import model_storage

    run.setdefault("warnings", [])
    try:
        cfg = storage()
    except model_storage.StorageNotConfigured as e:
        run["warnings"].append(f"Not uploaded to S3: {e}")
        return
    if cfg is None:
        return
    cfg = storage_config(cfg)
    status(f"Uploading the model to s3://{cfg.bucket}/{cfg.prefix}...")
    try:
        model_storage.upload_and_record(run, cfg, client)
    except Exception as e:
        logger.exception(f"Could not upload BERTopic run {run['run_id']} to S3: {e}")
        run["warnings"].append(
            f"Saved locally, but the upload to s3://{cfg.bucket} failed: {e}. "
            f"Retry with: python -m core.model_storage upload --bertopic {run['run_id']}"
        )


def _document(run: dict) -> dict:
    """What Mongo keeps of a run: its details, not this computer's transient notes."""
    return {k: v for k, v in run.items() if k not in ("_id", "warnings", "assignment")}


def record_run(runs: Collection, run: dict) -> None:
    """Stores the run's details (not its files) in bertopic_run_log, marked as BERTopic."""
    runs.replace_one(
        {"run_id": run["run_id"]}, {**_document(run), "method": METHOD}, upsert=True
    )


# --- saved runs -------------------------------------------------------------------


def list_runs(
    base_dir: Optional[pathlib.Path] = None, runs: Optional[Collection] = None
) -> List[dict]:
    """BERTopic runs, newest first: from Mongo when given, else those saved on this computer."""
    if runs is not None:
        return list(runs.find({"method": METHOD}, {"_id": 0}).sort("ran_at", -1))
    base_dir = base_dir or bertopic_model.model_dir()
    if not base_dir.exists():
        return []
    found = (
        topic_store._read_run_file(p)
        for p in base_dir.glob(f"*/{topic_store.RUN_FILE}")
    )
    runs_found = [r for r in found if r and r.get("method") == METHOD]
    return sorted(runs_found, key=lambda r: str(r.get("ran_at")), reverse=True)


def known_runs(
    runs: Optional[Collection] = None, base_dir: Optional[pathlib.Path] = None
) -> List[dict]:
    """Runs from Mongo (shared between computers) and this computer's disk, by run_id, newest first.

    As topic_store.known_runs: Mongo's record wins, but a local upload record
    survives if Mongo missed it.
    """
    local = {run["run_id"]: run for run in list_runs(base_dir)}
    merged = dict(local)
    if runs is not None:
        for run in list_runs(runs=runs):
            if not run.get("storage") and (local.get(run["run_id"]) or {}).get(
                "storage"
            ):
                run["storage"] = local[run["run_id"]]["storage"]
            merged[run["run_id"]] = run
    return sorted(merged.values(), key=lambda r: str(r.get("ran_at")), reverse=True)


def where(run: dict, base_dir: Optional[pathlib.Path] = None) -> str:
    if has_local_files(run, base_dir):
        return "on this computer"
    return (
        f"in {run['storage']['url']}"
        if run.get("storage")
        else "in MongoDB only (its files aren't available)"
    )


def lda_runs(base_dir: Optional[pathlib.Path] = None) -> List[dict]:
    """LDA runs a BERTopic run can be compared with: those whose run.json is on this computer."""
    return [
        run for run in topic_store.list_runs(base_dir=base_dir) if run.get("topics")
    ]


def fmt_score(value: Optional[float]) -> str:
    return f"{value:.3f}" if value is not None else "–"


def describe_run(run: dict) -> str:
    ran_at = run.get("ran_at")
    ran_at = (
        ran_at.strftime("%Y-%m-%d %H:%M")
        if isinstance(ran_at, datetime)
        else str(ran_at)
    )
    m = run.get("metrics") or {}
    return (
        f"{run['run_id']} · {ran_at} UTC · {m.get('num_topics', '?')} topics · "
        f"c_v {fmt_score(m.get('coherence_c_v'))} · diversity {fmt_score(m.get('topic_diversity'))} · "
        f"{m.get('ticket_count', '?')} tickets"
    )


def comparison_rows(run: dict) -> List[Tuple[str, str, str, str]]:
    """(model, topics, c_v, diversity) for this run and, if any, the LDA run it was compared with."""
    m = run.get("metrics") or {}
    rows = [
        (
            "BERTopic",
            str(m.get("num_topics", "?")),
            fmt_score(m.get("coherence_c_v")),
            fmt_score(m.get("topic_diversity")),
        )
    ]
    c = run.get("comparison")
    if c:
        rows.append(
            (
                f"LDA {c['run_id']}",
                str(c.get("num_topics", "?")),
                fmt_score(c.get("coherence_c_v")),
                fmt_score(c.get("topic_diversity")),
            )
        )
    return rows


def summary_text(run: dict) -> str:
    """One paragraph on what the run was trained on and how many tickets it left out."""
    m = run.get("metrics") or {}
    embedding = run.get("embedding") or {}
    skipped = (run.get("dataset") or {}).get("skipped") or {}
    text = (
        f"{m.get('ticket_count', '?')} tickets embedded with {embedding.get('model', '?')} "
        f"({embedding.get('dimensions', '?')} dimensions). Outcomes: "
        f"{bertopic_model.describe_outcomes((run.get('dataset') or {}).get('outcomes'))}. "
        f"Outliers: {m.get('outliers_before', '?')} before reassignment, "
        f"{m.get('outliers_after', '?')} after."
    )
    if skipped:
        text += (
            " Left out: "
            + ", ".join(f"{n} {why.replace('_', ' ')}" for why, n in skipped.items())
            + "."
        )
    return text


def format_topics(run: dict) -> str:
    """Each topic, largest first: its words, then (for runs that have them) its
    outcomes and example ticket IDs."""
    topics = run.get("topics") or []
    if not topics:
        return "No topics were found."
    lines = []
    for rank, t in enumerate(topics, start=1):
        lines.append(
            f"{rank}. {t['share']:.0%} of tickets ({t['count']}): {', '.join(t['words'])}"
        )
        if t.get("outcomes"):
            lines.append(
                f"   Outcomes: {bertopic_model.format_outcome_counts(t['outcomes'])}"
            )
        if t.get("examples"):
            lines.append(f"   Example tickets: {', '.join(t['examples'])}")
    return "\n".join(lines)


# --- topic list for curation ----------------------------------------------------------

TOPIC_LIST_COLUMNS = (
    "rank",
    "topic_id",
    "tickets",
    "share",
    "outcomes",
    "top_words",
    "example_tickets",
    "example_subjects",
    "proposed_name",
    "decision",
)


# A spreadsheet runs a cell starting with these as a formula.
FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def spreadsheet_safe(text: str) -> str:
    """text that a spreadsheet shows as text: one starting like a formula gets a
    leading apostrophe. Ticket subjects and words are customer-written."""
    return "'" + text if text.startswith(FORMULA_PREFIXES) else text


def example_subjects(tickets: Optional[Collection], run: dict) -> Dict[str, str]:
    """{zendesk_id: subject} for the run's example tickets, read from Mongo; {} without it.

    Subjects come from Mongo, not the run, because a saved run holds no ticket text.
    """
    ids = sorted({i for t in run.get("topics") or [] for i in t.get("examples") or []})
    if tickets is None or not ids:
        return {}
    return {
        str(doc["zendesk_id"]): doc.get("subject") or ""
        for doc in tickets.find(
            {"zendesk_id": {"$in": ids}}, {"zendesk_id": 1, "subject": 1}
        )
    }


def topic_list_csv(run: dict, subjects: Optional[Dict[str, str]] = None) -> str:
    """The run's topics as CSV for a person to curate: one row per topic, largest first.

    proposed_name and decision are left blank to fill in (e.g. keep, merge
    into N, drop). Subjects appear when given (see example_subjects). Text
    from tickets is made spreadsheet-safe (spreadsheet_safe), so opening the
    file can't run a formula a customer wrote into a subject.
    """
    subjects = subjects or {}
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(TOPIC_LIST_COLUMNS)
    for rank, t in enumerate(run.get("topics") or [], start=1):
        examples = t.get("examples") or []
        writer.writerow(
            [
                rank,
                t["topic_id"],
                t["count"],
                f"{t['share']:.1%}",
                spreadsheet_safe(
                    bertopic_model.format_outcome_counts(t.get("outcomes") or {})
                ),
                spreadsheet_safe(", ".join(t["words"])),
                spreadsheet_safe(", ".join(examples)),
                " | ".join(
                    spreadsheet_safe(subjects[i]) for i in examples if subjects.get(i)
                ),
                "",
                "",
            ]
        )
    return out.getvalue()


def local_dir(run: dict, base_dir: Optional[pathlib.Path] = None) -> pathlib.Path:
    """<BERTOPIC_MODEL_DIR>/<run_id>; the run's recorded model_dir is never trusted.

    Raises:
        ValueError: the run ID isn't a plain folder name.
    """
    from core import model_storage

    base = (base_dir or bertopic_model.model_dir()).resolve()
    folder = (base / model_storage.check_run_id(run["run_id"])).resolve()
    if folder.parent != base:
        raise ValueError(f"Refusing run {run['run_id']!r}: it would be outside {base}.")
    return folder


def _complete(folder: pathlib.Path) -> bool:
    return (folder / bertopic_model.MODEL_FILE).is_file() and (
        folder / topic_store.RUN_FILE
    ).is_file()


def has_local_files(run: dict, base_dir: Optional[pathlib.Path] = None) -> bool:
    """True if the run's model and run.json are in this computer's BERTopic folder."""
    try:
        return _complete(local_dir(run, base_dir))
    except ValueError:
        return False


def ensure_local(
    run: dict,
    base_dir: Optional[pathlib.Path] = None,
    storage: Callable[[], Optional[object]] = None,
    client=None,
) -> dict:
    """Makes the run's files available in <BERTOPIC_MODEL_DIR>/<run_id>, downloading them if needed.

    As topic_store.ensure_local: the upload is verified against
    MODEL_SIGNING_KEY before anything is installed. Updates the run's paths
    to that folder.

    Raises:
        FileNotFoundError: the files aren't here and can't be downloaded.
        ValueError: the run ID isn't a plain folder name.
        model_storage.UntrustedRun: the upload failed verification.
    """
    from core import model_storage

    local = local_dir(run, base_dir)
    if not _complete(local):
        if not run.get("storage"):
            raise FileNotFoundError(
                f"BERTopic run {run['run_id']}'s files aren't on this computer and it was "
                "never uploaded."
            )
        cfg = (storage or model_storage.config)()
        if cfg is None:
            raise FileNotFoundError(
                f"BERTopic run {run['run_id']} isn't on this computer. It was uploaded to {run['storage'].get('url')}; "
                "set the model storage settings to download it."
            )
        model_storage.download_run(run, storage_config(cfg), local, client)
        if not _complete(local):
            raise FileNotFoundError(
                f"BERTopic run {run['run_id']}'s upload has no {bertopic_model.MODEL_FILE}."
            )
    topic_store._relocate(run, local)
    return run


def chart_paths(
    run: dict, base_dir: Optional[pathlib.Path] = None
) -> Dict[str, Optional[str]]:
    """The run's HTML charts in its own folder; None for a chart that wasn't drawn or is gone."""
    try:
        folder = local_dir(run, base_dir)
    except ValueError:
        return {name: None for name in CHARTS}
    recorded = run.get("charts") or {}
    paths = {}
    for name in CHARTS:
        path = folder / f"{name.removesuffix('_html')}.html"
        paths[name] = str(path) if recorded.get(name) and path.is_file() else None
    return paths
