"""Training and listing BERTopic runs for the app, without any Qt.

bertopic_model does the work; this module strings its steps together for
the app's worker thread and reads back the runs it saved. BERTopic runs live
in BERTOPIC_MODEL_DIR, never TOPIC_MODEL_DIR, and their run.json says
`"method": "bertopic"`, which topic_store.list_runs skips, so the LDA lists
never try to load one as an LDA model.

Writing each ticket's topic to Mongo is opt-in (TrainOptions.assign) and
only ever touches `embedding_topic`, never the LDA `topic` field.
"""
import pathlib
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from pymongo.collection import Collection

import bertopic_model
import topic_store
from LDA_logic import tokenize

METHOD = "bertopic"
CHARTS = ("barchart_html", "intertopic_html")


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

    def settings(self) -> bertopic_model.Settings:
        return bertopic_model.Settings(min_cluster_size=self.min_cluster_size, nr_topics=self.nr_topics,
                                       reduce_outliers=not self.keep_outliers)


def train_and_save(tickets: Collection, nlp, stop_words: Sequence[str], options: TrainOptions,
                   status: Callable[[str], None] = lambda message: None,
                   base_dir: Optional[pathlib.Path] = None, lda_dir: Optional[pathlib.Path] = None) -> dict:
    """Trains BERTopic on the tickets' stored embeddings and saves the run.

    A comparison LDA run that can't be read becomes a warning, so the
    trained model isn't lost. The returned run has `warnings` and, when
    options.assign is set, `assignment`; neither is in its run.json.

    Raises:
        ValueError: no tickets have a current embedding, or too few to find topics.
    """
    status("Reading tickets and their stored embeddings...")
    corpus = bertopic_model.load_corpus(tickets)
    if not corpus.ids:
        raise ValueError("No tickets have an embedding made from their current text. "
                         "Embed them first: cd src && python -m embeddings backfill")
    status(f"Lemmatizing {len(corpus.ids)} tickets...")
    tokens = tokenize(nlp, stop_words, corpus.texts)
    status(f"Clustering {len(corpus.ids)} tickets...")
    result = bertopic_model.train(corpus, tokens, options.settings())

    status("Scoring the topics...")
    warnings: List[str] = []
    comparison = None
    if options.compare_lda:
        try:
            comparison = bertopic_model.compare_lda(options.compare_lda, tokens, lda_dir)
        except (OSError, ValueError, KeyError) as e:
            warnings.append(f"LDA run {options.compare_lda} could not be compared: {e}")
    scores = bertopic_model.metrics(result, tokens)

    status("Saving the model and its charts...")
    run = bertopic_model.save_run(result, corpus, topic_store.preprocessing_info(nlp, stop_words), scores,
                                  comparison, base_dir=base_dir)
    run["warnings"] = warnings
    if options.assign:
        status(f"Writing `{bertopic_model.ASSIGNMENT_FIELD}` to {len(corpus.ids)} tickets...")
        run["assignment"] = bertopic_model.assign_topics(tickets, run, corpus.ids, result.topics)
    return run


# --- saved runs -------------------------------------------------------------------


def list_runs(base_dir: Optional[pathlib.Path] = None) -> List[dict]:
    """BERTopic runs saved on this computer, newest first."""
    base_dir = base_dir or bertopic_model.model_dir()
    if not base_dir.exists():
        return []
    found = (topic_store._read_run_file(p) for p in base_dir.glob(f"*/{topic_store.RUN_FILE}"))
    runs = [r for r in found if r and r.get("method") == METHOD]
    return sorted(runs, key=lambda r: str(r.get("ran_at")), reverse=True)


def lda_runs(base_dir: Optional[pathlib.Path] = None) -> List[dict]:
    """LDA runs a BERTopic run can be compared with: those whose run.json is on this computer."""
    return [run for run in topic_store.list_runs(base_dir=base_dir) if run.get("topics")]


def fmt_score(value: Optional[float]) -> str:
    return f"{value:.3f}" if value is not None else "–"


def describe_run(run: dict) -> str:
    ran_at = run.get("ran_at")
    ran_at = ran_at.strftime("%Y-%m-%d %H:%M") if isinstance(ran_at, datetime) else str(ran_at)
    m = run.get("metrics") or {}
    return (f"{run['run_id']} · {ran_at} UTC · {m.get('num_topics', '?')} topics · "
            f"c_v {fmt_score(m.get('coherence_c_v'))} · diversity {fmt_score(m.get('topic_diversity'))} · "
            f"{m.get('ticket_count', '?')} tickets")


def comparison_rows(run: dict) -> List[Tuple[str, str, str, str]]:
    """(model, topics, c_v, diversity) for this run and, if any, the LDA run it was compared with."""
    m = run.get("metrics") or {}
    rows = [("BERTopic", str(m.get("num_topics", "?")), fmt_score(m.get("coherence_c_v")),
             fmt_score(m.get("topic_diversity")))]
    c = run.get("comparison")
    if c:
        rows.append((f"LDA {c['run_id']}", str(c.get("num_topics", "?")), fmt_score(c.get("coherence_c_v")),
                     fmt_score(c.get("topic_diversity"))))
    return rows


def summary_text(run: dict) -> str:
    """One paragraph on what the run was trained on and how many tickets it left out."""
    m = run.get("metrics") or {}
    embedding = run.get("embedding") or {}
    skipped = (run.get("dataset") or {}).get("skipped") or {}
    text = (f"{m.get('ticket_count', '?')} tickets embedded with {embedding.get('model', '?')} "
            f"({embedding.get('dimensions', '?')} dimensions). Outliers: {m.get('outliers_before', '?')} "
            f"before reassignment, {m.get('outliers_after', '?')} after.")
    if skipped:
        text += " Left out: " + ", ".join(f"{n} {why.replace('_', ' ')}" for why, n in skipped.items()) + "."
    return text


def format_topics(run: dict) -> str:
    """One numbered line per topic, largest first."""
    topics = run.get("topics") or []
    if not topics:
        return "No topics were found."
    return "\n".join(f"{rank}. {t['share']:.0%} of tickets ({t['count']}): {', '.join(t['words'])}"
                     for rank, t in enumerate(topics, start=1))


def local_dir(run: dict, base_dir: Optional[pathlib.Path] = None) -> pathlib.Path:
    """<BERTOPIC_MODEL_DIR>/<run_id>; the run's recorded model_dir is never trusted.

    Raises:
        ValueError: the run ID isn't a plain folder name.
    """
    import model_storage

    base = (base_dir or bertopic_model.model_dir()).resolve()
    folder = (base / model_storage.check_run_id(run["run_id"])).resolve()
    if folder.parent != base:
        raise ValueError(f"Refusing run {run['run_id']!r}: it would be outside {base}.")
    return folder


def chart_paths(run: dict, base_dir: Optional[pathlib.Path] = None) -> Dict[str, Optional[str]]:
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
