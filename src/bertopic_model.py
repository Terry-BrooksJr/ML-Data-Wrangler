"""BERTopic topic modelling on the Voyage embeddings stored on each ticket.

An alternative to LDA (LDA_logic) suited to short tickets: BERTopic groups
tickets by meaning, clustering their embeddings (UMAP, then HDBSCAN), and
names each group by the words that set it apart (c-TF-IDF). It reuses the
embeddings the backfill stored (see embeddings.py), so a run makes no API
calls and needs no PyTorch. Only tickets whose embedding matches their
current text are used; run the backfill first to bring the rest up to date.

Topic words come from the same lemmatized tokens LDA uses (LDA_logic.tokenize),
and a run is scored the same way, c_v coherence, plus topic diversity and the
share of tickets HDBSCAN left as outliers. --compare-lda rescores a saved LDA
run's topics on this run's tokens, so the two coherences are comparable.

Runs are saved under BERTOPIC_MODEL_DIR (default bertopic_models/), apart
from the LDA runs in TOPIC_MODEL_DIR; run.json's `"method": "bertopic"` also
keeps topic_store.list_runs from loading one as an LDA model. The app trains
and shows runs through bertopic_runs. With --assign, each ticket's topic is written to `embedding_topic`,
never to `topic`, which holds the LDA assignment that feedback reports use.

A run can be limited to tickets with chosen Zendesk outcomes (the
`historical_outcome` the wrangler stores), e.g. only Client Education, to
find the themes inside one outcome. Each saved topic lists how many of its
tickets have each outcome, and the IDs of the tickets nearest its centre;
no ticket text is saved with a run.

    cd src && python -m bertopic_model [--min-cluster-size N] [--nr-topics N]
        [--keep-outliers] [--seed N] [--compare-lda RUN_ID] [--assign]
        [--outcome OUTCOME ...]   (--outcome none for tickets with no outcome)
"""

import argparse
import json
import os
import pathlib
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

import embeddings
import numpy as np
import topic_store
from loguru import logger
from pymongo.collection import Collection

ASSIGNMENT_FIELD = "embedding_topic"
MODEL_FILE = "bertopic.pickle"
ASSIGNMENTS_FILE = "assignments.json"
OUTLIER = -1
TOP_WORDS = 10
DEFAULT_SEED = 42
DEFAULT_N_NEIGHBORS = 15
DEFAULT_N_COMPONENTS = 5
# HDBSCAN's smallest topic: 1% of the tickets, but never below this.
MIN_CLUSTER_SIZE_FLOOR = 10
# The Zendesk ticket outcome, as the wrangler stores it on each ticket.
OUTCOME_FIELD = "historical_outcome"
# Tickets per topic listed as its examples: those nearest the topic's centre.
EXAMPLES_PER_TOPIC = 5


def model_dir() -> pathlib.Path:
    return pathlib.Path(
        os.environ.get("BERTOPIC_MODEL_DIR", "bertopic_models")
    ).resolve()


# --- corpus -----------------------------------------------------------------------


@dataclass
class Corpus:
    """Tickets with a current embedding, all from one embedding model, in the same order."""

    ids: List[str]
    texts: List[str]
    vectors: np.ndarray
    # endpoint, model and dimensions shared by every vector.
    embedding: dict
    # Why the other tickets were left out: no_text, no_embedding, stale, other_model.
    skipped: Dict[str, int] = field(default_factory=dict)
    # Each ticket's Zendesk outcome (None when unset), in the same order as ids.
    outcomes: List[Optional[str]] = field(default_factory=list)


def outcome_query(outcomes: Optional[Sequence[Optional[str]]]) -> dict:
    """The Mongo filter for tickets with any of outcomes; None (or empty) means every ticket.

    None in outcomes matches tickets whose outcome is unset or missing.
    """
    if not outcomes:
        return {}
    return {OUTCOME_FIELD: {"$in": list(outcomes)}}


def outcome_counts(tickets: Collection) -> List[Tuple[Optional[str], int]]:
    """(outcome, tickets) for every outcome in the collection, most tickets first; None is unset."""
    counts = Counter(
        doc.get(OUTCOME_FIELD) for doc in tickets.find({}, {OUTCOME_FIELD: 1})
    )
    return sorted(counts.items(), key=lambda item: (-item[1], item[0] or ""))


def load_corpus(
    tickets: Collection, outcomes: Optional[Sequence[Optional[str]]] = None
) -> Corpus:
    """Reads the tickets whose stored embedding was made from their current text.

    Vectors from different models (or dimensions) can't be clustered together,
    so only the model most tickets were embedded with is used.

    Args:
        outcomes: Only tickets with one of these outcomes (None for unset);
            None or empty reads every ticket.
    """
    skipped: Counter = Counter()
    by_model: Dict[tuple, list] = {}
    projection = {
        "zendesk_id": 1,
        "subject": 1,
        "transcript": 1,
        "embedding": 1,
        "embedding_meta": 1,
        OUTCOME_FIELD: 1,
    }
    for doc in tickets.find(outcome_query(outcomes), projection):
        text = embeddings.TICKET_TEXT(doc)
        meta = doc.get("embedding_meta") or {}
        vector = doc.get("embedding")
        if not text:
            skipped["no_text"] += 1
        elif not vector or not meta:
            skipped["no_embedding"] += 1
        elif meta.get("text_sha256") != embeddings.text_hash(text) or len(
            vector
        ) != meta.get("dimensions"):
            skipped["stale"] += 1
        else:
            key = (meta.get("endpoint"), meta.get("model"), meta.get("dimensions"))
            by_model.setdefault(key, []).append(
                (str(doc.get("zendesk_id")), text, vector, doc.get(OUTCOME_FIELD))
            )
    if not by_model:
        return Corpus([], [], np.empty((0, 0), dtype=np.float32), {}, dict(skipped))
    key, rows = max(by_model.items(), key=lambda item: len(item[1]))
    skipped["other_model"] = sum(len(r) for k, r in by_model.items() if k != key)
    ids, texts, vectors, ticket_outcomes = zip(*rows)
    return Corpus(
        ids=list(ids),
        texts=list(texts),
        vectors=np.asarray(vectors, dtype=np.float32),
        embedding=dict(zip(("endpoint", "model", "dimensions"), key)),
        skipped={k: v for k, v in skipped.items() if v},
        outcomes=list(ticket_outcomes),
    )


# --- training ---------------------------------------------------------------------


@dataclass
class Settings:
    """What a run was trained with; recorded in run.json."""

    min_cluster_size: Optional[int] = (
        None  # None: 1% of the tickets, at least MIN_CLUSTER_SIZE_FLOOR
    )
    n_neighbors: int = DEFAULT_N_NEIGHBORS
    n_components: int = DEFAULT_N_COMPONENTS
    # Merge similar topics down to this many (None keeps what HDBSCAN found).
    nr_topics: Optional[int] = None
    # Move each outlier to the topic its embedding is closest to.
    reduce_outliers: bool = True
    seed: int = DEFAULT_SEED

    def cluster_size(self, n_docs: int) -> int:
        return self.min_cluster_size or max(MIN_CLUSTER_SIZE_FLOOR, n_docs // 100)


@dataclass
class Result:
    model: object  # BERTopic
    topics: List[int]  # per ticket, in Corpus order; OUTLIER for none
    outliers_before: int  # left unclustered by HDBSCAN, before any reassignment
    settings: Settings


def _vectorizer():
    from sklearn.feature_extraction.text import CountVectorizer

    # Documents are already lemmatized, stopword-free tokens joined by spaces.
    return CountVectorizer(token_pattern=r"\S+", lowercase=False)


def build_model(settings: Settings, n_docs: int):
    """A BERTopic with seeded UMAP and no embedding model: vectors are always passed in."""
    from bertopic import BERTopic
    from bertopic.vectorizers import ClassTfidfTransformer
    from hdbscan import HDBSCAN
    from umap import UMAP

    return BERTopic(
        umap_model=UMAP(
            n_neighbors=min(settings.n_neighbors, n_docs - 1),
            n_components=settings.n_components,
            min_dist=0.0,
            metric="cosine",
            random_state=settings.seed,
        ),
        hdbscan_model=HDBSCAN(
            min_cluster_size=settings.cluster_size(n_docs),
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        ),
        vectorizer_model=_vectorizer(),
        ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True),
        nr_topics=settings.nr_topics,
        top_n_words=TOP_WORDS,
    )


def train(
    corpus: Corpus, tokens: Sequence[Sequence[str]], settings: Settings
) -> Result:
    """Fits BERTopic on the corpus's stored vectors, with topic words from tokens.

    Raises:
        ValueError: too few tickets, or HDBSCAN found no topics.
    """
    n_docs = len(corpus.ids)
    if n_docs <= settings.n_components + 1 or n_docs < 2 * settings.cluster_size(
        n_docs
    ):
        raise ValueError(
            f"{n_docs} tickets with a current embedding is too few for topics of at least "
            f"{settings.cluster_size(n_docs)} tickets. Embed more tickets or lower the minimum topic size."
        )
    documents = [" ".join(t) for t in tokens]
    model = build_model(settings, n_docs)
    topics, _ = model.fit_transform(documents, corpus.vectors)
    topics = [int(t) for t in topics]
    outliers_before = topics.count(OUTLIER)
    if outliers_before == n_docs:
        raise ValueError(
            "Every ticket was left as an outlier: no topics were found. Lower the minimum topic size."
        )
    if settings.reduce_outliers and outliers_before:
        topics = [
            int(t)
            for t in model.reduce_outliers(
                documents, topics, strategy="embeddings", embeddings=corpus.vectors
            )
        ]
        # Recompute topic words with the reassigned tickets included.
        model.update_topics(
            documents,
            topics=topics,
            vectorizer_model=_vectorizer(),
            ctfidf_model=model.ctfidf_model,
            top_n_words=TOP_WORDS,
        )
    logger.info(
        f"BERTopic: {len(set(topics) - {OUTLIER})} topics from {n_docs} tickets, "
        f"{outliers_before} outliers before reassignment"
    )
    return Result(model, topics, outliers_before, settings)


def topic_summaries(result: Result) -> List[dict]:
    """Each topic's size and top words, largest first. Outliers aren't a topic."""
    counts = Counter(result.topics)
    total = len(result.topics) or 1
    summaries = [
        {
            "topic_id": topic,
            "count": count,
            "share": count / total,
            "words": [word for word, _ in (result.model.get_topic(topic) or [])][
                :TOP_WORDS
            ],
        }
        for topic, count in counts.items()
        if topic != OUTLIER
    ]
    return sorted(summaries, key=lambda s: (-s["count"], s["topic_id"]))


def add_topic_details(
    summaries: List[dict], result: Result, corpus: Corpus, n: int = EXAMPLES_PER_TOPIC
) -> List[dict]:
    """Adds each topic's outcome counts and example ticket IDs to its summary.

    `outcomes`: {outcome: tickets} for the topic, largest first, with an unset
    outcome under "unset". `examples`: the IDs of up to n of the topic's
    tickets whose embeddings are nearest its centre, nearest first; IDs only,
    so a saved run holds no ticket text.
    """
    vectors = np.asarray(corpus.vectors, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit = np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 0)
    for summary in summaries:
        rows = [i for i, t in enumerate(result.topics) if t == summary["topic_id"]]
        counts = Counter(
            (corpus.outcomes[i] if i < len(corpus.outcomes) else None) or "unset"
            for i in rows
        )
        summary["outcomes"] = dict(counts.most_common())
        centre = unit[rows].mean(axis=0)
        nearest = sorted(rows, key=lambda i: -float(unit[i] @ centre))[:n]
        summary["examples"] = [corpus.ids[i] for i in nearest]
    return summaries


# --- scoring ----------------------------------------------------------------------


def topic_diversity(
    topic_words: Sequence[Sequence[str]], topn: int = TOP_WORDS
) -> Optional[float]:
    """Share of the topics' top words that are unique across topics (1.0: no overlap)."""
    words = [w for topic in topic_words for w in list(topic)[:topn]]
    return len(set(words)) / len(words) if words else None


def coherence(
    topic_words: Sequence[Sequence[str]],
    tokens: Sequence[Sequence[str]],
    topn: int = TOP_WORDS,
) -> Optional[float]:
    """c_v coherence of the topics against tokens, as LDA_logic scores its models.

    Words absent from tokens are dropped (a saved LDA run's words come from a
    different text), as are topics left with fewer than two words.
    """
    from gensim.corpora.dictionary import Dictionary
    from gensim.models import CoherenceModel

    dictionary = Dictionary(tokens)
    known = [
        [w for w in list(topic)[:topn] if w in dictionary.token2id]
        for topic in topic_words
    ]
    known = [topic for topic in known if len(topic) >= 2]
    if not known:
        return None
    return float(
        CoherenceModel(
            topics=known,
            texts=[list(t) for t in tokens],
            dictionary=dictionary,
            coherence="c_v",
            topn=topn,
            processes=1,
        ).get_coherence()
    )


def metrics(result: Result, tokens: Sequence[Sequence[str]]) -> dict:
    words = [s["words"] for s in topic_summaries(result)]
    return {
        "num_topics": len(words),
        "coherence_c_v": coherence(words, tokens),
        "topic_diversity": topic_diversity(words),
        "outliers_before": result.outliers_before,
        "outliers_after": result.topics.count(OUTLIER),
        "ticket_count": len(result.topics),
    }


def lda_topic_words(
    run_id: str, base_dir: Optional[pathlib.Path] = None
) -> List[List[str]]:
    """The topics of a saved LDA run, from its run.json.

    Raises:
        FileNotFoundError: no such run in TOPIC_MODEL_DIR.
    """
    path = (base_dir or topic_store.model_dir()) / run_id / topic_store.RUN_FILE
    run = json.loads(path.read_text(encoding="utf-8"))
    return [topic["words"] for topic in run.get("topics") or []]


def compare_lda(
    run_id: str,
    tokens: Sequence[Sequence[str]],
    base_dir: Optional[pathlib.Path] = None,
) -> dict:
    """A saved LDA run's topics scored on this corpus's tokens."""
    words = lda_topic_words(run_id, base_dir)
    return {
        "run_id": run_id,
        "num_topics": len(words),
        "coherence_c_v": coherence(words, tokens),
        "topic_diversity": topic_diversity(words),
    }


# --- saving and assigning ---------------------------------------------------------


def save_run(
    result: Result,
    corpus: Corpus,
    tokens_info: dict,
    scores: dict,
    comparison: Optional[dict] = None,
    base_dir: Optional[pathlib.Path] = None,
    now: Optional[datetime] = None,
    outcomes: Optional[Sequence[Optional[str]]] = None,
) -> dict:
    """Saves the model, per-ticket topics, charts and run.json; returns the run's details.

    outcomes: the outcomes the corpus was limited to (None: every ticket),
    recorded under dataset.outcomes.

    The model is pickled (BERTopic's other formats need PyTorch or
    safetensors). Load it only from a run directory you made.
    """
    import bertopic

    now = now or datetime.now(timezone.utc)
    run_id = topic_store.new_run_id(now)
    run_dir = (base_dir or model_dir()) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    result.model.save(
        str(run_dir / MODEL_FILE), serialization="pickle", save_embedding_model=False
    )
    (run_dir / ASSIGNMENTS_FILE).write_text(
        json.dumps(dict(zip(corpus.ids, result.topics))), encoding="utf-8"
    )
    run = {
        "run_id": run_id,
        "ran_at": now,
        "method": "bertopic",
        "bertopic_version": bertopic.__version__,
        "settings": {
            **asdict(result.settings),
            "min_cluster_size": result.settings.cluster_size(len(corpus.ids)),
        },
        "embedding": corpus.embedding,
        "preprocessing": tokens_info,
        "dataset": {
            **topic_store.dataset_fingerprint(corpus.texts, corpus.ids),
            "skipped": corpus.skipped,
            "outcomes": list(outcomes) if outcomes else None,
        },
        "metrics": scores,
        "comparison": comparison,
        "topics": add_topic_details(topic_summaries(result), result, corpus),
        "model_dir": str(run_dir),
        "charts": save_charts(result, run_dir),
    }
    topic_store.write_run_file(run)
    logger.success(f"Saved BERTopic run {run_id} to {run_dir}")
    return run


def save_charts(result: Result, run_dir: pathlib.Path) -> Dict[str, Optional[str]]:
    """Plotly charts of the topics as HTML; a chart that can't be drawn is None.

    The plotly script loads from its CDN, so the files stay small.
    """
    charts: Dict[str, Optional[str]] = {}
    n_topics = len(set(result.topics) - {OUTLIER})
    for name, draw in (
        (
            "barchart_html",
            lambda: result.model.visualize_barchart(top_n_topics=min(n_topics, 12)),
        ),
        ("intertopic_html", lambda: result.model.visualize_topics()),
    ):
        path = run_dir / f"{name.removesuffix('_html')}.html"
        try:
            draw().write_html(str(path), include_plotlyjs="cdn")
            charts[name] = str(path)
        except Exception as e:  # visualize_topics fails with too few topics
            logger.warning(f"Couldn't draw the {name} chart: {e}")
            charts[name] = None
    return charts


def assign_topics(
    tickets: Collection,
    run: dict,
    ids: Sequence[str],
    topics: Sequence[int],
    now: Optional[datetime] = None,
) -> dict:
    """Writes each ticket's topic under the run to `embedding_topic`.

    Outliers are written too (topic -1, no words), so it's clear the run
    looked at them. Only existing tickets are updated.

    Returns:
        Counts: assigned, outliers, not_in_mongo.
    """
    now = now or datetime.now(timezone.utc)
    words = {t["topic_id"]: t["words"][:5] for t in run["topics"]}
    summary = {"assigned": 0, "outliers": 0, "not_in_mongo": 0}
    for zendesk_id, topic in zip(ids, topics):
        value = {
            "run_id": run["run_id"],
            "topic": int(topic),
            "words": words.get(topic, []),
            "assigned_at": now,
        }
        if not tickets.update_one(
            {"zendesk_id": zendesk_id}, {"$set": {ASSIGNMENT_FIELD: value}}
        ).matched_count:
            summary["not_in_mongo"] += 1
        else:
            summary["outliers" if topic == OUTLIER else "assigned"] += 1
    logger.info(f"BERTopic assignment for run {run['run_id']}: {summary}")
    return summary


# --- report -----------------------------------------------------------------------


def format_report(run: dict, top: int = 15) -> str:
    m = run["metrics"]
    fmt = lambda v: f"{v:.3f}" if v is not None else "-"
    lines = [
        f"BERTopic run {run['run_id']}: {m['num_topics']} topics from {m['ticket_count']} tickets "
        f"({run['embedding'].get('model')} @ {run['embedding'].get('dimensions')})",
        f"Outliers: {m['outliers_before']} before reassignment, {m['outliers_after']} after",
        f"Outcomes: {describe_outcomes((run.get('dataset') or {}).get('outcomes'))}",
        "",
        f"{'':<22}{'topics':>8}{'c_v':>8}{'diversity':>11}",
        f"{'BERTopic':<22}{m['num_topics']:>8}{fmt(m['coherence_c_v']):>8}{fmt(m['topic_diversity']):>11}",
    ]
    if run.get("comparison"):
        c = run["comparison"]
        lines.append(
            f"{'LDA ' + c['run_id'][:18]:<22}{c['num_topics']:>8}{fmt(c['coherence_c_v']):>8}"
            f"{fmt(c['topic_diversity']):>11}"
        )
        lines.append("(both scored on this run's tickets and tokens)")
    lines.append("")
    for rank, topic in enumerate(run["topics"][:top], start=1):
        lines.append(
            f"{rank:>3}. {topic['share']:>4.0%}  ({topic['count']:>4})  {', '.join(topic['words'])}"
        )
        if topic.get("outcomes"):
            lines.append(f"{'':>7}outcomes: {format_outcome_counts(topic['outcomes'])}")
    if len(run["topics"]) > top:
        lines.append(
            f"     … {len(run['topics']) - top} more in {run['model_dir']}/run.json"
        )
    return "\n".join(lines)


def describe_outcomes(outcomes: Optional[Sequence[Optional[str]]]) -> str:
    """The outcomes a run was limited to, for display; every ticket when there's no limit."""
    if not outcomes:
        return "every ticket"
    return ", ".join(o or "unset" for o in outcomes)


def format_outcome_counts(counts: Dict[str, int]) -> str:
    return ", ".join(f"{outcome} {n}" for outcome, n in counts.items())


def _outcome_arg(value: str) -> Optional[str]:
    """--outcome value: "none" (any case) stands for tickets with no outcome."""
    return None if value.strip().lower() in ("none", "unset") else value.strip()


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {number}")
    return number


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bertopic_model",
        description="Topic-model tickets with BERTopic on their stored embeddings.",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=_positive_int,
        default=None,
        help=f"smallest topic, in tickets (default 1%% of tickets, at least {MIN_CLUSTER_SIZE_FLOOR})",
    )
    parser.add_argument(
        "--nr-topics",
        type=_positive_int,
        default=None,
        help="merge similar topics down to this many (default: keep what is found)",
    )
    parser.add_argument(
        "--n-neighbors",
        type=_positive_int,
        default=DEFAULT_N_NEIGHBORS,
        help="UMAP neighbourhood size: larger gives broader topics",
    )
    parser.add_argument(
        "--keep-outliers",
        action="store_true",
        help="leave unclustered tickets as topic -1 instead of moving them to the nearest topic",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--compare-lda",
        metavar="RUN_ID",
        help="score a saved LDA run on the same tickets",
    )
    parser.add_argument(
        "--assign",
        action="store_true",
        help=f"write each ticket's topic to its `{ASSIGNMENT_FIELD}` field in Mongo",
    )
    parser.add_argument(
        "--outcome",
        dest="outcomes",
        action="append",
        type=_outcome_arg,
        metavar="OUTCOME",
        help="only tickets with this Zendesk outcome (repeatable; 'none' for no outcome)",
    )
    return parser


def main(argv: List[str]) -> int:
    args = _parser().parse_args(argv)
    import settings

    settings.load()  # .env and the app's saved Settings (MONGO_URI etc.)
    import en_core_web_lg
    from LDA_logic import stopwords, tokenize
    from wrangler import DataWrangler

    tickets = DataWrangler.get_tickets_collection()
    corpus = load_corpus(tickets, args.outcomes)
    skipped = (
        ", ".join(f"{n} {why.replace('_', ' ')}" for why, n in corpus.skipped.items())
        or "none"
    )
    print(f"{len(corpus.ids)} tickets with a current embedding; skipped: {skipped}")
    if corpus.skipped.get("no_embedding") or corpus.skipped.get("stale"):
        print("  (python -m embeddings backfill embeds the missing and stale ones)")

    nlp = en_core_web_lg.load()
    stop_words = stopwords.words("english")
    tokens = tokenize(nlp, stop_words, corpus.texts)
    run_settings = Settings(
        min_cluster_size=args.min_cluster_size,
        n_neighbors=args.n_neighbors,
        nr_topics=args.nr_topics,
        reduce_outliers=not args.keep_outliers,
        seed=args.seed,
    )
    try:
        result = train(corpus, tokens, run_settings)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    comparison = None
    if args.compare_lda:
        try:
            comparison = compare_lda(args.compare_lda, tokens)
        except FileNotFoundError:
            print(
                f"error: no LDA run {args.compare_lda!r} in {topic_store.model_dir()}",
                file=sys.stderr,
            )
            return 2
    run = save_run(
        result,
        corpus,
        topic_store.preprocessing_info(nlp, stop_words),
        metrics(result, tokens),
        comparison,
        outcomes=args.outcomes,
    )
    print()
    print(format_report(run))
    if args.assign:
        summary = assign_topics(tickets, run, corpus.ids, result.topics)
        print(f"\nWrote `{ASSIGNMENT_FIELD}`: {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
