"""Compares embedding models on hand-labelled tickets.

For each model, every ticket in an evaluation set (see evaluation.py) is
embedded, and its nearest neighbours are checked for the same expected
classification. The model that most often puts same-category tickets
together is the better choice for "similar tickets" and for picking
classifier examples.

    cd src && python -m embedding_eval EVAL_SET.json [--providers spacy voyage openai]
        [--text opening full] [-k 5] [--show-neighbours 3] [--yes]

Voyage uses the app's embedding Settings (EMBEDDING_API_KEY, endpoint,
dimensions; --voyage-model overrides EMBEDDING_MODEL). OpenAI needs
OPENAI_API_KEY. Both are paid calls
that send ticket text to that vendor; the run asks first unless --yes.
Providers without a key are skipped. Embeddings are cached in
--cache (default embedding_cache/), so re-runs don't pay again.

Lives outside decision_engine on purpose: that package must not import an
HTTP client (see test_engine_never_references_zendesk_endpoints).
"""
import argparse
import dataclasses
import hashlib
import json
import os
import pathlib
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

import embeddings
from decision_engine.evaluation import load_eval_set
from decision_engine.guards import CLASSIFICATIONS
from decision_engine.search import terminal_safe

DEFAULT_K = 5
BATCH_SIZE = 64
TEXT_VIEWS = embeddings.TEXT_VIEWS
OPENAI_URL = "https://api.openai.com/v1/embeddings"

# An embedder returns one row per text and the tokens billed (0 if free).
Embedder = Callable[[List[str]], Tuple[np.ndarray, int]]


# --- embedders --------------------------------------------------------------------


def _api_embedder(url: str, api_key: str, model: str, extra: Optional[dict] = None) -> Embedder:
    """Embedder for an OpenAI-style /embeddings API (Voyage uses the same shape)."""

    def embed(texts: List[str]) -> Tuple[np.ndarray, int]:
        rows, tokens = [], 0
        for start in range(0, len(texts), BATCH_SIZE):
            reply = embeddings.post_json(url, api_key, {"model": model, "input": texts[start:start + BATCH_SIZE], **(extra or {})})
            data = sorted(reply["data"], key=lambda d: d["index"])
            rows.extend(d["embedding"] for d in data)
            tokens += (reply.get("usage") or {}).get("total_tokens", 0)
        return np.asarray(rows, dtype=np.float32), tokens

    return embed


def voyage_embedder(model: str) -> Embedder:
    """The app's own Voyage client and settings (endpoint, dimensions), with model overridable.

    input_type "document" for every text: each ticket is compared with every other.
    """
    cfg = dataclasses.replace(embeddings.config(), model=model)
    return lambda texts: embeddings.embed(texts, cfg, input_type="document")


def openai_embedder(model: str) -> Embedder:
    return _api_embedder(OPENAI_URL, os.environ["OPENAI_API_KEY"], model)


def spacy_embedder() -> Embedder:
    """en_core_web_lg's document vector: the average of its 300-d word vectors."""
    import en_core_web_lg

    nlp = en_core_web_lg.load()

    def embed(texts: List[str]) -> Tuple[np.ndarray, int]:
        # Static word vectors need only the tokenizer, not the pipeline.
        return np.asarray([nlp.make_doc(text).vector for text in texts], dtype=np.float32), 0

    return embed


@dataclass
class Candidate:
    name: str
    provider: str
    model: str
    key_env: Optional[str]
    make: Callable[[], Embedder]
    sends_text: bool


def candidates(voyage_model: str, openai_model: str) -> Dict[str, Candidate]:
    return {
        "spacy": Candidate("spaCy en_core_web_lg", "spacy", "en_core_web_lg", None, spacy_embedder, False),
        "voyage": Candidate(f"Voyage {voyage_model}", "voyage",
                            f"{voyage_model}@{os.environ.get('EMBEDDING_DIMENSIONS') or embeddings.DEFAULT_DIMENSIONS}",
                            "EMBEDDING_API_KEY",
                            lambda: voyage_embedder(voyage_model), True),
        "openai": Candidate(f"OpenAI {openai_model}", "openai", openai_model, "OPENAI_API_KEY",
                            lambda: openai_embedder(openai_model), True),
    }


def cached_embed(candidate: Candidate, view: str, texts: List[str], cache_dir: pathlib.Path) -> Tuple[np.ndarray, int, bool]:
    """Embeds texts, reusing a cached result for the same model and texts.

    Returns:
        (vectors, tokens billed this call, whether it came from the cache).
    """
    digest = hashlib.sha256(json.dumps([candidate.provider, candidate.model, texts]).encode("utf-8")).hexdigest()[:16]
    path = cache_dir / f"{candidate.provider}-{candidate.model.replace('/', '_')}-{view}-{digest}.npy"
    if path.exists():
        return np.load(path), 0, True
    vectors, tokens = candidate.make()(texts)
    if vectors.shape[0] != len(texts):
        raise RuntimeError(f"{candidate.name} returned {vectors.shape[0]} vectors for {len(texts)} texts")
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(path, vectors)
    return vectors, tokens, False


# --- scoring ----------------------------------------------------------------------


def neighbours(vectors: np.ndarray, k: int) -> np.ndarray:
    """Indices of each row's k most cosine-similar other rows, nearest first."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit = vectors / np.where(norms == 0, 1, norms)
    similarity = unit @ unit.T
    np.fill_diagonal(similarity, -np.inf)
    # Stable sort so ties break the same way on every run.
    return np.argsort(-similarity, axis=1, kind="stable")[:, :k]


def score(labels: Sequence[str], nearest: np.ndarray) -> dict:
    """How often neighbours share each ticket's label.

    Returns:
        top1: share of tickets whose nearest neighbour has the same label.
        precision: mean share of the k neighbours with the same label.
        knn_accuracy: share where the neighbours' majority label is right
            (ties go to the label of the nearer neighbour).
        per_category: precision for tickets of each label.
    """
    labels = list(labels)
    top1, precision, knn, per_category = [], [], [], {c: [] for c in CLASSIFICATIONS}
    for i, row in enumerate(nearest):
        neighbour_labels = [labels[j] for j in row]
        same = [label == labels[i] for label in neighbour_labels]
        top1.append(same[0])
        precision.append(sum(same) / len(same))
        per_category[labels[i]].append(precision[-1])
        counts = Counter(neighbour_labels)
        best = max(counts.values())
        vote = next(label for label in neighbour_labels if counts[label] == best)
        knn.append(vote == labels[i])
    return {
        "top1": float(np.mean(top1)),
        "precision": float(np.mean(precision)),
        "knn_accuracy": float(np.mean(knn)),
        "per_category": {c: float(np.mean(v)) if v else None for c, v in per_category.items()},
    }


def chance_precision(labels: Sequence[str]) -> float:
    """Expected precision if neighbours were picked at random (self excluded)."""
    counts = Counter(labels)
    n = len(labels)
    return sum(counts[label] - 1 for label in labels) / (n * (n - 1)) if n > 1 else 0.0


# --- report -----------------------------------------------------------------------


def format_results(view: str, labels: Sequence[str], results: List[dict], k: int) -> str:
    fmt = lambda v: f"{v:.0%}" if v is not None else "-"
    lines = [f"== Text: {view} ({len(labels)} tickets, k={k}) ==",
             f"{'model':<34}{'dims':>6}{'top-1':>8}{f'P@{k}':>8}{f'kNN@{k}':>9}{'tokens':>10}"]
    for r in results:
        tokens = "cached" if r["cached"] else (str(r["tokens"]) if r["tokens"] else "-")
        lines.append(f"{r['name']:<34}{r['dims']:>6}{fmt(r['top1']):>8}{fmt(r['precision']):>8}"
                     f"{fmt(r['knn_accuracy']):>9}{tokens:>10}")
    lines.append(f"{'random neighbours (baseline)':<34}{'':>6}{'':>8}{fmt(chance_precision(labels)):>8}")
    lines.append("")
    counts = Counter(labels)
    lines.append(f"P@{k} by category".ljust(24) + "".join(f"{r['name'][:22]:>24}" for r in results))
    for c in CLASSIFICATIONS:
        if counts[c]:
            lines.append(f"{c} ({counts[c]})".ljust(24) + "".join(f"{fmt(r['per_category'][c]):>24}" for r in results))
    return "\n".join(lines)


def format_neighbours(cases: List[dict], texts: List[str], result: dict, count: int) -> str:
    """A few tickets and their nearest neighbour, for spot-checking by eye."""
    lines = [f"-- {result['name']}: nearest neighbour of the first {count} tickets --"]
    for i in range(min(count, len(cases))):
        j = int(result["nearest"][i][0])
        mark = "same" if cases[i]["expected"] == cases[j]["expected"] else "DIFF"
        for label, index in (("ticket", i), ("nearest", j)):
            ticket = cases[index]["ticket"]
            lines.append(f"  {label:<8}{terminal_safe(ticket.get('zendesk_id')):>10}  {cases[index]['expected']:<20}"
                         f"{terminal_safe(texts[index])[:90]}")
        lines.append(f"  -> {mark}")
    return "\n".join(lines)


def _confirm(names: List[str], count: int) -> bool:
    print(f"This sends the text of {count} tickets to: {', '.join(names)} (paid API calls).")
    if not sys.stdin.isatty():
        print("Not a terminal: pass --yes to confirm.", file=sys.stderr)
        return False
    return input("Continue? [y/N] ").strip().lower() in ("y", "yes")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m embedding_eval",
                                     description="Compare embedding models on a labelled evaluation set.")
    parser.add_argument("eval_set", help="JSON evaluation set (see decision_engine.evaluation)")
    parser.add_argument("--providers", nargs="+", choices=["spacy", "voyage", "openai"],
                        default=["spacy", "voyage", "openai"])
    parser.add_argument("--text", nargs="+", choices=list(TEXT_VIEWS), default=["opening"],
                        help="opening: subject + first requester comment; full: all public comments")
    parser.add_argument("-k", type=int, default=DEFAULT_K, help=f"neighbours per ticket (default {DEFAULT_K})")
    parser.add_argument("--voyage-model", default=None, help="default: EMBEDDING_MODEL, else voyage-3.5")
    parser.add_argument("--openai-model", default="text-embedding-3-small")
    parser.add_argument("--cache", default="embedding_cache", help="folder for cached embeddings")
    parser.add_argument("--show-neighbours", type=int, default=0, metavar="N",
                        help="print N tickets with their nearest neighbour per model")
    parser.add_argument("--yes", action="store_true", help="don't ask before sending ticket text to APIs")
    return parser


def main(argv: List[str]) -> int:
    args = _parser().parse_args(argv)
    import settings

    settings.load()  # .env and the app's saved Settings (EMBEDDING_API_KEY etc.)
    voyage_model = args.voyage_model or os.environ.get("EMBEDDING_MODEL") or embeddings.DEFAULT_MODEL
    cases = [case for case in load_eval_set(args.eval_set)
             if any(TEXT_VIEWS[view](case["ticket"]) for view in args.text)]
    labels = [case["expected"] for case in cases]
    if len(cases) <= args.k:
        print(f"error: need more than k={args.k} tickets with text, found {len(cases)}", file=sys.stderr)
        return 2
    lonely = [c for c, n in Counter(labels).items() if n == 1]
    if lonely:
        print(f"note: only one ticket labelled {', '.join(lonely)}; it can never have a same-label neighbour.")

    chosen = []
    for key in args.providers:
        candidate = candidates(voyage_model, args.openai_model)[key]
        if candidate.key_env and not os.environ.get(candidate.key_env):
            print(f"skipping {candidate.name}: {candidate.key_env} is not set")
            continue
        chosen.append(candidate)
    paid = [c.name for c in chosen if c.sends_text]
    if paid and not args.yes and not _confirm(paid, len(cases)):
        return 1

    cache_dir = pathlib.Path(args.cache)
    for view in args.text:
        texts = [TEXT_VIEWS[view](case["ticket"]) or " " for case in cases]
        results = []
        for candidate in chosen:
            try:
                vectors, tokens, cached = cached_embed(candidate, view, texts, cache_dir)
            except RuntimeError as e:
                print(f"{candidate.name} failed: {e}", file=sys.stderr)
                continue
            nearest = neighbours(vectors, args.k)
            results.append({"name": candidate.name, "dims": vectors.shape[1], "tokens": tokens, "cached": cached,
                            "nearest": nearest, **score(labels, nearest)})
        print()
        print(format_results(view, labels, results, args.k))
        for result in results if args.show_neighbours else []:
            print()
            print(format_neighbours(cases, texts, result, args.show_neighbours))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
