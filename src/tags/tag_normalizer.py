"""Groups ticket tags that mean the same thing, so they can be merged into one.

Tags come from a text file or pasted text: comma- or newline-separated, a
JSON / Python list, or a CSV table whose first column is the tag (a Zendesk
tag export: "name,count"; the count is ignored). Each is cleaned to plain words (underscores, hyphens and
camelCase split, lower-cased), so "car_maintenance" and "Car Maintenance"
are the same tag before any model sees them. The cleaned tags are embedded
(Voyage AI by default; spaCy offline) and grouped by average-linkage
clustering on cosine similarity: a tag joins a group only if its average
similarity to the group's tags reaches the threshold, which keeps one loose
link from chaining unrelated tags together.

Structured facet__value tags ("sentiment__positive", "intent_confidence__low")
come from a fixed taxonomy and share long prefixes, so they embed as
near-identical though they differ in meaning. Two of them are never grouped
together (unless asked); a free-form tag can still join one.

Each group's canonical tag is a structured tag if it has one, else the one most similar to the rest of the group
(ties: the more frequent, then the shorter), in its most common original
spelling, so the result maps onto tags that already exist.

Earlier decisions (tag_runs, the normalized_tags collection) carry over as a
Previous: each saved successor tag joins the grouping as an anchor, so new
synonyms join it and it keeps its name, and a tag already deprecated goes
straight to its successor's group. Anchors not in the input are never
mapped or exported.

Short tags score high against each other with Voyage (unrelated pairs often
reach 0.8), so its default threshold is 0.90; spaCy's averaged word vectors
spread wider and miss looser synonyms ("oil change" ~ "car maintenance").

    cd src && python -m tags.tag_normalizer tags.txt                 # Voyage; sends the tags only
    cd src && python -m tags.tag_normalizer tags.txt --backend spacy --threshold 0.8 --out mapping.csv
"""

import argparse
import ast
import csv
import io
import json
import pathlib
import re
import shutil
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage

VOYAGE = "voyage"
SPACY = "spacy"
# A language model that follows tag_normalization_reference.md (see tag_llm);
# it maps tags directly, so it has no vectors or threshold.
LLM = "llm"
BACKENDS = (VOYAGE, SPACY, LLM)
DEFAULT_THRESHOLDS = {VOYAGE: 0.90, SPACY: 0.80}
# Separates facet from value in tags from a fixed taxonomy, which share long
# prefixes ("intent_confidence__low", "intent_confidence__high") and so embed
# as near-identical though they mean different things.
STRUCTURED_SEPARATOR = "__"
# Longer than any real tag; a pasted paragraph is not a tag.
MAX_TAG_CHARS = 200

# An embedder returns one row per text.
Embedder = Callable[[List[str]], np.ndarray]


# --- parsing ----------------------------------------------------------------------


def _strip_quotes(item: str) -> str:
    item = item.strip()
    while len(item) >= 2 and item[0] == item[-1] and item[0] in "\"'`":
        item = item[1:-1].strip()
    return item


def _as_list(text: str) -> Optional[list]:
    """text as a JSON or Python list literal, or None if it isn't one."""
    for parse in (json.loads, ast.literal_eval):
        try:
            value = parse(text)
        except (ValueError, SyntaxError, MemoryError, RecursionError):
            continue
        if isinstance(value, (list, tuple)):
            return list(value)
    return None


_NUMBER = re.compile(r"\s*-?\d+(\.\d+)?\s*")


def _table_column(text: str) -> Optional[List[str]]:
    """The first column of a CSV table such as a Zendesk tag export ("name,count"),
    or None if text isn't one.

    A table is two or more rows of the same width, with a later column that is
    a number in every row after the first; so "a, b\\nc, d" stays four tags.
    The first row is a header unless it has a number there too.
    """
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return None
    try:
        rows = list(csv.reader(lines, skipinitialspace=True))
    except csv.Error:
        return None
    width = len(rows[0])
    if width < 2 or any(len(row) != width for row in rows):
        return None
    numeric = [
        c for c in range(1, width) if all(_NUMBER.fullmatch(row[c]) for row in rows[1:])
    ]
    if not numeric:
        return None
    if not any(_NUMBER.fullmatch(rows[0][c]) for c in numeric):
        rows = rows[1:]
    return [row[0] for row in rows]


def parse_tags(text: str) -> List[str]:
    """Splits pasted text or a file's contents into tags, in order, duplicates kept.

    Accepts a JSON or Python list (["a", "b"]), a CSV table whose first
    column is the tag (other columns, such as a count, are ignored), or tags
    separated by commas and/or newlines. Spaces inside a tag are kept: "tire
    rotation" is one tag. Surrounding quotes and brackets are dropped, as are
    empty items.
    """
    text = text.strip().lstrip("﻿")
    items = _as_list(text) if text.startswith(("[", "(")) else _table_column(text)
    if items is None:
        if text[:1] in "[(" and text[-1:] in "])":
            text = text[1:-1]
        items = re.split(r"[,\n\r]+", text)
    tags = []
    for item in items:
        if not isinstance(item, (str, int, float)):
            continue
        tag = _strip_quotes(str(item))
        if tag:
            tags.append(tag)
    return tags


_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_SEPARATORS = re.compile(r"[_\-/\\|.:;#+]+")
_NOT_WORD = re.compile(r"[^\w&' ]+")


def clean_tag(tag: str) -> str:
    """The tag as plain lower-case words: "Car_Maintenance", "car-maintenance"
    and "carMaintenance" all become "car maintenance"."""
    tag = unicodedata.normalize("NFKC", tag)
    tag = _CAMEL.sub(" ", tag)
    tag = _SEPARATORS.sub(" ", tag)
    tag = _NOT_WORD.sub(" ", tag)
    return " ".join(tag.lower().split())[:MAX_TAG_CHARS]


# --- the tag set ------------------------------------------------------------------


@dataclass
class Tag:
    """One cleaned tag and every original spelling that cleaned to it."""

    text: str
    spellings: Counter = field(default_factory=Counter)
    # As written in normalized_tags, when an earlier run saved this tag as a
    # successor (see Previous); it then names the tag's group.
    successor: Optional[str] = None

    @property
    def count(self) -> int:
        return sum(self.spellings.values())

    @property
    def spelling(self) -> str:
        """The most common original spelling (first seen wins a tie)."""
        return self.spellings.most_common(1)[0][0]

    @property
    def known(self) -> bool:
        return self.successor is not None

    @property
    def anchor_only(self) -> bool:
        """A saved successor that isn't in this run's input."""
        return self.known and self.count == 0

    @property
    def structured(self) -> bool:
        """A facet__value tag ("sentiment__positive", "intent__software__sync__issue")."""
        return any(STRUCTURED_SEPARATOR in spelling for spelling in self.spellings)


def collect(raw_tags: Iterable[str]) -> List[Tag]:
    """Distinct cleaned tags in first-seen order. Tags that clean to nothing
    ("___", "!!") are dropped."""
    tags: Dict[str, Tag] = {}
    for raw in raw_tags:
        text = clean_tag(raw)
        if text:
            tags.setdefault(text, Tag(text)).spellings[raw.strip()] += 1
    return list(tags.values())


@dataclass
class Previous:
    """What earlier runs decided (from tag_runs.previous_decisions).

    successors: cleaned text -> the successor tag as written.
    replaced: cleaned text of a deprecated tag -> its successor's cleaned text.
    """

    successors: Dict[str, str] = field(default_factory=dict)
    replaced: Dict[str, str] = field(default_factory=dict)


def with_successors(tags: List[Tag], previous: Optional[Previous]) -> List[Tag]:
    """tags plus an anchor (no uses) for each saved successor not among them.

    A tag that is a saved successor is marked known, and so is its anchor; a
    deprecated tag's successor is always present to take it.
    """
    if not previous:
        return list(tags)
    by_text = {tag.text: tag for tag in tags}
    out = list(tags)
    for text, spelling in previous.successors.items():
        tag = by_text.get(text)
        if tag is None:
            tag = Tag(text, Counter({spelling: 0}))
            by_text[text] = tag
            out.append(tag)
        tag.successor = spelling
    return out


# --- embedding ----------------------------------------------------------------------


def voyage_embedder() -> Embedder:
    """The app's Voyage client and settings (EMBEDDING_API_KEY etc.).

    Raises:
        embeddings.EmbeddingNotConfigured: if the key or another setting is missing.
    """
    from embedding import embeddings

    cfg = embeddings.config()
    return lambda texts: embeddings.embed(texts, cfg, input_type="document")[0]


_nlp = None


def spacy_embedder(nlp=None) -> Embedder:
    """en_core_web_lg's averaged word vectors; free and local, but coarser.

    Args:
        nlp: An already-loaded spaCy model (the app has one); else en_core_web_lg is loaded once.
    """
    global _nlp
    if nlp is None:
        if _nlp is None:
            import en_core_web_lg

            _nlp = en_core_web_lg.load()
        nlp = _nlp
    return lambda texts: np.asarray(
        [nlp.make_doc(text).vector for text in texts], dtype=np.float32
    )


def embedder(backend: str) -> Embedder:
    if backend == VOYAGE:
        return voyage_embedder()
    if backend == SPACY:
        return spacy_embedder()
    raise ValueError(
        f"Unknown backend {backend!r}; expected one of {', '.join(BACKENDS)}"
    )


def unit_rows(vectors: np.ndarray) -> np.ndarray:
    """Rows scaled to length 1; an all-zero row (spaCy, no known words) stays zero."""
    vectors = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 0)


# --- grouping -----------------------------------------------------------------------


@dataclass
class Group:
    """Tags to merge into `canonical` (one of members' spellings).

    members are ordered canonical first, then by similarity to it;
    similarity[i] is members[i]'s cosine similarity to the canonical tag.
    """

    canonical: str
    members: List[Tag]
    similarity: List[float]

    @property
    def count(self) -> int:
        return sum(tag.count for tag in self.members)


def _labels(
    unit: np.ndarray, threshold: float, apart: Optional[np.ndarray] = None
) -> np.ndarray:
    """Cluster label per row: average linkage on cosine distance, cut at 1 - threshold.

    Args:
        apart: Boolean per row; no two True rows end up in one cluster.
    """
    n = len(unit)
    if n < 2:
        return np.zeros(n, dtype=int)
    distance = 1.0 - np.clip(unit @ unit.T, -1.0, 1.0)
    if apart is not None:
        # Any merge spanning such a pair then averages above 2 > the cut,
        # however many other pairs are averaged in with it.
        distance[np.outer(apart, apart)] = 2.0 * n * n + 1.0
    condensed = np.clip(distance[np.triu_indices(n, k=1)], 0.0, None)
    return fcluster(
        linkage(condensed, method="average"), t=1.0 - threshold, criterion="distance"
    )


def _group(tags: List[Tag], unit: np.ndarray) -> Group:
    similarity = unit @ unit.T
    # Mean similarity to the group's other tags; the most central tag names it.
    centrality = (similarity.sum(axis=1) - np.diag(similarity)) / max(len(tags) - 1, 1)
    # A saved successor keeps naming its group; then a taxonomy tag names the
    # group over free-form ones, so tags map onto it.
    best = min(
        range(len(tags)),
        key=lambda i: (
            not tags[i].known,
            not tags[i].structured,
            -round(float(centrality[i]), 6),
            -tags[i].count,
            len(tags[i].text),
        ),
    )
    order = sorted(
        range(len(tags)), key=lambda i: (i != best, -similarity[best, i], tags[i].text)
    )
    return Group(
        tags[best].successor or tags[best].spelling,
        [tags[i] for i in order],
        [float(similarity[best, i]) for i in order],
    )


def group_tags(
    tags: List[Tag],
    vectors: np.ndarray,
    threshold: float,
    merge_structured: bool = False,
    previous: Optional[Previous] = None,
) -> List[Group]:
    """Groups tags (one row of vectors each) whose meaning matches at threshold.

    Every tag lands in exactly one group; a tag with no match is a group of
    one. Groups with most tags come first, then most uses, then alphabetical.

    Args:
        threshold: Cosine similarity, 0 to 1. Higher merges only closer synonyms.
        merge_structured: Allow two facet__value tags into one group. Off by
            default: they come from a fixed taxonomy, where tags that differ
            only in value ("sentiment__positive", "sentiment__negative") embed
            as near-identical. A free-form tag can join a structured one either way.
        previous: Earlier runs' decisions; tags should already include their
            successors (with_successors). A deprecated tag joins its successor's
            group whatever its similarity, and two saved successors are never
            grouped together (each was kept apart on purpose; rename a group
            to merge them). Groups of anchors alone are left out.
    """
    if len(tags) != len(vectors):
        raise ValueError(f"{len(tags)} tags but {len(vectors)} vectors")
    unit = unit_rows(vectors)
    replaced = previous.replaced if previous else {}
    present = {tag.text for tag in tags}
    pinned = {
        i
        for i, tag in enumerate(tags)
        if not tag.known and replaced.get(tag.text) in present
    }
    free = [i for i in range(len(tags)) if i not in pinned]
    apart = np.array(
        [(tags[i].structured and not merge_structured) or tags[i].known for i in free],
        dtype=bool,
    )
    labels = _labels(unit[free], threshold, apart if apart.any() else None)
    clusters: Dict[int, List[int]] = {}
    for row, label in zip(free, labels):
        clusters.setdefault(int(label), []).append(row)
    home = {tags[row].text: label for label, rows in clusters.items() for row in rows}
    for row in sorted(pinned):
        clusters[home[replaced[tags[row].text]]].append(row)
    groups = [
        _group([tags[i] for i in rows], unit[rows])
        for rows in clusters.values()
        if not all(tags[i].anchor_only for i in rows)
    ]
    return sorted(
        groups, key=lambda g: (-len(g.members), -g.count, g.canonical.lower())
    )


def mapping(groups: List[Group], excluded: Iterable[str] = ()) -> Dict[str, str]:
    """{original spelling: canonical tag} for every tag in groups.

    Args:
        excluded: Cleaned tag texts to leave as they are (each maps to its own
            most common spelling) rather than merge.
    """
    excluded = set(excluded)
    result = {}
    for group in groups:
        for tag in group.members:
            target = tag.spelling if tag.text in excluded else group.canonical
            for spelling, uses in tag.spellings.items():
                if uses:  # an anchor's own spelling isn't in this run's input
                    result[spelling] = target
    return result


# --- output -------------------------------------------------------------------------


def to_csv(groups: List[Group], mapped: Dict[str, str]) -> str:
    """One row per original spelling: tag, canonical tag, uses, similarity."""
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["tag", "canonical_tag", "uses", "similarity"])
    for group in groups:
        for tag, similarity in zip(group.members, group.similarity):
            for spelling, uses in tag.spellings.items():
                if spelling not in mapped:
                    continue
                writer.writerow([spelling, mapped[spelling], uses, f"{similarity:.3f}"])
    return out.getvalue()


def to_json(
    groups: List[Group], mapped: Dict[str, str], extra: Optional[dict] = None
) -> str:
    """{"mapping": {tag: canonical}, "groups": [{canonical, tags}]} for merged groups.

    Args:
        extra: More top-level fields, such as tag_llm's decisions and flags.
    """
    merged = {}
    for spelling, canonical in mapped.items():
        if spelling != canonical:
            merged.setdefault(canonical, [canonical]).append(spelling)
    return json.dumps(
        {
            "mapping": mapped,
            "groups": [{"canonical": c, "tags": tags} for c, tags in merged.items()],
            **(extra or {}),
        },
        indent=2,
        ensure_ascii=False,
    )


def describe(groups: List[Group]) -> str:
    """Merged groups as indented text, for the command line."""
    merged = [
        g for g in groups if len(g.members) > 1 or len(g.members[0].spellings) > 1
    ]
    lines = [
        f"{len(merged)} group(s) to merge; {len(groups) - len(merged)} tag(s) with no match."
    ]
    for group in merged:
        lines.append(f"\n{group.canonical}  ({group.count} use(s))")
        for tag, similarity in zip(group.members, group.similarity):
            spellings = [s for s in tag.spellings if s != group.canonical]
            if spellings:
                lines.append(f"    {similarity:.2f}  {', '.join(spellings)}")
    return "\n".join(lines)


# --- Zendesk retag scripts ----------------------------------------------------------

# The scripts a person runs against Zendesk; this module only copies them.
# The packaged app bundles them under zendesk_retag/ (see packaging/lasso.spec).
SCRIPTS_DIR = (
    pathlib.Path(getattr(sys, "_MEIPASS", pathlib.Path(__file__).resolve().parent))
    / "zendesk_retag"
)
SCRIPT_FILES = (
    "zd.py",
    "1_find_tickets.py",
    "2_remove_old_tags.py",
    "3_add_new_tags.py",
    "README.txt",
)


def zendesk_tag(tag: str) -> str:
    """tag as Zendesk stores it: lower case, spaces as underscores (a space separates tags)."""
    return "_".join(tag.lower().split())


def retag_changes(mapped: Dict[str, str]) -> Tuple[List[Tuple[str, str]], List[str]]:
    """(old_tag, new_tag) pairs that change a Zendesk tag, and warnings about the rest.

    An old tag with a space can't be a Zendesk tag, so it is left out; a new
    tag with spaces or capitals is converted.
    """
    changes, warnings, seen = [], [], set()
    for old, new in mapped.items():
        target = zendesk_tag(new)
        if target != new and new not in seen:
            seen.add(new)
            warnings.append(f"{new!r} isn't a valid Zendesk tag; using {target!r}.")
        if old == target:
            continue
        if old != zendesk_tag(old):
            warnings.append(
                f"Skipped {old!r}: Zendesk tags have no spaces or capitals, so no ticket has it."
            )
            continue
        changes.append((old, target))
    return changes, warnings


def write_retag_scripts(
    mapped: Dict[str, str], folder: str
) -> Tuple[List[Tuple[str, str]], List[str]]:
    """Writes the Zendesk retag scripts and their tag_changes.csv into folder (created).

    Nothing is sent to Zendesk: a person reviews tag_changes.csv and runs the
    scripts. Returns retag_changes(mapped).

    Raises:
        FileExistsError: if folder already has a tag_changes.csv.
    """
    changes, warnings = retag_changes(mapped)
    out = pathlib.Path(folder)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "tag_changes.csv").exists():
        raise FileExistsError(
            f"{out} already holds retag scripts; choose an empty folder."
        )
    for name in SCRIPT_FILES:
        shutil.copyfile(SCRIPTS_DIR / name, out / name)
    with open(out / "tag_changes.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["old_tag", "new_tag"])
        writer.writerows(changes)
    return changes, warnings


# --- command line -------------------------------------------------------------------


def _threshold(value: str) -> float:
    number = float(value)
    if not 0 < number < 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return number


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tag_normalizer", description=__doc__.split("\n\n")[0]
    )
    parser.add_argument(
        "file", help="Text, CSV or JSON file of tags; - reads standard input"
    )
    parser.add_argument(
        "--backend",
        choices=BACKENDS,
        default=VOYAGE,
        help="voyage (default; sends the tags to Voyage AI), spacy (local, coarser) or llm "
        "(a language model following tag_normalization_reference.md; sends the tags to Cheaper Inference)",
    )
    parser.add_argument(
        "--threshold",
        type=_threshold,
        help=f"Cosine similarity to merge at (default {DEFAULT_THRESHOLDS[VOYAGE]} for voyage, "
        f"{DEFAULT_THRESHOLDS[SPACY]} for spacy)",
    )
    parser.add_argument(
        "--merge-structured",
        action="store_true",
        help="Allow facet__value tags (sentiment__positive) to merge with each other",
    )
    parser.add_argument(
        "--no-candidates",
        action="store_true",
        help="With --backend llm: don't ask Voyage for candidate pairs to send as hints",
    )
    parser.add_argument(
        "--out", help="Write the mapping here: .json for JSON, anything else for CSV"
    )
    parser.add_argument(
        "--scripts",
        metavar="FOLDER",
        help="Write the Zendesk retag scripts and tag_changes.csv into this new folder",
    )
    return parser


def main(argv: List[str]) -> int:
    args = _parser().parse_args(argv)
    if args.backend in (VOYAGE, LLM):
        from core import settings

        settings.load()
    if args.file == "-":
        text = sys.stdin.read()
    else:
        with open(args.file, encoding="utf-8-sig") as f:
            text = f.read()
    tags = collect(parse_tags(text))
    if not tags:
        print("No tags found.", file=sys.stderr)
        return 1
    extra = None
    if args.backend == LLM:
        from tags import tag_llm

        # Before the paid Voyage call, which is wasted if the model isn't set.
        try:
            tag_llm.model_name()
        except tag_llm.TagModelError as error:
            print(error, file=sys.stderr)
            return 1
        candidates = []
        if not args.no_candidates:
            # Zendesk AI tags are never embedded or sent (tag_llm.split_ignored).
            to_embed, _ = tag_llm.split_ignored(tags)
            if len(to_embed) > 1:
                candidates = tag_llm.candidate_pairs(
                    to_embed, embedder(VOYAGE)([tag.text for tag in to_embed])
                )
        result = tag_llm.normalize(tags, candidates=candidates)
        groups, extra = result.groups, result.extra()
    else:
        threshold = args.threshold or DEFAULT_THRESHOLDS[args.backend]
        groups = group_tags(
            tags,
            embedder(args.backend)([tag.text for tag in tags]),
            threshold,
            args.merge_structured,
        )
    mapped = mapping(groups)
    print(describe(groups))
    if args.backend == LLM and (result.flags or result.warnings):
        print(f"\nFor review ({result.model}):\n" + tag_llm.describe_flags(result))
    if args.out:
        output = (
            to_json(groups, mapped, extra)
            if args.out.lower().endswith(".json")
            else to_csv(groups, mapped)
        )
        with open(args.out, "w", encoding="utf-8", newline="") as f:
            f.write(output)
        print(f"\nWrote {len(mapped)} tag(s) to {args.out}")
    if args.scripts:
        changes, warnings = write_retag_scripts(mapped, args.scripts)
        for warning in warnings:
            print(f"Warning: {warning}", file=sys.stderr)
        print(
            f"\nWrote Zendesk retag scripts for {len(changes)} tag change(s) to {args.scripts}; "
            f"see README.txt there. Nothing was sent to Zendesk."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
