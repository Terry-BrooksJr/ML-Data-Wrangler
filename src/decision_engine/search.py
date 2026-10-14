"""Atlas Search (full-text) over the tickets collection.

Needs an Atlas cluster or the mongodb/mongodb-atlas-local Docker image: a
plain mongod has no $search or search indexes.

Also creates the vector index over each ticket's `embedding`, which
src/embeddings.py fills in from Voyage AI (ensure_vector_index). Nothing
queries it yet.

    python -m decision_engine.search ensure [--wait]
    python -m decision_engine.search query "import fails" [--status open]
"""
import argparse
import re
import sys
import time
from typing import List, Optional

import pymongo
from pymongo.collection import Collection
from pymongo.errors import OperationFailure, PyMongoError
from pymongo.operations import SearchIndexModel

TEXT_INDEX = "tickets_text"
TEXT_INDEX_DEFINITION = {
    "mappings": {
        "dynamic": False,
        "fields": {
            "subject": {"type": "string", "analyzer": "lucene.english"},
            "transcript": {
                "type": "document",
                "fields": {"body": {"type": "string", "analyzer": "lucene.english"}},
            },
            # token: exact-match filtering on status, not analysed text.
            "status": {"type": "token"},
        },
    }
}
SEARCH_PATHS = ["subject", "transcript.body"]
EXACT_BOOST = 3
FUZZY_PREFIX_LENGTH = 2
UNSUPPORTED_HINT = (
    "This MongoDB deployment doesn't support Atlas Search. Point MONGO_URI at an Atlas "
    "cluster or the mongodb/mongodb-atlas-local Docker image; a plain mongod can't run $search."
)


# Error codes meaning "this deployment has no Atlas Search": CommandNotFound,
# CommandNotSupported, SearchNotEnabled (atlas-local without mongot) and
# "Unrecognized pipeline stage name" (a plain mongod given $search).
_UNSUPPORTED_CODES = {59, 115, 31082, 40324}
# IndexAlreadyExists: another process created the index after our check.
_INDEX_EXISTS_CODES = {68}
MIN_LIST_SECONDS = 1.0


class SearchUnavailable(RuntimeError):
    """The deployment has no Atlas Search."""


class VectorIndexMismatch(ValueError):
    """The vector index exists with different dimensions than the embeddings."""


VECTOR_INDEX = "tickets_vector"
VECTOR_PATH = "embedding"


def vector_index_definition(dimensions: int) -> dict:
    """Cosine similarity over `embedding`, filterable by status."""
    return {
        "fields": [
            {"type": "vector", "path": VECTOR_PATH, "numDimensions": dimensions, "similarity": "cosine"},
            {"type": "filter", "path": "status"},
        ]
    }


def _is_unsupported(error: OperationFailure) -> bool:
    return error.code in _UNSUPPORTED_CODES or (error.details or {}).get("codeName") == "SearchNotEnabled"


def _search_indexes(tickets: Collection) -> dict:
    """Search indexes by name. Only "no Atlas Search" errors become SearchUnavailable;
    anything else (e.g. missing permissions) is raised as is."""
    try:
        return {index["name"]: index for index in tickets.list_search_indexes()}
    except OperationFailure as e:
        if _is_unsupported(e):
            raise SearchUnavailable(UNSUPPORTED_HINT) from e
        raise


def ensure_search_indexes(tickets: Collection, wait: bool = False, timeout: float = 300.0) -> List[str]:
    """Creates the full-text index if it doesn't exist yet. Safe to call repeatedly.

    An existing index is left as it is, even if its definition differs: change
    it in Atlas, or drop it and call this again. If another process creates
    the index between the check and the create, that counts as success.

    Args:
        wait: Block until the index is queryable (it builds asynchronously).
        timeout: Seconds to wait before raising TimeoutError.

    Returns:
        The names of the indexes created by this call.

    Raises:
        SearchUnavailable: on a deployment without Atlas Search.
        OperationFailure: for other server errors, such as missing permissions.
    """
    created = []
    if TEXT_INDEX not in _search_indexes(tickets):
        try:
            tickets.create_search_index(
                SearchIndexModel(name=TEXT_INDEX, type="search", definition=TEXT_INDEX_DEFINITION)
            )
            created.append(TEXT_INDEX)
        except OperationFailure as e:
            if _is_unsupported(e):
                raise SearchUnavailable(UNSUPPORTED_HINT) from e
            already_exists = e.code in _INDEX_EXISTS_CODES or "already exists" in str(e).lower()
            if not (already_exists and TEXT_INDEX in _search_indexes(tickets)):
                raise
    if wait:
        wait_until_queryable(tickets, TEXT_INDEX, timeout)
    return created


def ensure_vector_index(tickets: Collection, dimensions: int, wait: bool = False, timeout: float = 300.0) -> bool:
    """Creates the vector index if it doesn't exist yet. Safe to call repeatedly.

    Returns:
        True if this call created it.

    Raises:
        VectorIndexMismatch: it exists with a different numDimensions (after
            changing EMBEDDING_DIMENSIONS). Drop it in Atlas and call again.
        SearchUnavailable: on a deployment without Atlas Search.
    """
    existing = _search_indexes(tickets).get(VECTOR_INDEX)
    created = False
    if existing:
        definition = existing.get("latestDefinition") or existing.get("definition") or {}
        current = next((f.get("numDimensions") for f in definition.get("fields", []) if f.get("type") == "vector"), None)
        if current is not None and current != dimensions:
            raise VectorIndexMismatch(
                f"{VECTOR_INDEX} indexes {current}-dimension vectors but EMBEDDING_DIMENSIONS is {dimensions}. "
                f"Drop {VECTOR_INDEX} in Atlas, then embed and create it again."
            )
    else:
        try:
            tickets.create_search_index(
                SearchIndexModel(name=VECTOR_INDEX, type="vectorSearch", definition=vector_index_definition(dimensions))
            )
            created = True
        except OperationFailure as e:
            if _is_unsupported(e):
                raise SearchUnavailable(UNSUPPORTED_HINT) from e
            already_exists = e.code in _INDEX_EXISTS_CODES or "already exists" in str(e).lower()
            if not (already_exists and VECTOR_INDEX in _search_indexes(tickets)):
                raise
    if wait:
        wait_until_queryable(tickets, VECTOR_INDEX, timeout)
    return created


def wait_until_queryable(
    tickets: Collection, name: str = TEXT_INDEX, timeout: float = 300.0, poll_seconds: float = 5.0
) -> None:
    """Polls until the named search index reports queryable.

    Each listing runs under a client-side timeout bounded by the time left,
    so a stalled server can't hold the loop past `timeout`.

    Raises:
        TimeoutError: if it isn't queryable within `timeout` seconds.
    """
    deadline = time.monotonic() + timeout
    while True:
        # Bound each listing by the time left (at least MIN_LIST_SECONDS, so
        # even timeout=0 gets one real check).
        remaining = max(deadline - time.monotonic(), MIN_LIST_SECONDS)
        try:
            with pymongo.timeout(remaining):
                index = _search_indexes(tickets).get(name)
        except PyMongoError as e:
            if getattr(e, "timeout", False):
                raise TimeoutError(f"Search index {name}: listing indexes timed out after {timeout:.0f}s") from e
            raise
        if index and index.get("queryable"):
            return
        if time.monotonic() >= deadline:
            status = index.get("status") if index else "missing"
            raise TimeoutError(f"Search index {name} not queryable after {timeout:.0f}s (status: {status})")
        time.sleep(min(poll_seconds, max(deadline - time.monotonic(), 0)))


def build_search_pipeline(query: str, status: Optional[str] = None, limit: int = 20) -> List[dict]:
    """The aggregation pipeline for a full-text ticket search.

    Exact matches are boosted above fuzzy ones. Fuzzy matching (one edit)
    tolerates typos, but the first FUZZY_PREFIX_LENGTH letters must match:
    without that, short terms drift (e.g. "sso" matched "so"). A status
    filter is an exact match and doesn't affect the score.
    """
    compound = {
        "should": [
            {"text": {"query": query, "path": SEARCH_PATHS, "score": {"boost": {"value": EXACT_BOOST}}}},
            {"text": {"query": query, "path": SEARCH_PATHS,
                      "fuzzy": {"maxEdits": 1, "prefixLength": FUZZY_PREFIX_LENGTH}}},
        ],
        "minimumShouldMatch": 1,
    }
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError(f"limit must be a positive integer, got {limit!r}")
    if status:
        compound["filter"] = [{"equals": {"path": "status", "value": status}}]
    return [
        # $search must be the first stage.
        {"$search": {"index": TEXT_INDEX, "compound": compound, "highlight": {"path": SEARCH_PATHS}}},
        {"$limit": limit},
        {
            "$project": {
                "_id": 0,
                "zendesk_id": 1,
                "subject": 1,
                "status": 1,
                "score": {"$meta": "searchScore"},
                "highlights": {"$meta": "searchHighlights"},
            }
        },
    ]


def search_tickets(tickets: Collection, query: str, status: Optional[str] = None, limit: int = 20) -> List[dict]:
    """Full-text search over ticket subjects and comment bodies, best match first.

    Raises:
        ValueError: for an empty query or a limit below 1.
        SearchUnavailable: on a deployment without Atlas Search.
    """
    if not query or not query.strip():
        raise ValueError("Search query is empty.")
    try:
        return list(tickets.aggregate(build_search_pipeline(query.strip(), status, limit)))
    except OperationFailure as e:
        if _is_unsupported(e):
            raise SearchUnavailable(UNSUPPORTED_HINT) from e
        raise


def highlight_text(result: dict) -> str:
    """The matched passages of one result, with the hits in [brackets]."""
    passages = []
    for highlight in result.get("highlights") or []:
        passages.append(
            "".join(f"[{t['value']}]" if t.get("type") == "hit" else t.get("value", "") for t in highlight.get("texts", []))
        )
    return " … ".join(passages)


# Characters that would act on a terminal rather than print: C0/C1 controls
# (ESC starts ANSI/OSC sequences) and bidi overrides, which can reorder text.
_TERMINAL_UNSAFE = re.compile(r"[\x00-\x1f\x7f-\x9f\u202a-\u202e\u2066-\u2069]")


def terminal_safe(text) -> str:
    """Text safe to print on one terminal line.

    Ticket text is customer-written; newlines and tabs become spaces, and
    other control characters are shown escaped (e.g. "\\x1b") instead of
    being sent to the terminal.
    """
    text = str(text if text is not None else "").replace("\n", " ").replace("\t", " ")
    return _TERMINAL_UNSAFE.sub(lambda m: m.group().encode("unicode_escape").decode("ascii"), text)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m decision_engine.search", description="Atlas full-text ticket search.")
    commands = parser.add_subparsers(dest="command", required=True)
    ensure = commands.add_parser("ensure", help=f"create the {TEXT_INDEX} index if missing")
    ensure.add_argument("--wait", action="store_true", help="wait until the index is queryable")
    ensure.add_argument("--timeout", type=float, default=300.0, help="seconds to wait (default 300)")
    query = commands.add_parser("query", help="search ticket subjects and comments")
    query.add_argument("text")
    query.add_argument("--status", help="only tickets with this exact status")
    query.add_argument("--limit", type=int, default=20)
    return parser


def main(argv: List[str]) -> int:
    from .persistence import get_tickets_collection

    args = _parser().parse_args(argv)  # exits with usage on bad arguments
    if args.command == "ensure":
        created = ensure_search_indexes(get_tickets_collection(), wait=args.wait, timeout=args.timeout)
        print(f"Created: {', '.join(created)}" if created else f"{TEXT_INDEX} already exists.")
        return 0
    try:
        results = search_tickets(get_tickets_collection(), args.text, args.status, args.limit)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    for result in results:
        print(f"{result['score']:.2f}  {terminal_safe(result['zendesk_id'])}  [{terminal_safe(result.get('status'))}]  "
              f"{terminal_safe(result.get('subject'))}")
        if result.get("highlights"):
            print(f"        {terminal_safe(highlight_text(result))[:200]}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except (SearchUnavailable, TimeoutError) as e:
        print(e, file=sys.stderr)
        sys.exit(1)
