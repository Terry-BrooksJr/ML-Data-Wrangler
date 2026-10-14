"""Atlas Search (full-text) over the tickets collection.

Needs an Atlas cluster or the mongodb/mongodb-atlas-local Docker image: a
plain mongod has no $search or search indexes.

Vector search is deliberately not set up yet. Nothing populates `embedding`,
and the index's numDimensions depends on an embedding model that hasn't been
chosen.

    python -m decision_engine.search ensure [--wait]
    python -m decision_engine.search query "import fails" [--status open]
"""
import sys
import time
from typing import List, Optional

from pymongo.collection import Collection
from pymongo.errors import OperationFailure
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


class SearchUnavailable(RuntimeError):
    """The deployment has no Atlas Search support."""


def _search_indexes(tickets: Collection) -> dict:
    try:
        return {index["name"]: index for index in tickets.list_search_indexes()}
    except OperationFailure as e:
        raise SearchUnavailable(UNSUPPORTED_HINT) from e


def ensure_search_indexes(tickets: Collection, wait: bool = False, timeout: float = 300.0) -> List[str]:
    """Creates the full-text index if it doesn't exist yet. Safe to call repeatedly.

    An existing index is left as it is, even if its definition differs: change
    it in Atlas, or drop it and call this again.

    Args:
        wait: Block until the index is queryable (it builds asynchronously).
        timeout: Seconds to wait before raising TimeoutError.

    Returns:
        The names of the indexes created by this call.

    Raises:
        SearchUnavailable: on a deployment without Atlas Search.
    """
    created = []
    if TEXT_INDEX not in _search_indexes(tickets):
        try:
            tickets.create_search_index(
                SearchIndexModel(name=TEXT_INDEX, type="search", definition=TEXT_INDEX_DEFINITION)
            )
        except OperationFailure as e:
            raise SearchUnavailable(UNSUPPORTED_HINT) from e
        created.append(TEXT_INDEX)
    if wait:
        wait_until_queryable(tickets, TEXT_INDEX, timeout)
    return created


def wait_until_queryable(
    tickets: Collection, name: str = TEXT_INDEX, timeout: float = 300.0, poll_seconds: float = 5.0
) -> None:
    """Polls until the named search index reports queryable.

    Raises:
        TimeoutError: if it isn't queryable within `timeout` seconds.
    """
    deadline = time.monotonic() + timeout
    while True:
        index = _search_indexes(tickets).get(name)
        if index and index.get("queryable"):
            return
        if time.monotonic() >= deadline:
            status = index.get("status") if index else "missing"
            raise TimeoutError(f"Search index {name} not queryable after {timeout:.0f}s (status: {status})")
        time.sleep(poll_seconds)


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
        ValueError: for an empty query.
        SearchUnavailable: on a deployment without Atlas Search.
    """
    if not query or not query.strip():
        raise ValueError("Search query is empty.")
    try:
        return list(tickets.aggregate(build_search_pipeline(query.strip(), status, limit)))
    except OperationFailure as e:
        # 40324: "Unrecognized pipeline stage name", which a plain mongod returns.
        if getattr(e, "code", None) == 40324 or "$search" in str(e):
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


def main(argv: List[str]) -> int:
    from .persistence import get_tickets_collection

    if argv[:1] == ["ensure"]:
        created = ensure_search_indexes(get_tickets_collection(), wait="--wait" in argv)
        print(f"Created: {', '.join(created)}" if created else f"{TEXT_INDEX} already exists.")
        return 0
    if argv[:1] == ["query"] and len(argv) >= 2:
        status = argv[argv.index("--status") + 1] if "--status" in argv else None
        for result in search_tickets(get_tickets_collection(), argv[1], status):
            print(f"{result['score']:.2f}  {result['zendesk_id']}  [{result.get('status')}]  {result.get('subject')}")
            if result.get("highlights"):
                print(f"        {highlight_text(result)[:200]}")
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except SearchUnavailable as e:
        print(e, file=sys.stderr)
        sys.exit(1)
