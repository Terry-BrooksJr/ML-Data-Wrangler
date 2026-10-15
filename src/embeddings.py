"""Voyage AI embeddings for tickets, stored on each ticket in MongoDB.

Each ticket gets `embedding` (a list of floats) and `embedding_meta`
(endpoint, model, dimensions, a hash of the embedded text, and when).

A backfill is two steps: plan_backfill() lists what is stale (shown to the
user for approval), then run_backfill() embeds exactly that list under a
MongoDB lock, so two backfills can't overlap. Each write only lands if the
ticket's text is unchanged since the plan; tickets edited meanwhile are left
for the next run. The Atlas vector index (decision_engine.search) is checked
before any paid call and created only once every approved ticket is done.

Settings (see settings.SETTINGS): EMBEDDING_API_KEY (required),
EMBEDDING_API_URL, EMBEDDING_MODEL, EMBEDDING_DIMENSIONS.

    cd src && python -m embeddings check            # one tiny call, no ticket text
    cd src && python -m embeddings backfill [--yes] # paid; sends ticket text to Voyage

Lives outside decision_engine on purpose: that package must not import an
HTTP client (see test_engine_never_references_zendesk_endpoints).
"""
import argparse
import hashlib
import json
import os
import socket
import sys
import time
import urllib.error
import urllib.request
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np
from loguru import logger
from pymongo.collection import Collection
from pymongo.errors import DuplicateKeyError

from decision_engine.classify import api_url_problem

DEFAULT_URL = "https://api.voyageai.com/v1/embeddings"
DEFAULT_MODEL = "voyage-3.5"
DEFAULT_DIMENSIONS = 1024
# The output_dimension values voyage-3.5 (and -lite, -3-large) accept.
ALLOWED_DIMENSIONS = (256, 512, 1024, 2048)
# Voyage allows 1,000 texts and 320k tokens per request (voyage-3.5). English
# runs ~4 characters per token, but CJK or symbol-heavy text can be 1-2 tokens
# per character, so budget for ~2.5: 120k characters stays under 320k tokens.
# A batch Voyage still rejects for size is split in half and retried.
MAX_BATCH_TEXTS = 128
MAX_BATCH_CHARS = 120_000
# Per ticket; Voyage truncates anything longer than the model's context anyway.
MAX_CHARS = 24_000
MAX_ATTEMPTS = 5
REQUEST_TIMEOUT_SECONDS = 60
# Statuses where the request wasn't processed, so a retry can't bill twice.
RETRYABLE_STATUSES = {429, 502, 503, 504}
# MongoDB lock that keeps two backfills from overlapping; renewed every batch.
LOCK_COLLECTION = "embedding_jobs"
LOCK_ID = "ticket_backfill"
LOCK_TTL = timedelta(minutes=15)


class EmbeddingNotConfigured(RuntimeError):
    """A required embedding setting is missing or invalid."""


class ApiError(RuntimeError):
    """The embedding API refused a request (or redirected it)."""

    def __init__(self, message: str, status: Optional[int] = None, detail: str = ""):
        super().__init__(message)
        self.status = status
        self.detail = detail


class BackfillRunning(RuntimeError):
    """Another backfill holds the lock."""


@dataclass(frozen=True)
class Config:
    url: str
    model: str
    dimensions: int
    api_key: str


def config() -> Config:
    """The embedding settings from the environment, with Voyage defaults.

    Raises:
        EmbeddingNotConfigured: no API key, unsupported dimensions, or an
            insecure endpoint (the key and ticket text are sent there).
    """
    api_key = os.environ.get("EMBEDDING_API_KEY", "").strip()
    if not api_key:
        raise EmbeddingNotConfigured("EMBEDDING_API_KEY is not set. Add your Voyage API key in Settings.")
    url = (os.environ.get("EMBEDDING_API_URL") or DEFAULT_URL).strip()
    problem = api_url_problem(url)
    if problem:
        raise EmbeddingNotConfigured(f"EMBEDDING_API_URL: {problem}")
    raw_dimensions = (os.environ.get("EMBEDDING_DIMENSIONS") or str(DEFAULT_DIMENSIONS)).strip()
    dimensions = parse_dimensions(raw_dimensions)
    if dimensions is None:
        raise EmbeddingNotConfigured(
            f"EMBEDDING_DIMENSIONS must be one of {', '.join(map(str, ALLOWED_DIMENSIONS))}, got {raw_dimensions!r}."
        )
    model = (os.environ.get("EMBEDDING_MODEL") or DEFAULT_MODEL).strip()
    return Config(url=url, model=model, dimensions=dimensions, api_key=api_key)


def parse_dimensions(value: str) -> Optional[int]:
    """value as an allowed dimension count, or None if it isn't one."""
    try:
        dimensions = int(value)
    except ValueError:
        return None
    return dimensions if dimensions in ALLOWED_DIMENSIONS else None


# --- what gets embedded -----------------------------------------------------------


def _public_comments(ticket: dict) -> List[dict]:
    return [c for c in ticket.get("transcript") or [] if not c.get("is_internal_note") and (c.get("body") or "").strip()]


def opening_text(ticket: dict) -> str:
    """Subject plus the requester's first public comment: where the problem is stated.

    Only a comment marked as the requester's counts; if there is none (the
    requester is unknown), the subject alone is used rather than risk
    embedding an agent's reply as the request.
    """
    first = next((c for c in _public_comments(ticket) if c.get("role") in ("requester", "customer")), None)
    parts = [ticket.get("subject") or "", (first or {}).get("body") or ""]
    return "\n\n".join(p.strip() for p in parts if p.strip())[:MAX_CHARS]


def full_text(ticket: dict) -> str:
    """Subject plus every public comment, cut to MAX_CHARS. Internal notes are left out."""
    parts = [ticket.get("subject") or ""] + [c["body"] for c in _public_comments(ticket)]
    return "\n\n".join(p.strip() for p in parts if p.strip())[:MAX_CHARS]


TEXT_VIEWS: Dict[str, Callable[[dict], str]] = {"opening": opening_text, "full": full_text}
# What the backfill stores. Changing it changes every text hash, so the next
# backfill re-embeds everything.
TICKET_TEXT = opening_text


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --- API --------------------------------------------------------------------------


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuses redirects: following one would resend the key and ticket text to
    a URL nobody configured, possibly over plain http."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # urllib then raises HTTPError with the 3xx status


_open = urllib.request.build_opener(_NoRedirect).open


def _not_sent(error: urllib.error.URLError) -> bool:
    """True if the connection failed before the request could reach the server."""
    reason = getattr(error, "reason", None)
    return isinstance(reason, (ConnectionRefusedError, socket.gaierror))


def post_json(url: str, api_key: str, body: dict) -> dict:
    """POSTs JSON. Retries only failures where nothing was processed or billed.

    Retried with backoff: 429 and 502/503/504, and connections refused or
    unresolvable. Not retried: timeouts, resets and other 5xx, since the
    server may already have processed (and billed) the request. Redirects
    are refused.

    Raises:
        ApiError: with the status and the server's message. Messages never
            include the request body (it holds ticket text).
    """
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            with _open(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                return json.load(response)
        except urllib.error.HTTPError as e:
            if 300 <= e.code < 400:
                raise ApiError(f"{url} redirected to {e.headers.get('Location')!r} (HTTP {e.code}); not following, "
                               "since that would resend the API key and ticket text.", e.code) from None
            detail = e.read().decode("utf-8", "replace")[:300]
            if e.code not in RETRYABLE_STATUSES or attempt == MAX_ATTEMPTS:
                raise ApiError(f"{url}: HTTP {e.code}: {detail}", e.code, detail) from None
        except urllib.error.URLError as e:
            if not _not_sent(e) or attempt == MAX_ATTEMPTS:
                raise ApiError(f"{url}: {e.reason}") from None
        except (TimeoutError, ConnectionError) as e:
            raise ApiError(f"{url}: {e} (not retried: the request may already have been processed)") from None
        time.sleep(min(2 ** attempt, 30))
    raise AssertionError("unreachable")


def _batches(texts: List[str]) -> List[Tuple[int, int]]:
    """(start, end) slices of texts under MAX_BATCH_TEXTS and MAX_BATCH_CHARS."""
    batches, start, chars = [], 0, 0
    for i, text in enumerate(texts):
        if i > start and (i - start >= MAX_BATCH_TEXTS or chars + len(text) > MAX_BATCH_CHARS):
            batches.append((start, i))
            start, chars = i, 0
        chars += len(text)
    if start < len(texts):
        batches.append((start, len(texts)))
    return batches


def _too_many_tokens(error: ApiError) -> bool:
    return error.status == 400 and "token" in error.detail.lower()


def _embed_request(texts: List[str], cfg: Config, input_type: str) -> Tuple[List[list], int]:
    """One request; a batch rejected as over the token limit is split in half and retried."""
    try:
        reply = post_json(cfg.url, cfg.api_key, {
            "model": cfg.model,
            "input": texts,
            "input_type": input_type,
            "output_dimension": cfg.dimensions,
            "truncation": True,
        })
    except ApiError as e:
        if not _too_many_tokens(e) or len(texts) == 1:
            raise
        middle = len(texts) // 2
        logger.warning(f"Batch of {len(texts)} was over the token limit; splitting it.")
        first, first_tokens = _embed_request(texts[:middle], cfg, input_type)
        second, second_tokens = _embed_request(texts[middle:], cfg, input_type)
        return first + second, first_tokens + second_tokens
    data = sorted(reply["data"], key=lambda d: d["index"])
    if len(data) != len(texts):
        raise ApiError(f"{cfg.url} returned {len(data)} embeddings for {len(texts)} texts")
    return [d["embedding"] for d in data], (reply.get("usage") or {}).get("total_tokens", 0)


def embed(texts: List[str], cfg: Config, input_type: str = "document") -> Tuple[np.ndarray, int]:
    """Embeds texts with Voyage.

    Args:
        input_type: "document" for text being stored, "query" for a search query.

    Returns:
        (one row per text, tokens billed).
    """
    rows, tokens = [], 0
    for start, end in _batches(texts):
        batch_rows, batch_tokens = _embed_request(texts[start:end], cfg, input_type)
        rows.extend(batch_rows)
        tokens += batch_tokens
    vectors = np.asarray(rows, dtype=np.float32).reshape(len(rows), -1)
    if len(rows) and vectors.shape[1] != cfg.dimensions:
        raise ApiError(f"{cfg.model} returned {vectors.shape[1]}-dimension vectors, expected {cfg.dimensions}")
    return vectors, tokens


# --- tickets ----------------------------------------------------------------------

_META_KEYS = ("endpoint", "model", "dimensions", "text_sha256")


def _meta(cfg: Config, digest: str) -> dict:
    return {"endpoint": cfg.url, "model": cfg.model, "dimensions": cfg.dimensions, "text_sha256": digest}


@dataclass
class StaleTicket:
    doc_id: Any
    zendesk_id: str
    text: str
    digest: str
    # The fields the text came from, as read: a write only lands if they're unchanged.
    source: dict


@dataclass
class Plan:
    """What a backfill will do, computed before asking the user."""
    embed: List[StaleTicket] = field(default_factory=list)
    # Tickets with an embedding whose text is now empty: their vectors are removed.
    clear: List[Any] = field(default_factory=list)


def plan_backfill(tickets: Collection, cfg: Config) -> Plan:
    """Tickets whose embedding is missing or out of date, and ones to clear."""
    plan = Plan()
    for doc in tickets.find({}, {"zendesk_id": 1, "subject": 1, "transcript": 1, "embedding_meta": 1}):
        text = TICKET_TEXT(doc)
        if not text:
            if doc.get("embedding_meta") is not None:
                plan.clear.append(doc["_id"])
            continue
        digest = text_hash(text)
        meta = doc.get("embedding_meta") or {}
        if {k: meta.get(k) for k in _META_KEYS} != _meta(cfg, digest):
            plan.embed.append(StaleTicket(doc["_id"], doc.get("zendesk_id"), text, digest,
                                          {"subject": doc.get("subject"), "transcript": doc.get("transcript")}))
    return plan


@contextmanager
def backfill_lock(tickets: Collection) -> Iterator[Callable[[], None]]:
    """Holds the backfill lock; yields a function that renews it.

    The lock expires after LOCK_TTL without renewal, so one left by a
    crashed process doesn't block forever.

    Raises:
        BackfillRunning: another backfill holds it.
    """
    locks = tickets.database[LOCK_COLLECTION]
    holder = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"

    def take() -> None:
        now = datetime.now(timezone.utc)
        try:
            locks.find_one_and_update(
                {"_id": LOCK_ID, "$or": [{"holder": holder}, {"expires_at": {"$lt": now}}]},
                {"$set": {"holder": holder, "expires_at": now + LOCK_TTL}},
                upsert=True,
            )
        except DuplicateKeyError:
            current = locks.find_one({"_id": LOCK_ID}) or {}
            raise BackfillRunning(
                f"Another embedding backfill is running ({current.get('holder')}, lock until "
                f"{current.get('expires_at')}). Wait for it to finish."
            ) from None

    take()
    try:
        yield take
    finally:
        locks.delete_one({"_id": LOCK_ID, "holder": holder})


def run_backfill(
    tickets: Collection,
    cfg: Config,
    plan: Plan,
    should_stop: Optional[Callable[[], bool]] = None,
    embed_fn: Optional[Callable[[List[str], Config], Tuple[np.ndarray, int]]] = None,
) -> dict:
    """Embeds the tickets in an approved plan and clears its empty ones.

    Only the plan's tickets are sent, never ones added since. Works in
    batches, each written before the next is sent, so a stop or an error
    keeps what was done. A write is skipped if the ticket's subject or
    transcript changed since the plan (counted as `changed`; the next
    backfill picks it up).

    Returns:
        {"embedded", "changed", "cleared", "tokens", "cancelled", "complete"}:
        complete is True only if every planned ticket was handled.

    Raises:
        BackfillRunning: another backfill is running.
    """
    embed_fn = embed_fn or embed
    summary = {"embedded": 0, "changed": 0, "cleared": 0, "tokens": 0, "cancelled": 0, "complete": False}
    with backfill_lock(tickets) as renew:
        for doc_id in plan.clear:
            result = tickets.update_one({"_id": doc_id}, {"$unset": {"embedding": "", "embedding_meta": ""}})
            summary["cleared"] += result.modified_count
        texts = [item.text for item in plan.embed]
        for start, end in _batches(texts):
            if should_stop and should_stop():
                summary["cancelled"] = len(texts) - start
                logger.warning(f"Embedding cancelled; {summary['cancelled']} ticket(s) not started.")
                return summary
            renew()
            vectors, tokens = embed_fn(texts[start:end], cfg)
            summary["tokens"] += tokens
            now = datetime.now(timezone.utc)
            for item, vector in zip(plan.embed[start:end], vectors):
                result = tickets.update_one(
                    {"_id": item.doc_id, **item.source},
                    {"$set": {"embedding": vector.tolist(), "embedding_meta": {**_meta(cfg, item.digest), "at": now}}},
                )
                summary["embedded" if result.matched_count else "changed"] += 1
            logger.info(f"Embedded {summary['embedded']}/{len(texts)} ticket(s).")
    summary["complete"] = True
    return summary


def format_summary(summary: dict) -> str:
    lines = [f"Embedded: {summary['embedded']}", f"Tokens billed: {summary['tokens']}"]
    if summary.get("changed"):
        lines.append(f"Edited while embedding (left for the next run): {summary['changed']}")
    if summary.get("cleared"):
        lines.append(f"Embeddings removed (ticket has no text now): {summary['cleared']}")
    if summary.get("cancelled"):
        lines.append(f"Not started (cancelled): {summary['cancelled']}")
    if summary.get("index"):
        lines.append(f"Vector index: {summary['index']}")
    return "\n".join(lines)


def index_status(tickets: Collection, cfg: Config) -> str:
    """Checks the vector index before any paid call.

    Returns:
        "ok" (matches), "missing" (will be created) or "unavailable" (no
        Atlas Search: embeddings are stored, no index).

    Raises:
        search.VectorIndexMismatch: it exists but differs from what these
            settings need; embedding now would store unusable vectors.
    """
    from decision_engine import search

    try:
        return search.check_vector_index(tickets, cfg.dimensions)
    except search.SearchUnavailable:
        return "unavailable"


def finish_index(tickets: Collection, cfg: Config, summary: dict, status: str) -> None:
    """Creates the vector index, but only after a complete backfill."""
    from decision_engine import search

    if status == "unavailable":
        summary["index"] = "skipped: this MongoDB deployment has no Atlas Search"
    elif not summary.get("complete") or summary.get("changed"):
        summary["index"] = "not created yet: some tickets still need embedding; run again"
    else:
        created = search.ensure_vector_index(tickets, cfg.dimensions)
        summary["index"] = f"{search.VECTOR_INDEX} {'created' if created else 'already exists'}"


def describe_plan(plan: Plan, cfg: Config) -> str:
    message = (f"{len(plan.embed)} ticket(s) need embeddings. Each one's subject and first requester comment "
               f"will be sent to {cfg.url} ({cfg.model}, {cfg.dimensions} dimensions); these are paid calls.")
    if plan.clear:
        message += f" {len(plan.clear)} ticket(s) with no text left will have their embedding removed."
    return message


# --- CLI --------------------------------------------------------------------------


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m embeddings", description="Voyage AI ticket embeddings.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="embed one short test string (no ticket text)")
    backfill = commands.add_parser("backfill", help="embed tickets that are new or changed, then ensure the vector index")
    backfill.add_argument("--yes", action="store_true", help="don't ask before sending ticket text to Voyage")
    args = parser.parse_args(argv)

    import settings

    settings.load()
    try:
        cfg = config()
    except EmbeddingNotConfigured as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if args.command == "check":
        vectors, tokens = embed(["connection check"], cfg, input_type="query")
        print(f"OK: {cfg.model} at {cfg.url} returned a {vectors.shape[1]}-dimension vector ({tokens} tokens).")
        return 0

    from decision_engine import search
    from decision_engine.persistence import get_tickets_collection

    tickets = get_tickets_collection()
    try:
        status = index_status(tickets, cfg)
    except search.VectorIndexMismatch as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    plan = plan_backfill(tickets, cfg)
    if plan.embed:
        print(describe_plan(plan, cfg))
        if not args.yes:
            if not sys.stdin.isatty():
                print("Not a terminal: pass --yes to confirm.", file=sys.stderr)
                return 1
            if input("Continue? [y/N] ").strip().lower() not in ("y", "yes"):
                return 1
    try:
        summary = run_backfill(tickets, cfg, plan)
    except BackfillRunning as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    finish_index(tickets, cfg, summary, status)
    print(format_summary(summary) if plan.embed or plan.clear else
          "Every ticket's embedding is up to date.\n" + format_summary(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
