"""Voyage AI embeddings for tickets, stored on each ticket in MongoDB.

Each ticket gets `embedding` (a list of floats) and `embedding_meta` (model,
dimensions, a hash of the embedded text, and when). embed_tickets() only
embeds tickets whose text, model or dimensions changed since their last
embedding, so it is safe to re-run. The Atlas vector index over `embedding`
is decision_engine.search.ensure_vector_index().

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
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger
from pymongo.collection import Collection

from decision_engine.classify import api_url_problem

DEFAULT_URL = "https://api.voyageai.com/v1/embeddings"
DEFAULT_MODEL = "voyage-3.5"
DEFAULT_DIMENSIONS = 1024
# The output_dimension values voyage-3.5 (and -lite, -3-large) accept.
ALLOWED_DIMENSIONS = (256, 512, 1024, 2048)
# Voyage allows 1,000 texts and 320k tokens per request (voyage-3.5); stay
# well under both. ~4 characters per token.
MAX_BATCH_TEXTS = 128
MAX_BATCH_CHARS = 400_000
# ~6k tokens per ticket; Voyage would truncate longer text anyway.
MAX_CHARS = 24_000
MAX_ATTEMPTS = 5
REQUEST_TIMEOUT_SECONDS = 60


class EmbeddingNotConfigured(RuntimeError):
    """A required embedding setting is missing or invalid."""


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
    """Subject plus the requester's first public comment: where the problem is stated."""
    comments = _public_comments(ticket)
    first = next((c for c in comments if c.get("role") in ("requester", "customer")), comments[0] if comments else None)
    parts = [ticket.get("subject") or "", (first or {}).get("body") or ""]
    return "\n\n".join(p.strip() for p in parts if p.strip())[:MAX_CHARS]


def full_text(ticket: dict) -> str:
    """Subject plus every public comment, cut to MAX_CHARS. Internal notes are left out."""
    parts = [ticket.get("subject") or ""] + [c["body"] for c in _public_comments(ticket)]
    return "\n\n".join(p.strip() for p in parts if p.strip())[:MAX_CHARS]


TEXT_VIEWS: Dict[str, Callable[[dict], str]] = {"opening": opening_text, "full": full_text}
# What embed_tickets stores. Changing it changes every text hash, so the next
# backfill re-embeds everything.
TICKET_TEXT = opening_text


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --- API --------------------------------------------------------------------------


def post_json(url: str, api_key: str, body: dict) -> dict:
    """POSTs JSON, retrying 429/5xx and connection errors with backoff."""
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                return json.load(response)
        except urllib.error.HTTPError as e:
            # Read the body for the message, but never echo the request (it holds ticket text).
            detail = e.read().decode("utf-8", "replace")[:300]
            if (e.code != 429 and e.code < 500) or attempt == MAX_ATTEMPTS:
                raise RuntimeError(f"{url}: HTTP {e.code}: {detail}") from None
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == MAX_ATTEMPTS:
                raise RuntimeError(f"{url}: {e}") from None
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


def embed(texts: List[str], cfg: Config, input_type: str = "document") -> Tuple[np.ndarray, int]:
    """Embeds texts with Voyage.

    Args:
        input_type: "document" for text being stored, "query" for a search query.

    Returns:
        (one row per text, tokens billed).
    """
    rows, tokens = [], 0
    for start, end in _batches(texts):
        reply = post_json(cfg.url, cfg.api_key, {
            "model": cfg.model,
            "input": texts[start:end],
            "input_type": input_type,
            "output_dimension": cfg.dimensions,
            "truncation": True,
        })
        data = sorted(reply["data"], key=lambda d: d["index"])
        if len(data) != end - start:
            raise RuntimeError(f"{cfg.url} returned {len(data)} embeddings for {end - start} texts")
        rows.extend(d["embedding"] for d in data)
        tokens += (reply.get("usage") or {}).get("total_tokens", 0)
    vectors = np.asarray(rows, dtype=np.float32).reshape(len(rows), -1)
    if len(rows) and vectors.shape[1] != cfg.dimensions:
        raise RuntimeError(f"{cfg.model} returned {vectors.shape[1]}-dimension vectors, expected {cfg.dimensions}")
    return vectors, tokens


# --- tickets ----------------------------------------------------------------------


def _meta(cfg: Config, digest: str) -> dict:
    return {"model": cfg.model, "dimensions": cfg.dimensions, "text_sha256": digest}


def stale_tickets(tickets: Collection, cfg: Config) -> List[Tuple[str, str, str]]:
    """Tickets with text whose embedding is missing or out of date.

    Returns:
        (zendesk_id, text, text hash) for each, in collection order.
    """
    stale = []
    projection = {"zendesk_id": 1, "subject": 1, "transcript": 1, "embedding_meta": 1}
    for doc in tickets.find({}, projection):
        text = TICKET_TEXT(doc)
        if not text:
            continue
        digest = text_hash(text)
        meta = doc.get("embedding_meta") or {}
        if {k: meta.get(k) for k in ("model", "dimensions", "text_sha256")} != _meta(cfg, digest):
            stale.append((doc["zendesk_id"], text, digest))
    return stale


def embed_tickets(
    tickets: Collection,
    cfg: Optional[Config] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    embed_fn: Optional[Callable[[List[str], Config], Tuple[np.ndarray, int]]] = None,
) -> dict:
    """Embeds every stale ticket and stores the result on it.

    Works in batches; each batch is written before the next is sent, so a
    stop or an error keeps what was already done. A ticket edited while its
    batch is in flight keeps the old text's hash and is picked up next run.

    Returns:
        {"embedded": int, "tokens": int, "cancelled": int (not started)}.
    """
    cfg = cfg or config()
    embed_fn = embed_fn or embed
    stale = stale_tickets(tickets, cfg)
    summary = {"embedded": 0, "tokens": 0, "cancelled": 0}
    texts = [text for _, text, _ in stale]
    for start, end in _batches(texts):
        if should_stop and should_stop():
            summary["cancelled"] = len(stale) - start
            logger.warning(f"Embedding cancelled; {summary['cancelled']} ticket(s) not started.")
            break
        vectors, tokens = embed_fn(texts[start:end], cfg)
        now = datetime.now(timezone.utc)
        for (zendesk_id, _, digest), vector in zip(stale[start:end], vectors):
            tickets.update_one(
                {"zendesk_id": zendesk_id},
                {"$set": {"embedding": vector.tolist(), "embedding_meta": {**_meta(cfg, digest), "at": now}}},
            )
        summary["embedded"] += end - start
        summary["tokens"] += tokens
        logger.info(f"Embedded {summary['embedded']}/{len(stale)} ticket(s).")
    return summary


def format_summary(summary: dict) -> str:
    lines = [f"Embedded: {summary['embedded']}", f"Tokens billed: {summary['tokens']}"]
    if summary.get("cancelled"):
        lines.append(f"Not started (cancelled): {summary['cancelled']}")
    if summary.get("index"):
        lines.append(f"Vector index: {summary['index']}")
    return "\n".join(lines)


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
    count = len(stale_tickets(tickets, cfg))
    if count == 0:
        print("Every ticket's embedding is up to date.")
    else:
        print(f"{count} ticket(s) need embeddings. Their subject and first requester comment will be sent to "
              f"{cfg.url} ({cfg.model}, {cfg.dimensions} dimensions); paid calls.")
        if not args.yes:
            if not sys.stdin.isatty():
                print("Not a terminal: pass --yes to confirm.", file=sys.stderr)
                return 1
            if input("Continue? [y/N] ").strip().lower() not in ("y", "yes"):
                return 1
        print(format_summary(embed_tickets(tickets, cfg)))
    try:
        created = search.ensure_vector_index(tickets, cfg.dimensions)
        print(f"Vector index {search.VECTOR_INDEX}: {'created' if created else 'already exists'}.")
    except (search.SearchUnavailable, search.VectorIndexMismatch) as e:
        print(f"Vector index not created: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
