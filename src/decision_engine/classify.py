"""Zendesk ticket triage decision engine (Module 2).

Reads mapped ticket documents from Mongo, asks an LLM (via Cheaper Inference)
for one of five triage classifications, and appends the result to
tickets.agent_reviews. Never writes to Zendesk; write-back is Module 3.
"""

import ipaddress
import json
import os
import time
from functools import lru_cache
from typing import Optional
from urllib.parse import urlsplit

import anthropic
from loguru import logger

from .guards import apply_guards, InvalidReply, SOLVE_CONFIDENCE_FLOOR, validate_reply
from .prompts import PROMPT_VERSION, SYSTEM_INSTRUCTIONS

CHEAPER_INFERENCE_BASE_URL = (
    "https://api.cheaperinference.com"  # no /v1 — SDK appends /v1/messages
)

# TODO(#15): confirm the model ID Cheaper Inference expects. The brief specifies
# "claude-opus-4.8"; Anthropic's own ID for that model is "claude-opus-4-8".
DEFAULT_MODEL = "claude-opus-4.8"
# Reasoning, an internal note and a drafted message must all fit; a reply cut
# off at this limit is an engine error, not an abstention.
MAX_TOKENS = 4096
REQUEST_TIMEOUT_SECONDS = 120.0
MAX_RETRIES = 3  # the SDK retries only 408/409/429/5xx and connection errors
CONFIRMATION_CONFIDENCE_FLOOR = SOLVE_CONFIDENCE_FLOOR  # below this, re-check
# with the confirmation model


class EngineError(Exception):
    """The model answered, but not with a usable classification.

    Raised for a reply cut off at max_tokens, one with no JSON object, or JSON
    that fails validation. Unlike a manual_triage answer (a deliberate
    abstention), this is a failure: the ticket should stay in the queue.
    """

    def __init__(self, message: str, model: str, audit: Optional[dict] = None):
        super().__init__(message)
        self.model = model
        self.audit = audit or {}


def primary_model() -> str:
    return (
        os.environ.get("CHEAPER_INFERENCE_PRIMARY_MODEL_NAME")
        or os.environ.get("DECISION_ENGINE_MODEL")
        or DEFAULT_MODEL
    )


# Secondary and confirmation have no default: when unset, that step is
# skipped rather than re-running the ticket on a model that already saw it.
def secondary_model() -> Optional[str]:
    """Used only when the primary call fails (API error or unusable reply)."""
    return os.environ.get("CHEAPER_INFERENCE_SECONDARY_MODEL_NAME") or None


def confirmation_model() -> Optional[str]:
    """Re-classifies manual-triage and low-confidence results."""
    return os.environ.get("CHEAPER_INFERENCE_CONFIRMATION_MODEL_NAME") or None


class InsecureApiUrl(ValueError):
    """CHEAPER_INFERENCE_API_URL would send the API key and tickets unencrypted."""


def is_loopback_url(url: str) -> bool:
    """True if url's host is this machine (localhost, 127.0.0.0/8 or ::1)."""
    host = urlsplit(url).hostname or ""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def api_url_problem(url: str) -> Optional[str]:
    """Why url can't be used as the API URL, or None if it can.

    The API key and ticket text go to this URL, so it must be https://.
    Plain http:// is allowed only to this machine (a local proxy, say);
    check is_loopback_url() to warn about that case.
    """
    parts = urlsplit(url)
    if parts.scheme == "https" and parts.hostname:
        return None
    if parts.scheme == "http" and is_loopback_url(url):
        return None
    if parts.scheme == "http":
        return f"{url} is not encrypted: the API key and ticket text would be readable on the network. Use https://."
    return f"{url} is not an https:// URL."


def api_base_url() -> str:
    """CHEAPER_INFERENCE_API_URL without a trailing /v1, which the SDK adds.

    A base URL ending in /v1 would send requests to /v1/v1/messages, which
    Cheaper Inference answers with a 404 HTML page.

    Raises:
        InsecureApiUrl: The URL isn't https:// and isn't on this machine.
    """
    url = (
        (os.environ.get("CHEAPER_INFERENCE_API_URL") or CHEAPER_INFERENCE_BASE_URL)
        .strip()
        .rstrip("/")
    )
    problem = api_url_problem(url)
    if problem:
        raise InsecureApiUrl(f"CHEAPER_INFERENCE_API_URL: {problem}")
    if urlsplit(url).scheme == "http":
        logger.warning(
            f"CHEAPER_INFERENCE_API_URL {url} is plain http; allowed only because it is on this machine."
        )
    return url[: -len("/v1")] if url.endswith("/v1") else url


@lru_cache(maxsize=1)
def get_client() -> anthropic.Anthropic:
    """Builds the client on first use so importing this module needs no secrets.

    The SDK retries only 408/409/429/5xx and connection errors (with
    backoff); 400/401/402/413 are raised immediately: retrying them can't help.
    """
    return anthropic.Anthropic(
        api_key=os.environ["CHEAPER_INFERENCE_API_KEY"],
        base_url=api_base_url(),
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=MAX_RETRIES,
    )


def classify_ticket(
    ticket_payload: str, reengage_count: int = 0, assignee: Optional[str] = None
) -> dict:
    """Classifies one ticket, escalating through the configured models.

    The primary model classifies first; if that call fails (an API error, or
    an EngineError for an unusable reply) the secondary model is tried
    instead. A manual_triage result, or one below
    CONFIRMATION_CONFIDENCE_FLOOR, is then re-classified by the confirmation
    model, whose answer is returned. The answer it replaced is kept under
    "confirmation_of" for audit. If the confirmation call itself fails, the
    first answer is returned unchanged.

    A ticket is never sent twice to the same model: an unset secondary or
    confirmation model, or one naming the model that already answered, skips
    that step (the primary error is re-raised, or the first answer is kept).

    Args:
        ticket_payload: The output of build_ticket_payload.
        reengage_count: Re-engage attempts the ticket already has, for the
            re-engage limit guard.
        assignee: The ticket's assignee (Zendesk user ID), which decides a
            leadership escalation's recipients (guards.apply_guards).

    Raises:
        anthropic.APIError or EngineError: when no model gave a usable answer.
    """
    primary = primary_model()
    try:
        result = _classify_with(primary, ticket_payload, reengage_count, assignee)
    except (anthropic.APIError, EngineError) as e:
        secondary = secondary_model()
        if secondary in (None, primary):
            logger.error(
                f"Primary model failed ({e}) and no distinct secondary model is set."
            )
            raise
        logger.warning(f"Primary model failed ({e}); retrying on secondary model.")
        result = _classify_with(secondary, ticket_payload, reengage_count, assignee)

    if not needs_confirmation(result):
        return result
    confirmation = confirmation_model()
    if confirmation in (None, result["model_version"]):
        logger.warning(
            f"Result needs confirmation ({result.get('classification')}, "
            f"confidence {result.get('confidence')}) but no distinct confirmation model is set; keeping it."
        )
        return result
    try:
        confirmed = _classify_with(
            confirmation, ticket_payload, reengage_count, assignee
        )
    except (anthropic.APIError, EngineError) as e:
        logger.warning(f"Confirmation model failed ({e}); keeping the first result.")
        return result
    confirmed["confirmation_of"] = {
        key: result.get(key)
        for key in (
            "model_version",
            "classification",
            "confidence",
            "reasoning",
            "audit",
        )
    }
    return confirmed


def needs_confirmation(result: dict) -> bool:
    return (
        result.get("classification") == "manual_triage"
        or result.get("confidence", 0) < CONFIRMATION_CONFIDENCE_FLOOR
    )


def extract_json(text: str):
    """Parses the first JSON object in the reply.

    Tolerates markdown code fences and prose around the object, which
    json.loads alone rejects.

    Raises:
        ValueError: if the text contains no JSON object.
    """
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            value, _ = decoder.raw_decode(text, start)
            return value
        except json.JSONDecodeError:
            start = text.find("{", start + 1)
    raise ValueError("no JSON object found")


def _usage(response) -> dict:
    usage = getattr(response, "usage", None)
    return {
        key: getattr(usage, key, None) or 0
        for key in (
            "input_tokens",
            "output_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
        )
    }


def _classify_with(
    model: str,
    ticket_payload: str,
    reengage_count: int = 0,
    assignee: Optional[str] = None,
) -> dict:
    started = time.monotonic()
    response = get_client().messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        system=[
            {
                "type": "text",
                "text": SYSTEM_INSTRUCTIONS,
                "cache_control": {"type": "ephemeral"},  # static across every
                # ticket — cache it
            }
        ],
        messages=[{"role": "user", "content": ticket_payload}],
    )
    # Take the first text block rather than content[0], which may be a
    # thinking block on models that think by default.
    raw_text = next((b.text for b in response.content if b.type == "text"), "")
    stop_reason = getattr(response, "stop_reason", None)
    audit = {
        "usage": _usage(response),
        "latency_ms": int((time.monotonic() - started) * 1000),
        "request_id": getattr(response, "_request_id", None),
        "prompt_version": PROMPT_VERSION,
        "stop_reason": stop_reason,
        "raw_response": raw_text,
    }

    if stop_reason == "max_tokens":
        logger.error(f"{model}: reply cut off at max_tokens ({MAX_TOKENS}).")
        raise EngineError(f"Reply cut off at max_tokens ({MAX_TOKENS})", model, audit)
    try:
        result = validate_reply(extract_json(raw_text))
    except ValueError as e:  # InvalidReply is a ValueError too
        kind = "Invalid reply" if isinstance(e, InvalidReply) else "Non-JSON response"
        logger.error(f"{model}: {kind.lower()}: {e}")
        # The reply itself stays out of the message: the message is logged and
        # stored as the ticket's error, and a reply can quote ticket text. It is
        # kept in full in audit["raw_response"] (stored with last_engine_error).
        raise EngineError(
            f"{kind} from model ({len(raw_text)}-character reply): {e}", model, audit
        ) from e

    apply_guards(result, reengage_count, assignee)
    result["model_version"] = model
    result["audit"] = audit
    return result
