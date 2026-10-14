import json
import os
from functools import lru_cache
from typing import Optional

import anthropic
from loguru import logger

from .prompts import SYSTEM_INSTRUCTIONS

CHEAPER_INFERENCE_BASE_URL = "https://api.cheaperinference.com"  # no /v1 — SDK appends /v1/messages

# TODO: confirm the model ID Cheaper Inference expects. The brief specifies
# "claude-opus-4.8"; Anthropic's own ID for that model is "claude-opus-4-8".
MODEL_VERSION = os.environ.get("DECISION_ENGINE_MODEL", "claude-opus-4.8")
SOLVE_CONFIDENCE_FLOOR = 75  # mirrors the prompt's own stated bar — enforced
                             # here as a hard backstop, not just trusted from
                             # the model's self-report
CONFIRMATION_CONFIDENCE_FLOOR = SOLVE_CONFIDENCE_FLOOR  # below this, re-check
                                                        # with the confirmation model


def primary_model() -> str:
    return os.environ.get("CHEAPER_INFERENCE_PRIMARY_MODEL_NAME") or MODEL_VERSION


# Secondary and confirmation have no default: when unset, that step is
# skipped rather than re-running the ticket on a model that already saw it.
def secondary_model() -> Optional[str]:
    """Used only when the primary call raises an API error."""
    return os.environ.get("CHEAPER_INFERENCE_SECONDARY_MODEL_NAME") or None


def confirmation_model() -> Optional[str]:
    """Re-classifies manual-triage and low-confidence results."""
    return os.environ.get("CHEAPER_INFERENCE_CONFIRMATION_MODEL_NAME") or None


@lru_cache(maxsize=1)
def get_client() -> anthropic.Anthropic:
    """Builds the client on first use so importing this module needs no secrets.

    The SDK retries only 408/409/429/5xx and connection errors (with
    backoff); 400/401/402/413 are raised immediately, as the brief requires.
    """
    return anthropic.Anthropic(
        api_key=os.environ["CHEAPER_INFERENCE_API_KEY"],
        base_url=os.environ.get("CHEAPER_INFERENCE_API_URL") or CHEAPER_INFERENCE_BASE_URL,
    )


def classify_ticket(ticket_payload: str) -> dict:
    """Classifies one ticket, escalating through the configured models.

    The primary model classifies first; if that call raises an API error the
    secondary model is tried instead. A manual_triage result, or one below
    CONFIRMATION_CONFIDENCE_FLOOR, is then re-classified by the confirmation
    model, whose answer is returned. The answer it replaced is kept under
    "confirmation_of" for audit. If the confirmation call itself fails, the
    first answer is returned unchanged.

    A ticket is never sent twice to the same model: an unset secondary or
    confirmation model, or one naming the model that already answered, skips
    that step (the primary error is re-raised, or the first answer is kept).
    """
    primary = primary_model()
    try:
        result = _classify_with(primary, ticket_payload)
    except anthropic.APIError as e:
        secondary = secondary_model()
        if secondary in (None, primary):
            logger.error(f"Primary model failed ({e}) and no distinct secondary model is set.")
            raise
        logger.warning(f"Primary model failed ({e}); retrying on secondary model.")
        result = _classify_with(secondary, ticket_payload)

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
        confirmed = _classify_with(confirmation, ticket_payload)
    except anthropic.APIError as e:
        logger.warning(f"Confirmation model failed ({e}); keeping the first result.")
        return result
    confirmed["confirmation_of"] = {
        key: result.get(key) for key in ("model_version", "classification", "confidence", "reasoning")
    }
    return confirmed


def needs_confirmation(result: dict) -> bool:
    return (
        result.get("classification") == "manual_triage"
        or result.get("confidence", 0) < CONFIRMATION_CONFIDENCE_FLOOR
    )


def _classify_with(model: str, ticket_payload: str) -> dict:
    response = get_client().messages.create(
        model=model,
        max_tokens=1024,
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

    try:
        result = json.loads(raw_text)
    except json.JSONDecodeError:
        return _manual_triage_fallback(
            reason=f"Non-JSON response from model: {raw_text[:500]}",
            model=model,
        )

    # Hard backstop: don't trust a self-reported Solve under the stated bar.
    if result.get("classification") == "flag_solve" and result.get("confidence", 0) < SOLVE_CONFIDENCE_FLOOR:
        result["classification"] = "manual_triage"
        result["tag"] = "manual_triage"
        result["reasoning"] = (
            f"Downgraded from flag_solve: confidence {result.get('confidence')} "
            f"is below the {SOLVE_CONFIDENCE_FLOOR} floor. Original reasoning: "
            + result.get("reasoning", "")
        )

    result["model_version"] = model
    return result


def _manual_triage_fallback(reason: str, model: str) -> dict:
    return {
        "ticket_id": None,
        "classification": "manual_triage",
        "confidence": 0,
        "signals_matched": [],
        "reasoning": reason,
        "tag": "manual_triage",
        "internal_note": "Auto-routed: model did not return valid JSON.",
        "draft_message": None,
        "leadership_recipient": None,
        "model_version": model,
    }
