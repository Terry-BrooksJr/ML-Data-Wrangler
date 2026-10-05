import json
import os
from functools import lru_cache

import anthropic

from .prompts import SYSTEM_INSTRUCTIONS

CHEAPER_INFERENCE_BASE_URL = "https://api.cheaperinference.com"  # no /v1 — SDK appends /v1/messages

# TODO: confirm the model ID Cheaper Inference expects. The brief specifies
# "claude-opus-4.8"; Anthropic's own ID for that model is "claude-opus-4-8".
MODEL_VERSION = os.environ.get("DECISION_ENGINE_MODEL", "claude-opus-4.8")
SOLVE_CONFIDENCE_FLOOR = 75  # mirrors the prompt's own stated bar — enforced
                             # here as a hard backstop, not just trusted from
                             # the model's self-report


@lru_cache(maxsize=1)
def get_client() -> anthropic.Anthropic:
    """Builds the client on first use so importing this module needs no secrets.

    The SDK retries only 408/409/429/5xx and connection errors (with
    backoff); 400/401/402/413 are raised immediately, as the brief requires.
    """
    return anthropic.Anthropic(
        api_key=os.environ["CHEAPER_INFERENCE_API_KEY"],
        base_url=CHEAPER_INFERENCE_BASE_URL,
    )


def classify_ticket(ticket_payload: str) -> dict:
    response = get_client().messages.create(
        model=MODEL_VERSION,
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
            reason=f"Non-JSON response from model: {raw_text[:500]}"
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

    result["model_version"] = MODEL_VERSION
    return result


def _manual_triage_fallback(reason: str) -> dict:
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
        "model_version": MODEL_VERSION,
    }
