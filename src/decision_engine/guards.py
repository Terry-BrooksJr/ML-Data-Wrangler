"""Checks on the model's reply, enforced in code rather than trusted to the prompt.

validate_reply rejects a reply the engine can't use (missing keys, wrong
types, unknown classification); the caller treats that as an engine error.
apply_guards then overrides a valid reply that breaks a hard rule, recording
each override in `reasoning` and `guard_overrides`.
"""
from typing import List

CLASSIFICATIONS = (
    "escalate_leadership",
    "escalation_internal",
    "flag_reengage",
    "flag_solve",
    "manual_triage",
)
SOLVE_CONFIDENCE_FLOOR = 75  # mirrors the prompt's own stated bar — enforced
                             # here as a hard backstop, not just trusted from
                             # the model's self-report
REENGAGE_LIMIT = 2  # the prompt's ceiling: at most two re-engage attempts

_OPTIONAL_TEXT = ("internal_note", "draft_message", "leadership_recipient", "tag", "ticket_id")


class InvalidReply(ValueError):
    """The model's JSON parsed but doesn't match the output format."""


def describe_value(value) -> str:
    """A value's type and size, never its text.

    Validation messages are logged and stored with the ticket's error; a
    model can put anything in a field (including text quoted from the
    ticket), so values are described rather than shown. That includes
    numbers: a model can echo a date of birth or an account number as one.
    The full reply is kept in the error's audit (`raw_response`) for
    debugging.
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "a boolean"
    if isinstance(value, (int, float)):
        return "a number"
    if isinstance(value, str):
        return f"a {len(value)}-character string"
    if isinstance(value, list):
        return f"a list of {len(value)} item(s)"
    if isinstance(value, dict):
        return "an object"
    return f"a {type(value).__name__}"


def validate_reply(reply) -> dict:
    """Checks the reply's shape and returns a normalised copy.

    Raises:
        InvalidReply: listing every problem found.
    """
    if not isinstance(reply, dict):
        raise InvalidReply(f"expected a JSON object, got {type(reply).__name__}")

    problems: List[str] = []
    if reply.get("classification") not in CLASSIFICATIONS:
        problems.append(f"classification ({describe_value(reply.get('classification'))}) is not one of "
                        f"{', '.join(CLASSIFICATIONS)}")

    confidence = reply.get("confidence")
    # bool is an int subclass; a string "80" is rejected rather than guessed at.
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        problems.append(f"confidence must be a number, got {describe_value(confidence)}")
    elif not 0 <= confidence <= 100:
        problems.append("confidence is outside 0-100")

    if not isinstance(reply.get("reasoning"), str):
        problems.append("reasoning must be a string")

    signals = reply.get("signals_matched", [])
    if not isinstance(signals, list) or not all(isinstance(s, str) for s in signals):
        problems.append("signals_matched must be a list of strings")

    for key in _OPTIONAL_TEXT:
        if reply.get(key) is not None and not isinstance(reply[key], str):
            problems.append(f"{key} must be a string or null")

    if problems:
        raise InvalidReply("; ".join(problems))

    result = dict(reply)
    result["confidence"] = int(round(confidence))
    result["signals_matched"] = signals
    for key in _OPTIONAL_TEXT:
        result.setdefault(key, None)
    return result


def _override(result: dict, message: str) -> None:
    result.setdefault("guard_overrides", []).append(message)
    result["reasoning"] = f"{message} Original reasoning: {result.get('reasoning', '')}"


def apply_guards(result: dict, reengage_count: int = 0) -> dict:
    """Overrides a validated reply that breaks a hard rule.

    Args:
        result: A reply that passed validate_reply. Modified in place.
        reengage_count: How many re-engage attempts the ticket already has.

    Returns:
        The same dict.
    """
    model_classification = result["classification"]
    model_tag = result.get("tag")
    if result["classification"] == "flag_reengage" and reengage_count >= REENGAGE_LIMIT:
        drafted = result.get("draft_message")
        result["classification"] = "escalation_internal"
        result["draft_message"] = None
        result["internal_note"] = (
            f"Re-engage limit reached ({reengage_count} attempts already). The owner must decide whether to "
            "keep pursuing or close." + (f"\n\nThe model drafted this follow-up:\n{drafted}" if drafted else "")
        )
        _override(
            result,
            f"Escalated instead of flag_reengage: the ticket already has {reengage_count} re-engage "
            f"attempts (limit {REENGAGE_LIMIT}).",
        )

    if result["classification"] == "flag_reengage" and not (result.get("draft_message") or "").strip():
        result["classification"] = "manual_triage"
        _override(result, "Downgraded from flag_reengage: no draft_message was provided.")

    if result["classification"] == "flag_solve" and result["confidence"] < SOLVE_CONFIDENCE_FLOOR:
        result["classification"] = "manual_triage"
        _override(
            result,
            f"Downgraded from flag_solve: confidence {result['confidence']} "
            f"is below the {SOLVE_CONFIDENCE_FLOOR} floor.",
        )

    if result["classification"] != "escalate_leadership" and result.get("leadership_recipient"):
        result["leadership_recipient"] = None
        _override(result, "Cleared leadership_recipient: only escalate_leadership names one.")

    # A tag that disagreed with the model's own classification is an override
    # worth recording; one that only trails a guard's change above is not.
    if model_tag is not None and model_tag != model_classification:
        _override(result, f"Set tag to {result['classification']} (model gave {model_tag!r}).")
    result["tag"] = result["classification"]

    return result
