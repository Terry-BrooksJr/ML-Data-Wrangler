"""Checks on the model's reply, enforced in code rather than trusted to the prompt.

validate_reply rejects a reply the engine can't use (missing keys, wrong
types, unknown classification); the caller treats that as an engine error.
apply_guards then overrides a valid reply that breaks a hard rule, recording
each override in `reasoning` and `guard_overrides`.
"""

import re
from datetime import datetime
from typing import Iterable, List, Optional, Set

from . import leadership

CLASSIFICATIONS = (
    "escalate_leadership",
    "escalation_internal",
    "flag_reengage",
    "flag_solve",
    "on_track",
    "manual_triage",
)
# Confidence bars from the prompt, enforced here as a hard backstop rather
# than trusted from the model's self-report.
SOLVE_CONFIDENCE_FLOOR = 80
LEADERSHIP_CONFIDENCE_FLOOR = 80  # below it: the owner, with a visibility note
ACTION_CONFIDENCE_FLOOR = 70  # escalation_internal and flag_reengage
REENGAGE_LIMIT = 2  # the prompt's ceiling: at most two attempts per stall
# What an internal note starts with when Leadership should see a ticket the
# owner handles (the prompt asks for it too).
LEADERSHIP_VISIBILITY = "Leadership visibility:"
# The trigger IDs (prompts.py) that can justify each category. An owner
# escalation may cite a Leadership trigger: the prompt sends a Leadership
# case below its confidence bar to the owner.
TRIGGER_PREFIXES = {
    "escalate_leadership": "L",
    "escalation_internal": "OL",
    "flag_reengage": "R",
    "flag_solve": "S",
}
# How many triggers each letter has in the prompt (L1-L9, ...).
TRIGGER_COUNTS = {"L": 9, "O": 7, "R": 7, "S": 7}
_TRIGGER_ID = re.compile(r"\s*([LORS])(\d+)\b")
# A short name for each trigger in prompts.py, shown beside its ID.
TRIGGER_NAMES = {
    "L1": "Closing would bypass customer communication",
    "L2": "Customer Success Manager needed",
    "L3": "Ownership moves outside Support",
    "L4": "Product concern for product or executive leadership",
    "L5": "Platform-level risk",
    "L6": "Genuine churn risk",
    "L7": "Contractual obligation or exposure",
    "L8": "Customer data risk",
    "L9": "SLA breach",
    "O1": "Customer question unanswered for 48 business hours",
    "O2": "Linked Jira bug idle for 28 days",
    "O3": "Blocked on an internal dependency",
    "O4": "Re-engage attempts exhausted",
    "O5": "Customer frustration or tone escalation",
    "O6": "Account-level action needed",
    "O7": "Closing needs owner confirmation",
    "R1": "No reply to Support's request in 72 business hours",
    "R2": "Customer reply missing what was asked",
    "R3": "Internal party silent for 48 business hours",
    "R4": "Linked Jira issue idle for 14 days",
    "R5": "Promised follow-up date passed",
    "R6": "No public comment for 72 business hours",
    "R7": "Scheduled action has come due",
    "S1": "Customer confirms no further help needed",
    "S2": "Investigation results delivered",
    "S3": "Linked feature released",
    "S4": "Linked defect resolved or Behaves as Expected",
    "S5": "Feature request marked Won't Do",
    "S6": "Defined task completed",
    "S7": "All requests and questions addressed",
}

_OPTIONAL_TEXT = (
    "internal_note",
    "draft_message",
    "leadership_recipient",
    "tag",
    "ticket_id",
)


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
        problems.append(
            f"classification ({describe_value(reply.get('classification'))}) is not one of "
            f"{', '.join(CLASSIFICATIONS)}"
        )

    confidence = reply.get("confidence")
    # bool is an int subclass; a string "80" is rejected rather than guessed at.
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        problems.append(
            f"confidence must be a number, got {describe_value(confidence)}"
        )
    elif not 0 <= confidence <= 100:
        problems.append("confidence is outside 0-100")

    if not isinstance(reply.get("reasoning"), str):
        problems.append("reasoning must be a string")

    signals = reply.get("signals_matched", [])
    if not isinstance(signals, list) or not all(isinstance(s, str) for s in signals):
        problems.append("signals_matched must be a list of strings")

    attempts = reply.get("reengage_attempts", [])
    if not isinstance(attempts, list) or not all(isinstance(a, str) for a in attempts):
        problems.append("reengage_attempts must be a list of strings")

    for key in _OPTIONAL_TEXT:
        if reply.get(key) is not None and not isinstance(reply[key], str):
            problems.append(f"{key} must be a string or null")

    if problems:
        raise InvalidReply("; ".join(problems))

    result = dict(reply)
    result["confidence"] = int(round(confidence))
    result["signals_matched"] = signals
    result["reengage_attempts"] = attempts
    for key in _OPTIONAL_TEXT:
        result.setdefault(key, None)
    return result


def _override(result: dict, message: str) -> None:
    result.setdefault("guard_overrides", []).append(message)
    result["reasoning"] = f"{message} Original reasoning: {result.get('reasoning', '')}"


def _route_leadership(result: dict, assignee: Optional[str]) -> None:
    """Sets leadership_recipient from the assignee (leadership.recipients_for),
    and makes the internal note open with the routing review when one is needed."""
    expected, review = leadership.recipients_for(assignee)
    given = result.get("leadership_recipient")
    if set(leadership.parse_recipients(given)) != set(expected):
        _override(
            result,
            f"Set leadership_recipient to {leadership.format_recipients(expected) or 'nobody'} "
            f"(model gave {given!r}): recipients follow the assignee.",
        )
    # Same people in another order is tidied without an override.
    result["leadership_recipient"] = leadership.format_recipients(expected)
    note = result.get("internal_note") or ""
    if review and not note.startswith(leadership.REVIEW_PREFIX):
        result["internal_note"] = f"{leadership.REVIEW_PREFIX} {review}" + (
            f"\n\n{note}" if note else ""
        )
        _override(result, f"Added the routing review note: {review}")


def _instant(text) -> Optional[datetime]:
    """A timestamp string as a datetime, so "Z" and "+00:00" compare equal."""
    try:
        return datetime.fromisoformat(str(text).strip().replace("Z", "+00:00"))
    except ValueError:
        return None


def describe_signal(signal: str) -> str:
    """A signal with its trigger's name: "R1" -> "R1 — No reply to Support's
    request in 72 business hours". Text the model added after the ID is kept;
    a signal that isn't a known trigger ID is returned as is."""
    m = _TRIGGER_ID.match(signal)
    name = m and TRIGGER_NAMES.get(m.group(1) + str(int(m.group(2))))
    if not name:
        return signal
    rest = signal[m.end() :].strip(" :-—")
    return f"{m.group(1)}{int(m.group(2))} — {name}" + (f" ({rest})" if rest else "")


def cited_triggers(signals: Iterable[str]) -> Set[str]:
    """The letters of the real trigger IDs among signals_matched
    ("R1", "R1: no reply" -> "R"); an ID the prompt doesn't define (S99) is ignored."""
    return {
        m.group(1)
        for m in map(_TRIGGER_ID.match, signals)
        if m and 1 <= int(m.group(2)) <= TRIGGER_COUNTS[m.group(1)]
    }


def _verified_attempts(result: dict, support_comment_times) -> List[str]:
    """The cited re-engage attempts that are public comments from Support.

    With support_comment_times None (not known), every cited attempt counts.
    """
    cited = result.get("reengage_attempts") or []
    if support_comment_times is None:
        return list(cited)
    known = {_instant(t) for t in support_comment_times} - {None}
    verified, seen = [], set()
    for attempt in cited:
        instant = _instant(attempt)
        # Each comment counts once, however its time is written.
        if instant in known and instant not in seen:
            seen.add(instant)
            verified.append(attempt)
    if len(verified) < len(cited):
        _override(
            result,
            f"Ignored {len(cited) - len(verified)} cited re-engage attempt(s) that "
            "aren't distinct comments from Support.",
        )
    return verified


def apply_guards(
    result: dict,
    reengage_count: int = 0,
    assignee: Optional[str] = None,
    classification_only: bool = False,
    support_comment_times: Optional[Iterable[str]] = None,
) -> dict:
    """Overrides a validated reply that breaks a hard rule.

    Args:
        result: A reply that passed validate_reply. Modified in place.
        reengage_count: Re-engage attempts recorded outside the transcript
            (none are today); the larger of this and the verified attempts
            the model cites counts toward the limit.
        assignee: The ticket's assignee (Zendesk user ID) the reply was made
            for; it decides a leadership escalation's recipients. None means
            the ticket has no assignee.
        classification_only: The answer comes from a model that can't write
            text (decisions.decide), so a flag_reengage without a draft
            message is kept rather than downgraded, and no trigger IDs are
            required.
        support_comment_times: Timestamps of the ticket's public comments
            from Support; cited re-engage attempts must be among them.

    Returns:
        The same dict.
    """
    model_classification = result["classification"]
    model_tag = result.get("tag")
    confidence = result["confidence"]

    allowed = TRIGGER_PREFIXES.get(model_classification)
    if (
        allowed
        and not classification_only
        and not set(allowed) & cited_triggers(result.get("signals_matched") or [])
    ):
        result["classification"] = "manual_triage"
        _override(
            result,
            f"Downgraded from {model_classification}: no {' or '.join(allowed)} "
            "trigger ID in signals_matched.",
        )

    if result["classification"] == "flag_solve" and confidence < SOLVE_CONFIDENCE_FLOOR:
        result["classification"] = "manual_triage"
        _override(
            result,
            f"Downgraded from flag_solve: confidence {confidence} "
            f"is below the {SOLVE_CONFIDENCE_FLOOR} floor.",
        )

    if (
        result["classification"] in ("escalation_internal", "flag_reengage")
        and confidence < ACTION_CONFIDENCE_FLOOR
    ):
        _override(
            result,
            f"Downgraded from {result['classification']}: confidence {confidence} "
            f"is below the {ACTION_CONFIDENCE_FLOOR} floor.",
        )
        result["classification"] = "manual_triage"

    if (
        result["classification"] == "escalate_leadership"
        and confidence < LEADERSHIP_CONFIDENCE_FLOOR
    ):
        # A missed leadership concern costs more than a misrouted one: the
        # owner gets it, with a note Leadership will see.
        result["classification"] = "escalation_internal"
        suspected = sorted(
            s for s in result.get("signals_matched") or [] if s.strip().startswith("L")
        )
        note = result.get("internal_note") or ""
        if not note.startswith(LEADERSHIP_VISIBILITY):
            result["internal_note"] = (
                f"{LEADERSHIP_VISIBILITY} possible leadership concern "
                f"({', '.join(suspected) or 'unspecified'}), confidence {confidence}, "
                f"below the {LEADERSHIP_CONFIDENCE_FLOOR} bar."
                + (f"\n\n{note}" if note else "")
            )
        _override(
            result,
            f"Escalated to the owner instead of Leadership: confidence {confidence} "
            f"is below the {LEADERSHIP_CONFIDENCE_FLOOR} floor.",
        )

    attempts = _verified_attempts(result, support_comment_times)
    result["reengage_attempts"] = attempts
    attempt_count = max(reengage_count, len(attempts))
    if result["classification"] == "flag_reengage" and attempt_count >= REENGAGE_LIMIT:
        drafted = result.get("draft_message")
        result["classification"] = "escalation_internal"
        result["draft_message"] = None
        result["internal_note"] = (
            f"Re-engage limit reached ({attempt_count} attempts already). The owner must decide whether to "
            "keep pursuing or close."
            + (f"\n\nThe model drafted this follow-up:\n{drafted}" if drafted else "")
        )
        _override(
            result,
            f"Escalated instead of flag_reengage: the ticket already has {attempt_count} re-engage "
            f"attempts (limit {REENGAGE_LIMIT}).",
        )

    if (
        not classification_only
        and result["classification"] == "flag_reengage"
        and not (result.get("draft_message") or "").strip()
    ):
        result["classification"] = "manual_triage"
        _override(
            result, "Downgraded from flag_reengage: no draft_message was provided."
        )

    if result["classification"] != "escalate_leadership" and result.get(
        "leadership_recipient"
    ):
        result["leadership_recipient"] = None
        _override(
            result, "Cleared leadership_recipient: only escalate_leadership names one."
        )

    if result["classification"] == "escalate_leadership":
        _route_leadership(result, assignee)

    # A tag that disagreed with the model's own classification is an override
    # worth recording; one that only trails a guard's change above is not.
    if model_tag is not None and model_tag != model_classification:
        _override(
            result, f"Set tag to {result['classification']} (model gave {model_tag!r})."
        )
    result["tag"] = result["classification"]

    return result
