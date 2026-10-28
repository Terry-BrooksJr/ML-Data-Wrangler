"""The taxonomy as the prompt states it, and the reply fields it asks for."""

import re

import pytest
from decision_engine import decisions
from decision_engine.guards import CLASSIFICATIONS, InvalidReply, validate_reply
from decision_engine.prompts import SYSTEM_INSTRUCTIONS

TRIGGER_ID = re.compile(r"^\s+- ([LORS]\d+):", re.MULTILINE)


def reply(**extra):
    return {"classification": "on_track", "confidence": 80, "reasoning": "", **extra}


def test_every_classification_is_in_the_prompt_and_the_decision_choices():
    for name in CLASSIFICATIONS:
        assert f'"{name}"' in SYSTEM_INSTRUCTIONS
        assert name in decisions.CHOICE_LABELS


def test_trigger_ids_are_unique_and_numbered_in_order():
    ids = TRIGGER_ID.findall(SYSTEM_INSTRUCTIONS)
    assert len(ids) == len(set(ids))
    for prefix, count in (("L", 9), ("O", 7), ("R", 7), ("S", 7)):
        assert [i for i in ids if i[0] == prefix] == [
            f"{prefix}{n}" for n in range(1, count + 1)
        ]


def test_decision_rubric_keeps_the_new_sections():
    assert "## Re-engagement" in decisions.INSTRUCTIONS
    assert "## Time and status" in decisions.INSTRUCTIONS
    assert "## Output format" not in decisions.INSTRUCTIONS


def test_reengage_attempts_default_to_empty():
    assert validate_reply(reply())["reengage_attempts"] == []
    attempts = ["2026-10-01T10:00:00+00:00"]
    assert validate_reply(reply(reengage_attempts=attempts))["reengage_attempts"] == attempts


@pytest.mark.parametrize("bad", ["2026-10-01", [1], None])
def test_reengage_attempts_must_be_a_list_of_strings(bad):
    with pytest.raises(InvalidReply, match="reengage_attempts"):
        validate_reply(reply(reengage_attempts=bad))
