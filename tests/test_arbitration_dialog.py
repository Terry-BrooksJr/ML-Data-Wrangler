from datetime import datetime, timezone

import pytest
from bson import ObjectId
from PyQt5.QtWidgets import QApplication

from decision_engine import triage
from ui.dialogs.arbitration_dialog import ArbitrationDialog, has_results, results_text

AT = datetime(2026, 11, 4, 22, 30, tzinfo=timezone.utc)
JIRA_ID = ObjectId()


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def rerun_row(classification="flag_reengage", **extra):
    """Ticket 30906's shape: a guard-downgraded review re-run with evidence."""
    review = {
        "review_id": "rerun",
        "reviewed_at": AT,
        "model_version": "claude-opus-5.5",
        "classification": classification,
        "confidence": 80,
        "signals_matched": ["R1", "R2"],
        "reengage_attempts": ["2026-09-24T03:00:00Z"],
        "reasoning": "LRN-52770 only backpatches the flash fix.",
        "open_questions": [],
        "internal_note": "",
        "draft_message": "Hi Haley, following up…",
        "guard_overrides": [],
        "rerun_of": "first",
        "evidence_ids": [JIRA_ID],
        **extra,
    }
    return {
        "zendesk_id": "30906",
        "subject": "Imagetool Issue",
        "review": review,
        "earlier_reviews": [
            {
                "review_id": "first",
                "reviewed_at": AT,
                "classification": "manual_triage",
                "confidence": 65,
                "model_version": "claude-sonnet-5.5",
                "rerun_of": None,
            }
        ],
        "case": {
            "status": triage.RESOLVED,
            "open_questions": ["What does LRN-52770 cover?"],
            "evidence": [
                {
                    "evidence_id": JIRA_ID,
                    "kind": "jira",
                    "data": {"issues": [{"key": "LRN-52770", "status": "Released", "resolution": "Done"}]},
                }
            ],
        },
    }


def test_triggers_keep_the_names_of_the_prompt_that_matched_them():
    old = results_text(rerun_row(signals_matched=["R4"], audit={"prompt_version": "c33981b6d2fe"}))
    assert "  - R4 — Linked Jira issue idle for 14 days" in old
    new = results_text(rerun_row(signals_matched=["R4"]))
    assert "  - R4 — Linked Jira or Aha item idle for 6 months" in new


def test_only_reviews_past_one_model_have_results():
    assert not has_results({"classification": "flag_solve"})
    assert has_results({"confirmation_of": {"classification": "manual_triage"}})
    assert has_results({"arbitration": {"model": "arb"}})
    assert has_results({"rerun_of": "first"})


def test_a_rerun_that_changed_the_classification():
    text = results_text(rerun_row())
    assert (
        "Yes: manual_triage (confidence 65, claude-sonnet-5.5) became "
        "flag_reengage (confidence 80, claude-opus-5.5), on a re-run with 1 piece(s) "
        "of reviewer evidence." in text
    )
    assert "  - Jira LRN-52770" in text
    assert "Open questions the re-run had to answer:\n  - What does LRN-52770 cover?" in text
    assert "LRN-52770 only backpatches the flash fix." in text
    # One attempt counted in the stall, so this one is the last allowed.
    assert "attempt 2 of 2 in the current stall, so it should say it is the final follow-up." in text
    assert "Hi Haley, following up…" in text
    assert "Then mark Review Added to Zendesk once it's posted and mark the review Correct or Misclassified." in text


def test_a_rerun_still_manual_triage_lists_what_would_settle_it():
    row = rerun_row(
        "manual_triage", open_questions=["Does LRN-52977 cover the jump?"], draft_message=None
    )
    row["case"]["status"] = triage.OPEN
    text = results_text(row, added=True, decided=True)
    assert "No: still manual_triage" in text
    assert "  - Does LRN-52977 cover the jump?" in text
    assert "then use Re-run with Evidence." in text
    assert "Then mark" not in text


def test_an_arbitrated_disagreement():
    row = {
        "zendesk_id": "1",
        "review": {
            "classification": "manual_triage",
            "confidence": 60,
            "model_version": "conf-model",
            "confirmation_of": {"classification": "flag_solve", "confidence": 85, "model_version": "first-model"},
            "arbitration": {
                "model": "arbiter",
                "preferred": "flag_solve",
                "preferred_reasoning": "S3 is met.",
                "ruling": "on_track",
                "error": None,
                "confirmation": {"classification": "on_track", "confidence": 60, "model_version": "conf-model"},
            },
            "guard_overrides": ["Sent to manual triage: Models disagreed."],
            "open_questions": [],
        },
    }
    text = results_text(row)
    assert "Yes: flag_solve (confidence 85, first-model) became manual_triage" in text
    assert "after the confirmation model disagreed and the arbiter ruled" in text
    assert "  - Confirmation: on_track (confidence 60, conf-model)" in text
    assert "preferred flag_solve on the reasoning, ruled on_track on the ticket" in text
    assert "S3 is met." in text
    assert "A code check changed the model's own answer: Sent to manual triage" in text
    assert "(none recorded: see the reasoning above)" in text


def test_the_dialog_shows_plain_text(qapp):
    dialog = ArbitrationDialog("<b>not bold</b>", "30906")
    assert dialog.text.toPlainText() == "<b>not bold</b>"
    assert dialog.windowTitle() == "Arbitration Results: Ticket 30906"


def test_a_later_rerun_is_compared_with_the_review_just_before_it():
    # manual_triage -> flag_solve (re-run 1) -> manual_triage (re-run 2): both
    # re-runs name the case's first review in rerun_of.
    row = rerun_row("manual_triage", rerun_of="first", draft_message=None)
    row["earlier_reviews"].insert(
        0,
        {"review_id": "rerun1", "classification": "flag_solve", "confidence": 85,
         "model_version": "claude-opus-5.5", "rerun_of": "first"},
    )
    text = results_text(row)
    assert "Yes: flag_solve (confidence 85, claude-opus-5.5) became manual_triage" in text


def test_attempts_count_like_the_engines_limit_check():
    from ui.dialogs.arbitration_dialog import attempts_so_far

    # Older reviews: the ticket's recorded attempts count when the model cited none.
    row = rerun_row(reengage_attempts=[])
    row["reengage_attempts"] = ["2026-09-24T03:00:00Z"]
    assert attempts_so_far(row) == 1
    assert "attempt 2 of 2 in the current stall, so it should say it is the final follow-up." in results_text(row)
    # Newer reviews store the count guards checked.
    row["review"]["reengage_attempt_count"] = 0
    assert attempts_so_far(row) == 0
