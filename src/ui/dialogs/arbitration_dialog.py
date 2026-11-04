"""Arbitration Results: how a review's classification was reached.

For a review that went past one model's answer (a confirmation, the
arbiter, or a re-run with reviewer evidence), says whether the
classification changed, what was weighed, and what to do next on the
ticket. Plain text throughout: the reasoning and draft quote customer text.
"""

from typing import List, Optional

from decision_engine import evidence, triage
from decision_engine.guards import describe_signal, REENGAGE_LIMIT
from PyQt5.QtWidgets import QApplication, QDialog, QDialogButtonBox, QTextEdit, QVBoxLayout, QWidget
from ui.widgets import Button, heading, use_si_scrollbars


def has_results(review: dict) -> bool:
    """Whether the review went past a single model's answer."""
    return bool(
        review.get("arbitration")
        or review.get("confirmation_of")
        or review.get("rerun_of") is not None
    )


def _answer(answer: Optional[dict], model_key: str = "model_version") -> str:
    if not answer:
        return "(not recorded)"
    return (
        f"{answer.get('classification')} (confidence {answer.get('confidence')}, "
        f"{answer.get(model_key) or 'model not recorded'})"
    )


def _before(row: dict) -> Optional[dict]:
    """The answer this review replaced: the re-run review's, or the first model's."""
    review = row["review"]
    if review.get("rerun_of") is not None:
        rerun_of = review["rerun_of"]
        for earlier in row.get("earlier_reviews") or []:
            if earlier.get("review_id") == rerun_of:
                return earlier
        return None
    return review.get("confirmation_of")


def _changed(row: dict) -> List[str]:
    review = row["review"]
    before = _before(row)
    final = _answer(review)
    if review.get("rerun_of") is not None:
        how = f"on a re-run with {len(review.get('evidence_ids') or [])} piece(s) of reviewer evidence"
    elif review.get("arbitration"):
        how = "after the confirmation model disagreed and the arbiter ruled"
    else:
        how = "after the confirmation model re-checked it"
    if before is None:
        lines = [f"Final: {final} {how}. The earlier answer isn't recorded."]
    elif before.get("classification") != review.get("classification"):
        lines = [f"Yes: {_answer(before)} became {final}, {how}."]
    else:
        lines = [f"No: still {review.get('classification')}, {how}. Now {final}; before {_answer(before)}."]
    for override in review.get("guard_overrides") or []:
        lines.append(f"A code check changed the model's own answer: {override}")
    return lines


def _considered(row: dict) -> List[str]:
    review = row["review"]
    case = row.get("case") or {}
    lines = []
    arbitration = review.get("arbitration") or {}
    if arbitration:
        lines += [
            "Answers weighed:",
            f"  - First answer: {_answer(review.get('confirmation_of'))}",
            f"  - Confirmation: {_answer(arbitration.get('confirmation'))}",
            f"  - Arbiter ({arbitration.get('model') or 'none set'}): preferred "
            f"{arbitration.get('preferred') or '-'} on the reasoning, ruled "
            f"{arbitration.get('ruling') or '-'} on the ticket"
            + (f" (failed: {arbitration['error']})" if arbitration.get("error") else ""),
        ]
        if arbitration.get("preferred_reasoning"):
            lines += ["", "Arbiter's reasoning for its preference:", arbitration["preferred_reasoning"]]
    elif review.get("confirmation_of"):
        lines += [
            "Answers weighed:",
            f"  - First answer: {_answer(review.get('confirmation_of'))}",
            f"  - Confirmation: {_answer(review)}",
        ]
    if review.get("rerun_of") is not None:
        used = set(map(str, review.get("evidence_ids") or []))
        entries = [e for e in case.get("evidence") or [] if str(e.get("evidence_id")) in used]
        lines += ["Reviewer evidence:"]
        lines += [f"  - {evidence.describe(e)}" for e in entries] or [
            f"  - {len(used)} piece(s); not on the loaded case"
        ]
        # A resolved case keeps the questions it was opened with; an open
        # one holds the re-run's own, which Next Steps lists.
        if case.get("status") == triage.RESOLVED and case.get("open_questions"):
            lines += ["Open questions the re-run had to answer:"]
            lines += [f"  - {q}" for q in case["open_questions"]]
    lines += ["", "Triggers matched:"]
    lines += [f"  - {describe_signal(s)}" for s in review.get("signals_matched") or []] or ["  (none)"]
    lines += ["", "Final reasoning:", review.get("reasoning") or "(none)"]
    return lines


def _next_steps(row: dict, added: bool, decided: bool) -> List[str]:
    review = row["review"]
    classification = review.get("classification")
    note = review.get("internal_note")
    lines = []
    if review.get("engine_error"):
        lines.append("The engine failed on this ticket: review it by hand, or classify it again.")
    elif classification == "flag_reengage":
        attempt = len(review.get("reengage_attempts") or []) + 1
        lines.append(
            f"Check the draft below, then send it to the customer. This is re-engagement "
            f"attempt {attempt} of {REENGAGE_LIMIT} in the current stall"
            + (", so it should say it is the final follow-up." if attempt >= REENGAGE_LIMIT else ".")
        )
        if review.get("draft_message"):
            lines += ["", "Draft message:", review["draft_message"]]
    elif classification == "flag_solve":
        lines.append("Solve the ticket in Zendesk.")
    elif classification == "escalate_leadership":
        lines.append(
            f"Escalate to {review.get('leadership_recipient') or 'Leadership'} with the internal note."
        )
    elif classification == "escalation_internal":
        lines.append("Hand the ticket to its Support owner with the internal note.")
    elif classification == "on_track":
        lines.append("No action is needed now: the ticket is progressing.")
    elif classification == "manual_triage":
        questions = review.get("open_questions") or (row.get("case") or {}).get("open_questions") or []
        lines.append("A person needs to decide this ticket. Settle these open questions:")
        lines += [f"  - {q}" for q in questions] or ["  (none recorded: see the reasoning above)"]
        if (row.get("open_case") or row.get("case") or {}).get("status") == "open":
            lines.append("Add Jira or Aha evidence that answers them, then use Re-run with Evidence.")
    if note and classification != "flag_reengage":
        lines += ["", "Internal note:", note]
    after = []
    if not added:
        after.append("mark Review Added to Zendesk once it's posted")
    if not decided:
        after.append("mark the review Correct or Misclassified")
    if after:
        lines += ["", "Then " + " and ".join(after) + "."]
    return lines


def results_text(row: dict, added: bool = False, decided: bool = False) -> str:
    """The pop-up's text for a review row (triage_tab's row shape).

    Args:
        added: The review was already marked added to Zendesk.
        decided: The review was already marked correct or misclassified.
    """
    review = row["review"]
    return "\n".join(
        [
            f"Ticket {row.get('zendesk_id')}: {row.get('subject') or '(no subject)'}",
            f"Classification: {_answer(review)}",
            "",
            "WAS THE CLASSIFICATION CHANGED?",
            *_changed(row),
            "",
            "WHAT WAS CONSIDERED",
            *_considered(row),
            "",
            "NEXT STEPS",
            *_next_steps(row, added, decided),
        ]
    )


class ArbitrationDialog(QDialog):
    """Shows results_text for one review, with a Copy button."""

    def __init__(self, text: str, zendesk_id: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Arbitration Results: Ticket {zendesk_id}")
        self.resize(720, 640)
        self.text = use_si_scrollbars(QTextEdit())
        self.text.setReadOnly(True)
        self.text.setPlainText(text)  # never rich text: it quotes customers
        copy = Button("Copy")
        copy.clicked.connect(lambda: QApplication.clipboard().setText(text))
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.close)
        buttons.addButton(copy, QDialogButtonBox.ActionRole)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(10)
        layout.addWidget(heading("How this classification was reached"))
        layout.addWidget(self.text, stretch=1)
        layout.addWidget(buttons)
