"""Weekly review of misclassifications and the lessons learned from them.

Entries reach misclassification_log from the Decision Engine tab's Mark
Misclassified… button. Here a reviewer picks a week, marks its entries
reviewed, and writes a lessons_learned summary built on the reviewed ones,
which someone then approves. Every Mongo call goes through the tab's job
runner, so it runs on a worker thread and never overlaps another tab job.
"""

import getpass
from datetime import datetime, timezone
from typing import Any, Callable, List, Optional, Tuple

from ui import zendesk_links
from decision_engine import feedback
from loguru import logger
from pymongo.collection import Collection
from PyQt5 import QtCore
from PyQt5.QtCore import QDate
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QDateEdit,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from ui.widgets import Button, Card, hint, LineEdit, use_si_scrollbars

LOG_COLUMNS = [
    "Ticket",
    "Logged (UTC)",
    "Classified as",
    "Should be",
    "Misread signal",
    "Reviewed by",
]
LESSON_COLUMNS = ["Generated (UTC)", "Status", "Entries", "Summary"]

# run(task, description, on_done, on_error) -> False if another job is running.
Runner = Callable[
    [Callable[[], Any], str, Callable[[Any], None], Optional[Callable[[str], None]]],
    bool,
]


def _fmt(moment: Optional[datetime]) -> str:
    return f"{moment:%Y-%m-%d %H:%M}" if moment else ""


def _default_reviewer() -> str:
    try:
        return getpass.getuser()
    except Exception:  # no login name (some containers)
        return ""


class WeeklyReviewDialog(QDialog):
    """Lists one week's misclassification_log and lessons_learned entries and edits them.

    Args:
        run: The tab's job runner (TriageTab._run).
        collections: Returns (misclassification_log, lessons_learned); called on the worker thread.
    """

    def __init__(
        self,
        run: Runner,
        collections: Callable[[], Tuple[Collection, Collection]],
        parent: Optional[QWidget] = None,
        week: Optional[datetime] = None,
    ) -> None:
        super().__init__(parent)
        self.run = run
        self.collections = collections
        self.entries: List[dict] = []
        self.lessons: List[dict] = []
        self._busy = False
        self.setWindowTitle("Weekly Review")
        self.resize(900, 700)
        self._build_ui()
        monday = feedback.week_of(week or datetime.now(timezone.utc))
        self.week_input.setDate(QDate(monday.year, monday.month, monday.day))
        self.week_input.dateChanged.connect(lambda _: self.load())
        self.load()

    # --- layout ---------------------------------------------------------------

    def _build_ui(self) -> None:
        self.week_input = QDateEdit()
        self.week_input.setCalendarPopup(True)
        self.week_input.setDisplayFormat("yyyy-MM-dd")
        self.prev_button = Button("◀ Previous Week")
        self.prev_button.clicked.connect(lambda: self._shift_week(-7))
        self.next_button = Button("Next Week ▶")
        self.next_button.clicked.connect(lambda: self._shift_week(7))
        self.reviewer_input = LineEdit("Reviewer", _default_reviewer())
        self.reviewer_input.setPlaceholderText("Your name")
        self.reviewer_input.setMinimumWidth(240)
        self.refresh_button = Button("Refresh")
        self.refresh_button.clicked.connect(self.load)
        week_row = QHBoxLayout()
        week_row.addWidget(QLabel("Week of"))
        week_row.addWidget(self.week_input)
        week_row.addWidget(self.prev_button)
        week_row.addWidget(self.next_button)
        week_row.addStretch()
        week_row.addWidget(self.reviewer_input)
        week_row.addWidget(self.refresh_button)

        self.status_label = hint("")

        self.log_table = self._table(
            LOG_COLUMNS, 4, QAbstractItemView.ExtendedSelection
        )
        self.log_table.itemSelectionChanged.connect(self._update_buttons)
        self.mark_reviewed_button = Button("Mark Selected Reviewed")
        self.mark_reviewed_button.clicked.connect(self.mark_selected_reviewed)
        self.zendesk_button = Button(zendesk_links.BUTTON_TEXT)
        self.zendesk_button.setToolTip(zendesk_links.BUTTON_TIP)
        self.zendesk_button.clicked.connect(self.view_selected_on_zendesk)
        log_buttons = QHBoxLayout()
        log_buttons.addWidget(self.zendesk_button)
        log_buttons.addStretch()
        log_buttons.addWidget(self.mark_reviewed_button)
        log_box = Card("Misclassifications")
        log_layout = log_box.body
        log_layout.addWidget(self.log_table)
        log_layout.addLayout(log_buttons)

        self.lesson_table = self._table(
            LESSON_COLUMNS, 3, QAbstractItemView.SingleSelection
        )
        self.lesson_table.itemSelectionChanged.connect(self._show_selected_lesson)
        self.lesson_view = use_si_scrollbars(QTextEdit())
        self.lesson_view.setReadOnly(True)
        self.lesson_view.setPlaceholderText("Select a summary to read it.")
        self.approve_button = Button("Approve Selected")
        self.approve_button.clicked.connect(self.approve_selected)
        self.draft_input = use_si_scrollbars(QTextEdit())
        self.draft_input.setPlaceholderText(
            "What should the engine learn from this week's reviewed misclassifications?"
        )
        self.draft_input.textChanged.connect(self._update_buttons)
        self.suggest_button = Button("Suggest from Reviewed")
        self.suggest_button.clicked.connect(self.suggest_summary)
        self.save_draft_button = Button("Save Draft")
        self.save_draft_button.clicked.connect(self.save_draft)
        approve_row = QHBoxLayout()
        approve_row.addStretch()
        approve_row.addWidget(self.approve_button)
        lesson_buttons = QHBoxLayout()
        lesson_buttons.addStretch()
        lesson_buttons.addWidget(self.suggest_button)
        lesson_buttons.addWidget(self.save_draft_button)
        lesson_box = Card("Lessons Learned")
        lesson_layout = lesson_box.body
        lesson_layout.addWidget(self.lesson_table)
        lesson_layout.addWidget(self.lesson_view)
        lesson_layout.addLayout(approve_row)
        lesson_layout.addWidget(
            hint("New summary (built on the week's reviewed entries):")
        )
        lesson_layout.addWidget(self.draft_input)
        lesson_layout.addLayout(lesson_buttons)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(12)
        layout.addLayout(week_row)
        layout.addWidget(self.status_label)
        layout.addWidget(log_box, stretch=1)
        layout.addWidget(lesson_box, stretch=1)
        self._update_buttons()

    @staticmethod
    def _table(columns: List[str], stretch: int, selection) -> QTableWidget:
        table = use_si_scrollbars(QTableWidget(0, len(columns)))
        table.setHorizontalHeaderLabels(columns)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setSelectionMode(selection)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setSectionResizeMode(stretch, QHeaderView.Stretch)
        return table

    # --- state ------------------------------------------------------------------

    def week(self) -> datetime:
        """Midnight UTC on the Monday of the chosen week."""
        date = self.week_input.date()
        return feedback.week_of(
            datetime(date.year(), date.month(), date.day(), tzinfo=timezone.utc)
        )

    def _shift_week(self, days: int) -> None:
        self.week_input.setDate(self.week_input.date().addDays(days))

    def _selected_rows(self, table: QTableWidget) -> List[int]:
        return sorted({index.row() for index in table.selectionModel().selectedRows()})

    def _selected_entries(self) -> List[dict]:
        return [self.entries[row] for row in self._selected_rows(self.log_table)]

    def view_selected_on_zendesk(self) -> None:
        """Opens the selected entry's ticket in Zendesk (one entry selected)."""
        entries = self._selected_entries()
        if len(entries) == 1:
            zendesk_links.open_ticket(entries[0].get("ticket_id"))

    def _selected_lesson(self) -> Optional[dict]:
        rows = self._selected_rows(self.lesson_table)
        return self.lessons[rows[0]] if rows else None

    def _update_buttons(self) -> None:
        idle = not self._busy
        for widget in (
            self.week_input,
            self.prev_button,
            self.next_button,
            self.refresh_button,
        ):
            widget.setEnabled(idle)
        self.mark_reviewed_button.setEnabled(
            idle and any(not e.get("reviewed") for e in self._selected_entries())
        )
        self.zendesk_button.setEnabled(len(self._selected_entries()) == 1)
        lesson = self._selected_lesson()
        self.approve_button.setEnabled(
            idle and lesson is not None and not lesson.get("approved")
        )
        has_reviewed = any(e.get("reviewed") for e in self.entries)
        self.suggest_button.setEnabled(idle and has_reviewed)
        self.save_draft_button.setEnabled(
            idle and has_reviewed and bool(self.draft_input.toPlainText().strip())
        )

    def _reviewer(self, action: str) -> Optional[str]:
        name = self.reviewer_input.text().strip()
        if not name:
            QMessageBox.warning(
                self, "Weekly Review", f"Enter your name as Reviewer to {action}."
            )
            self.reviewer_input.setFocus()
            return None
        return name

    def _job(
        self, task: Callable[[], Any], description: str, on_done: Callable[[Any], None]
    ) -> None:
        """Runs `task` through the tab, with this dialog's controls disabled meanwhile."""

        def finish(result):
            self._busy = False
            on_done(result)
            self._update_buttons()

        def failed(message):
            self._busy = False
            self.status_label.setText(f"{description} failed; see the log.")
            self._update_buttons()

        self._busy = True
        self._update_buttons()
        self.status_label.setText(f"{description}…")
        if not self.run(task, description, finish, failed):
            self._busy = False
            self.status_label.setText(
                "Another Decision Engine job is running; try again when it finishes."
            )
            self._update_buttons()

    # --- actions ----------------------------------------------------------------

    def load(self) -> None:
        """Reads the week's misclassifications and lessons."""
        week = self.week()

        def read():
            log, lessons = self.collections()
            return feedback.misclassifications_for_week(
                log, week
            ), feedback.lessons_for_week(lessons, week)

        self._job(
            read,
            f"Loading the week of {week:%Y-%m-%d}",
            lambda found: self._show(week, *found),
        )

    def _show(self, week: datetime, entries: List[dict], lessons: List[dict]) -> None:
        if week != self.week():
            # The week changed while this one loaded; show the newer one instead.
            self.load()
            return
        self.entries, self.lessons = entries, lessons
        reviewed = sum(1 for e in entries if e.get("reviewed"))
        approved = sum(1 for lesson in lessons if lesson.get("approved"))
        self.status_label.setText(
            f"Week of {week:%Y-%m-%d}: {len(entries)} misclassification(s), {reviewed} reviewed; "
            f"{len(lessons)} summary(ies), {approved} approved."
        )
        self._fill(
            self.log_table,
            [
                [
                    e.get("ticket_id"),
                    _fmt(e.get("logged_at")),
                    e.get("agent_classification"),
                    e.get("correct_classification"),
                    e.get("misread_signal"),
                    e.get("reviewed_by") if e.get("reviewed") else "not reviewed",
                ]
                for e in entries
            ],
        )
        self._fill(
            self.lesson_table,
            [
                [
                    _fmt(lesson.get("generated_at")),
                    (
                        f"Approved by {lesson.get('approved_by')}"
                        if lesson.get("approved")
                        else "Draft"
                    ),
                    len(lesson.get("based_on_log_ids") or []),
                    (
                        (lesson.get("summary_text") or "").splitlines()[0]
                        if lesson.get("summary_text")
                        else ""
                    ),
                ]
                for lesson in lessons
            ],
        )
        self.lesson_view.clear()

    @staticmethod
    def _fill(table: QTableWidget, rows: List[list]) -> None:
        table.clearSelection()
        table.setRowCount(len(rows))
        for r, values in enumerate(rows):
            for c, value in enumerate(values):
                table.setItem(
                    r, c, QTableWidgetItem("" if value is None else str(value))
                )
        table.resizeColumnsToContents()

    def _show_selected_lesson(self) -> None:
        lesson = self._selected_lesson()
        if lesson is None:
            self.lesson_view.clear()
        else:
            based_on = len(lesson.get("based_on_log_ids") or [])
            approval = (
                f"Approved by {lesson.get('approved_by')} at {_fmt(lesson.get('approved_at'))} UTC"
                if lesson.get("approved")
                else "Draft, not approved"
            )
            self.lesson_view.setPlainText(
                f"{approval}. Built on {based_on} reviewed entry(ies).\n\n{lesson.get('summary_text') or ''}"
            )
        self._update_buttons()

    def mark_selected_reviewed(self) -> None:
        reviewer = self._reviewer("mark entries reviewed")
        if reviewer is None:
            return
        ids = [e["_id"] for e in self._selected_entries() if not e.get("reviewed")]
        if not ids:
            return

        def mark():
            log, _ = self.collections()
            return feedback.mark_misclassifications_reviewed(log, ids, reviewer)

        def done(count):
            logger.success(
                f"{count} misclassification(s) marked reviewed by {reviewer}."
            )
            self.load()

        self._job(mark, f"Marking {len(ids)} entry(ies) reviewed", done)

    def suggest_summary(self) -> None:
        """Fills the draft with a summary of the reviewed entries, unless something is typed there."""
        suggestion = feedback.suggest_lessons_summary(self.entries)
        if self.draft_input.toPlainText().strip():
            answer = QMessageBox.question(
                self,
                "Weekly Review",
                "Replace the summary you've written with a suggested one?",
                QMessageBox.Yes | QMessageBox.Cancel,
                QMessageBox.Cancel,
            )
            if answer != QMessageBox.Yes:
                return
        self.draft_input.setPlainText(suggestion)

    def save_draft(self) -> None:
        """Stores the summary as an unapproved lessons_learned entry for the week."""
        text = self.draft_input.toPlainText().strip()
        if not text:
            return
        week = self.week()

        def draft():
            log, lessons = self.collections()
            return feedback.draft_lessons_learned(log, lessons, week, text)

        def done(_):
            logger.success(f"Lessons learned drafted for the week of {week:%Y-%m-%d}.")
            self.draft_input.clear()
            self.load()

        self._job(draft, "Saving the draft", done)

    def approve_selected(self) -> None:
        lesson = self._selected_lesson()
        if lesson is None or lesson.get("approved"):
            return
        approver = self._reviewer("approve a summary")
        if approver is None:
            return
        answer = QMessageBox.question(
            self,
            "Approve Summary",
            f"Approve this summary as {approver}?",
            QMessageBox.Yes | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )
        if answer != QMessageBox.Yes:
            return

        def approve():
            _, lessons = self.collections()
            return feedback.approve_lessons_learned(lessons, lesson["_id"], approver)

        def done(approved):
            if not approved:
                QMessageBox.information(
                    self, "Approve Summary", "That summary was already approved."
                )
            self.load()

        self._job(approve, "Approving the summary", done)
