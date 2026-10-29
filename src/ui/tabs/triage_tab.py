"""Decision Engine tab for the PyQt app.

Shows how many tickets are awaiting review, runs the decision engine's own
entry points (run_review_batch, process_one_ticket) on a background thread,
and lists recent reviews read straight from Mongo. The engine is used as-is;
nothing here changes how tickets are classified or stored.
"""

import os
import threading
import time
from typing import Any, Callable, List, Optional

from embedding import embeddings
from decision_engine import ab_test, feedback, persistence, search, training_export
from decision_engine.zendesk_status import (
    ADDED_REVIEWS_KEY,
    mark_review_added,
    REVIEW_ADDED_KEY,
    review_added,
)
from decision_engine.guards import CLASSIFICATIONS, describe_signal
from ui.dialogs.ab_test_dialog import ab_collections, ABTestDialog
from ui import theme, zendesk_links
from ui.dialogs.lessons_dialog import WeeklyReviewDialog
from loguru import logger
from pymongo.collection import Collection
from pymongo.errors import ServerSelectionTimeoutError
from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSignal, QObject, QThread
from PyQt5.QtGui import QBrush, QColor
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QMessageBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from ui.utility import ThreadWorker, worker_thread
from ui.widgets import Button, Card, fluent_icon, hint, LineEdit, ProgressBar, use_si_scrollbars

# Reading reviews only touches Mongo; classifying also needs the primary and
# arbiter models. The secondary and confirmation models are optional: when
# unset, the engine skips that step.
READ_ENV = ("MONGO_URI",)
# A/B runs don't use the arbiter: a disagreement there goes to manual triage.
AB_TEST_ENV = READ_ENV + (
    "CHEAPER_INFERENCE_API_KEY",
    "CHEAPER_INFERENCE_API_URL",
    "CHEAPER_INFERENCE_PRIMARY_MODEL_NAME",
)
REVIEW_ENV = AB_TEST_ENV + ("CHEAPER_INFERENCE_ARBITER_MODEL_NAME",)
# Embedding tickets reads and writes Mongo and calls Voyage AI.
EMBED_ENV = READ_ENV + ("EMBEDDING_API_KEY",)
# Optional models, with what is lost when each is unset.
OPTIONAL_MODEL_ENV = {
    "CHEAPER_INFERENCE_SECONDARY_MODEL_NAME": "no fallback if the primary model call fails",
    "CHEAPER_INFERENCE_CONFIRMATION_MODEL_NAME": "manual-triage and low-confidence results won't be re-checked",
}
REVIEW_ADDED_BUTTON = "Review Added to Zendesk"
# SiliconUI's Fluent icon for copying (two overlapping squares).
COPY_ICON = "ic_fluent_copy_regular"
# The training-data export's save-dialog filters and the format each picks.
TRAINING_FORMATS = {
    "Fine-tuning JSONL (*.jsonl)": "jsonl",
    "Few-shot Markdown (*.md)": "markdown",
}
# Table data role on a row's first cell: its review's ID, to keep it selected.
REVIEW_ID_ROLE = QtCore.Qt.UserRole + 1
COLUMNS = [
    "Ticket",
    "Subject",
    "Classification",
    "Confidence",
    "Reviewed (UTC)",
    "Model",
    "Decision",
    "Added to Zendesk",
]
SUBJECT_MAX_WIDTH = 280
# The most reviews the table loads: past this, its widgets use too much memory.
REVIEW_ROW_LIMIT = 5000
# The Decision column's text for each manual outcome; "pending" shows blank.
DECISION_LABELS = {"good": "Correct", "bad": "Misclassified"}
CONNECTION_HINT = (
    "Could not reach MongoDB. Check that you are on the VPN, that its exit IP "
    "is on the Atlas IP access list, and that the cluster is not paused."
)


def missing_env(names=REVIEW_ENV) -> List[str]:
    """Returns the names of required environment variables that are not set.

    A variable counts as missing if it is unset, empty or only whitespace.

    Args:
        names: Environment variable names to check. Defaults to REVIEW_ENV,
            everything a classification run needs.

    Returns:
        The missing names, in the order given. An empty list means everything
        is configured.
    """
    return [name for name in names if not (os.getenv(name) or "").strip()]


def optional_models_note() -> str:
    """Describes which optional models are unset and what that skips.

    Returns:
        One line per unset model, or "" when both are set.
    """
    return "\n".join(
        f"{name} is not set: {effect}."
        for name, effect in OPTIONAL_MODEL_ENV.items()
        if not (os.getenv(name) or "").strip()
    )


def count_awaiting(tickets: Collection) -> int:
    """Counts tickets that are awaiting review.

    Args:
        tickets: The MongoDB tickets collection.

    Returns:
        The number of reviewable documents (not Solved or Closed) where
        `awaiting_review` is True: the tickets the queue lists.
    """
    return tickets.count_documents({"awaiting_review": True, **persistence.REVIEWABLE})


def recent_reviews(tickets: Collection, limit: int = REVIEW_ROW_LIMIT) -> List[dict]:
    """Returns the newest `limit` reviews across all tickets, newest first.

    Each row is {"zendesk_id", "subject", "outcomes", REVIEW_ADDED_KEY,
    ADDED_REVIEWS_KEY, "review"}, where review is one agent_reviews entry
    exactly as the engine stored it.
    """
    pipeline = [
        {"$match": {"agent_reviews.0": {"$exists": True}}},
        {"$unwind": "$agent_reviews"},
        {"$sort": {"agent_reviews.reviewed_at": -1}},
        {"$limit": limit},
        {
            "$project": {
                "_id": 0,
                "zendesk_id": 1,
                "subject": 1,
                "outcomes": 1,
                REVIEW_ADDED_KEY: 1,
                ADDED_REVIEWS_KEY: 1,
                "review": "$agent_reviews",
            }
        },
    ]
    # A large sort can pass Atlas's in-memory cap, so let it spill to disk.
    return list(tickets.aggregate(pipeline, allowDiskUse=True))


def format_review(row: dict) -> str:
    """Plain-text detail for one review. Plain text, not HTML, because the
    reasoning and draft message can quote customer-written content."""
    review = row["review"]
    reviewed_at = review.get("reviewed_at")
    lines = [
        f"Ticket {row.get('zendesk_id')}: {row.get('subject') or '(no subject)'}",
        f"Classification: {review.get('classification')} (confidence {review.get('confidence')})",
        (
            f"Reviewed: {reviewed_at:%Y-%m-%d %H:%M} UTC · Model: {review.get('model_version')}"
            if reviewed_at
            else f"Model: {review.get('model_version')}"
        ),
        "",
        "Signals matched:",
        *[
            f"  - {describe_signal(signal)}"
            for signal in review.get("signals_matched") or ["(none)"]
        ],
        "",
        "Reasoning:",
        review.get("reasoning") or "(none)",
        "",
        "Internal note:",
        review.get("internal_note") or "(none)",
    ]
    if review.get("draft_message"):
        lines += ["", "Draft message (unsent):", review["draft_message"]]
    if review.get("confirmation_of"):
        first = review["confirmation_of"]
        lines += [
            "",
            f"Confirmed: replaced {first.get('classification')} (confidence {first.get('confidence')}) "
            f"from {first.get('model_version')}",
        ]
    if review.get("arbitration"):
        ruling = review["arbitration"]
        lines += [
            "",
            f"Arbitration ({ruling.get('model') or 'no arbiter'}): preferred "
            f"{ruling.get('preferred') or '-'} on the reasoning, ruled "
            f"{ruling.get('ruling') or '-'} on the ticket"
            + (f" ({ruling['error']})" if ruling.get("error") else ""),
        ]
    if review.get("leadership_recipient"):
        lines += ["", f"Leadership recipient: {review['leadership_recipient']}"]
    if review.get("guard_overrides"):
        lines += [
            "",
            "Overridden by code checks:",
            *[f"  - {o}" for o in review["guard_overrides"]],
        ]
    if review.get("engine_error"):
        lines += [
            "",
            f"Engine error (routed to manual triage after repeated failures): {review['engine_error']}",
        ]
    lines += ["", f"Review added to Zendesk: {'Yes' if row_added(row) else 'No'}"]
    if outcome := latest_outcome(row):
        lines += [
            "",
            f"Outcome: {outcome.get('outcome')} ({outcome.get('detection_method')})"
            + (f": {outcome['detail']}" if outcome.get("detail") else ""),
        ]
    if audit := review.get("audit") or {}:
        usage = audit.get("usage") or {}
        lines += [
            "",
            f"Audit: {usage.get('input_tokens', 0)} in / {usage.get('output_tokens', 0)} out tokens "
            f"({usage.get('cache_read_input_tokens', 0)} cache read), {audit.get('latency_ms')} ms, "
            f"request {audit.get('request_id') or '-'}, prompt {audit.get('prompt_version')}",
        ]
    return "\n".join(lines)


def row_added(row: dict) -> bool:
    """Whether this row's review was added to Zendesk."""
    return review_added(row, row["review"].get("review_id"))


# What a review row carries from its ticket, besides the review itself.
TICKET_STATE_FIELDS = ("outcomes", REVIEW_ADDED_KEY, ADDED_REVIEWS_KEY)


def latest_outcome(row: dict) -> Optional[dict]:
    """The newest outcome recorded for this row's review, or None."""
    review_id = row["review"].get("review_id")
    matches = [o for o in row.get("outcomes") or [] if o.get("review_id") == review_id]
    return matches[-1] if matches else None


def decision_label(row: dict) -> str:
    """"Correct" or "Misclassified" from the review's latest outcome, else ""."""
    return DECISION_LABELS.get((latest_outcome(row) or {}).get("outcome"), "")


def fully_handled(row: dict) -> bool:
    """True once a review has a decision and is added to Zendesk; the table
    greys these rows out."""
    return bool(decision_label(row)) and row_added(row)


def awaiting_decision(row: dict) -> bool:
    """True while a review is neither marked correct or misclassified nor
    added to Zendesk."""
    return not decision_label(row) and not row_added(row)


def step_awaiting(rows: List[dict], current: Optional[int], step: int) -> Optional[int]:
    """The index of the nearest row awaiting a decision, or None.

    Rows are newest review first, as recent_reviews returns them, so step 1
    moves to the next older review and step -1 to the next newer one. With
    nothing selected, step 1 starts at the newest review.
    """
    if current is None:
        if step < 0:
            return None
        current = -1
    index = current + step
    while 0 <= index < len(rows):
        if awaiting_decision(rows[index]):
            return index
        index += step
    return None


def ask_batch_size(parent: QWidget, awaiting: int, note: str) -> Optional[int]:
    """Asks how many of the awaiting tickets to review; None if cancelled.

    A module function so tests can answer it without a dialog.
    """
    text = (
        f"{awaiting} ticket(s) are awaiting review. How many should this batch "
        "review, oldest first?\n\nEach ticket is a paid model call. Results are "
        "saved to MongoDB only. You can stop the batch after any ticket."
        + (f"\n\nWarning:\n{note}" if note else "")
    )
    count, ok = QInputDialog.getInt(parent, "Run Batch", text, awaiting, 1, awaiting, 1)
    return count if ok else None


def format_duration(seconds: float) -> str:
    """Seconds as "45 s", "3 min 20 s" or "1 h 5 min"."""
    seconds = max(0, int(round(seconds)))
    if seconds < 60:
        return f"{seconds} s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes} min {seconds} s" if seconds else f"{minutes} min"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes} min" if minutes else f"{hours} h"


def batch_status(progress: dict, elapsed: float) -> str:
    """The progress line: tickets done, outcomes, speed and time left.

    The estimate is the average time per finished ticket so far times the
    tickets left, so it follows the model's current speed.
    """
    done, total, counts = progress["done"], progress["total"], progress["summary"]
    parts = [f"{done} of {total} done"]
    outcomes = [
        f"{counts[key]} {label}"
        for key, label in (
            ("classified", "classified"),
            ("manual_fallback", "to manual triage"),
            ("skipped", "skipped"),
            ("errored", "errored"),
        )
        if counts.get(key)
    ]
    if outcomes:
        parts.append(", ".join(outcomes))
    if done:
        per_ticket = elapsed / done
        parts.append(f"{per_ticket:.1f} s per ticket")
        if done < total:
            parts.append(f"about {format_duration(per_ticket * (total - done))} left")
    return " · ".join(parts)


class BatchRelay(QObject):
    """Carries batch or embedding progress from a worker thread to the tab
    (queued, as it lives on the GUI thread)."""

    progress = pyqtSignal(object)


class TriageTab(QWidget):
    """The Decision Engine tab.

    Every Mongo or model call runs on a QThread, so a slow or unreachable
    cluster never freezes the window. One job runs at a time. A running
    QThread must not be destroyed, so the main window calls request_shutdown
    before closing and waits for `idle` if a job is still running.

    Signals:
        idle: Emitted when a job ends after request_shutdown, meaning the tab
            can now be destroyed safely.

    Args:
        collection_factory: Returns the tickets collection. Defaults to the
            engine's own connection so both share one client.
        review_batch: Reviews the given Zendesk IDs: called as
            review_batch(tickets, zendesk_ids). Defaults to run_review_batch,
            which also stops between tickets once shutdown is requested.
        review_one: Reviews one ticket by Zendesk ID; returns None if it is
            missing or not awaiting review.
    """

    idle = pyqtSignal()

    def __init__(
        self,
        collection_factory: Callable[
            [], Collection
        ] = persistence.get_tickets_collection,
        review_batch: Optional[Callable[[Collection, List[str]], Any]] = None,
        review_one: Callable[
            [str, Collection], Optional[dict]
        ] = persistence.process_one_ticket,
        parent: Optional[QWidget] = None,
    ) -> None:
        """Stores the injected callables, initializes job and table state, and builds the UI.

        The tickets collection is not touched here; data loads on first show
        (see showEvent).
        """
        super().__init__(parent)
        self.collection_factory = collection_factory
        # Set once shutdown is requested; the default batch checks it between
        # tickets. A Mongo or model call already in flight can't be
        # interrupted safely, so it always finishes.
        self._cancel = threading.Event()
        self._closing = False
        # Called on the worker thread as review_batch(tickets, ids, on_progress=...).
        self.review_batch = review_batch or (
            lambda tickets, zendesk_ids, on_progress=None: persistence.run_review_batch(
                tickets,
                zendesk_ids,
                should_stop=self._cancel.is_set,
                on_progress=on_progress,
            )
        )
        # The running batch: total, start time and subjects, or None.
        self._batch: Optional[dict] = None
        self._batch_relay = BatchRelay(self)
        self._batch_relay.progress.connect(self._on_batch_progress)
        self.review_one = review_one
        # Embedding runs on its own thread so reviewing continues meanwhile.
        # Stop is checked between batches; each batch is saved before the next.
        self._embed_cancel = threading.Event()
        self._embed_thread: Optional[QThread] = None
        self._embed_worker: Optional[ThreadWorker] = None
        self._embed_relay = BatchRelay(self)
        self._embed_relay.progress.connect(self._on_embed_progress)
        self._thread: Optional[QThread] = None
        self._worker: Optional[ThreadWorker] = None
        self._on_done: Optional[Callable[[Any], None]] = None
        self._on_error: Optional[Callable[[str], None]] = None
        self.weekly_review_dialog = None
        self.ab_test_dialog = None
        self._rows: List[dict] = []
        self._loaded = False
        self._build_ui()

    # --- layout ---------------------------------------------------------------

    def _build_ui(self) -> None:
        """Creates the widgets and layout, starting in the idle state."""

        intro = hint(
            "Classifies tickets awaiting review and records each result on the ticket "
            "in MongoDB. Nothing is written to Zendesk. Each ticket reviewed is a paid model call."
        )

        self.status_label = hint("")
        self.refresh_button = Button("Refresh")
        self.refresh_button.clicked.connect(self.refresh)
        status_row = QHBoxLayout()
        status_row.addWidget(self.status_label, stretch=1)
        status_row.addWidget(self.refresh_button)

        self.batch_button = Button("Run Batch…", primary=True)
        self.batch_button.setToolTip(
            "Review the next N tickets awaiting review, oldest first. "
            "Each ticket is a paid model call."
        )
        self.batch_button.clicked.connect(self.start_batch)

        self.ticket_id_input = LineEdit("Ticket")
        self.ticket_id_input.setPlaceholderText("Zendesk ticket ID")
        self.ticket_id_input.setClearButtonEnabled(True)
        self.ticket_id_input.returnPressed.connect(self.start_review_one)
        self.review_one_button = Button("Classify One Ticket")
        self.review_one_button.clicked.connect(self.start_review_one)
        one_row = QHBoxLayout()
        one_row.addWidget(self.ticket_id_input, stretch=1)
        one_row.addWidget(self.review_one_button)
        one_row.addWidget(self.batch_button)

        self.embed_button = Button("Embed Tickets (Voyage AI)…")
        self.embed_button.setToolTip(
            "Stores a Voyage embedding on each new or changed ticket and creates the vector index."
        )
        self.embed_button.clicked.connect(self.start_embed_tickets)
        one_row.addWidget(self.embed_button)
        # Shown while embedding runs in the background.
        self.embed_label = hint("")
        self.embed_label.setVisible(False)
        self.embed_stop_button = Button("Stop Embedding")
        self.embed_stop_button.setToolTip(
            "Finishes the batch being embedded, saves it, and leaves the rest for the next run."
        )
        self.embed_stop_button.clicked.connect(self.stop_embedding)
        self.embed_stop_button.setVisible(False)

        # Range 0..0 makes the bar an indeterminate "busy" indicator:
        # run_review_batch reports no progress.
        self.busy_bar = ProgressBar()
        self.busy_bar.setRange(0, 0)
        self.busy_bar.setVisible(False)
        # Shown while a batch runs: its progress, and a way to stop it safely.
        self.batch_label = hint("")
        self.batch_label.setVisible(False)
        self.stop_button = Button("Stop After This Ticket")
        self.stop_button.setToolTip(
            "Finishes the ticket being reviewed, saves it, and leaves the rest in the queue."
        )
        self.stop_button.clicked.connect(self.stop_batch)
        self.stop_button.setVisible(False)

        self.table = use_si_scrollbars(QTableWidget(0, len(COLUMNS)))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSortingEnabled(True)
        self.table.itemSelectionChanged.connect(self.show_selected_review)

        self.detail = use_si_scrollbars(QTextEdit())
        self.detail.setReadOnly(True)
        self.detail.setPlaceholderText(
            "Select a review to see its reasoning, note and draft message."
        )

        self.previous_button = Button("Previous Ticket")
        self.previous_button.setToolTip(
            "The next newer review not yet marked correct, misclassified or added to Zendesk."
        )
        self.previous_button.clicked.connect(lambda: self.select_awaiting(-1))
        self.next_button = Button("Next Ticket")
        self.next_button.setToolTip(
            "The next older review not yet marked correct, misclassified or added to Zendesk."
        )
        self.next_button.clicked.connect(lambda: self.select_awaiting(1))
        self.mark_good_button = Button("Mark Correct")
        self.mark_good_button.clicked.connect(self.mark_selected_correct)
        self.mark_bad_button = Button("Mark Misclassified…")
        self.mark_bad_button.clicked.connect(self.mark_selected_misclassified)
        self.zendesk_button = Button(zendesk_links.BUTTON_TEXT)
        self.zendesk_button.setToolTip(zendesk_links.BUTTON_TIP)
        self.zendesk_button.clicked.connect(self.view_selected_on_zendesk)
        self.review_added_button = Button(REVIEW_ADDED_BUTTON)
        self.review_added_button.setToolTip(
            "Mark the ticket once you have copied its review into Zendesk by hand."
        )
        self.review_added_button.clicked.connect(self.mark_selected_review_added)
        self.copy_button = Button()
        self.copy_button.setSvgIcon(fluent_icon(COPY_ICON))
        self.copy_button.setToolTip("Copy the whole review to the clipboard.")
        self.copy_button.clicked.connect(self.copy_selected_review)
        self.accuracy_button = Button("Accuracy Report")
        self.accuracy_button.clicked.connect(self.show_accuracy_report)
        self.training_button = Button("Export Training Data…")
        self.training_button.setToolTip(
            "Save reviews marked correct and A/B verdicts as examples: .jsonl for "
            "fine-tuning, or .md for few-shot prompting (with corrections)."
        )
        self.training_button.clicked.connect(self.export_training_data)
        self.weekly_review_button = Button("Weekly Review…")
        self.weekly_review_button.setToolTip(
            "Review the week's misclassifications and write the lessons learned from them."
        )
        self.weekly_review_button.clicked.connect(self.open_weekly_review)
        self.ab_test_button = Button("A/B Test…")
        self.ab_test_button.setToolTip(
            "Classify a sample of tickets with two engine configurations and "
            "compare the answers blind. Nothing is saved to the tickets."
        )
        self.ab_test_button.clicked.connect(self.open_ab_test)
        # The detail pane sits beside the list, so its buttons wrap onto rows.
        nav_row = QHBoxLayout()
        nav_row.addWidget(self.previous_button)
        nav_row.addWidget(self.next_button)
        nav_row.addStretch()
        nav_row.addWidget(self.zendesk_button)
        nav_row.addWidget(self.copy_button)
        feedback_row = QHBoxLayout()
        feedback_row.addWidget(self.mark_good_button)
        feedback_row.addWidget(self.mark_bad_button)
        feedback_row.addWidget(self.review_added_button)
        feedback_row.addStretch()
        detail_pane = QWidget()
        detail_layout = QVBoxLayout(detail_pane)
        detail_layout.setContentsMargins(0, 0, 0, 0)
        detail_layout.addLayout(nav_row)
        detail_layout.addWidget(self.detail)
        detail_layout.addLayout(feedback_row)

        splitter = QSplitter(QtCore.Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self.table)
        splitter.addWidget(detail_pane)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        # Start at 3:2; the sizes are ratios until the splitter is laid out.
        splitter.setSizes([3000, 2000])

        tools_row = QHBoxLayout()
        tools_row.addStretch()
        tools_row.addWidget(self.ab_test_button)
        tools_row.addWidget(self.weekly_review_button)
        tools_row.addWidget(self.training_button)
        tools_row.addWidget(self.accuracy_button)

        review_card = Card("Review Tickets")
        review_card.body.addWidget(intro)
        review_card.body.addLayout(status_row)
        review_card.body.addLayout(one_row)
        review_card.body.addWidget(self.busy_bar)
        batch_row = QHBoxLayout()
        batch_row.addWidget(self.batch_label, stretch=1)
        batch_row.addWidget(self.stop_button)
        review_card.body.addLayout(batch_row)
        embed_row = QHBoxLayout()
        embed_row.addWidget(self.embed_label, stretch=1)
        embed_row.addWidget(self.embed_stop_button)
        review_card.body.addLayout(embed_row)

        reviews_card = Card("Reviews")
        reviews_card.body.addWidget(splitter)
        reviews_card.body.addLayout(tools_row)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 12, 0, 0)
        layout.setSpacing(12)
        layout.addWidget(review_card)
        layout.addWidget(reviews_card, stretch=1)

        self._set_busy(False)

    def showEvent(self, event) -> None:
        """Loads reviews the first time the tab is shown.

        Deferring the load until first view means opening the app never waits
        on MongoDB. Later shows do not reload; use Refresh.

        Args:
            event: The Qt show event, passed through to the base class.
        """
        # Load on first view rather than at startup, so opening the app never
        # waits on Mongo.
        super().showEvent(event)
        if not self._loaded:
            self._loaded = True
            self.refresh()

    # --- background jobs ------------------------------------------------------

    def _configured(self, names, notify: bool) -> bool:
        """Checks that the variables one operation needs are set.

        Args:
            names: The variables this operation needs (READ_ENV, AB_TEST_ENV
                or REVIEW_ENV).
            notify: True to also log and show a warning dialog; False only
                updates the status label (used for background refreshes).

        Returns:
            True if every variable in `names` is set.
        """
        missing = missing_env(names)
        if not missing:
            return True
        message = f"Not configured: set {', '.join(missing)} in Settings (Preferences on macOS)."
        self.status_label.setText(message)
        if notify:
            logger.error(message)
            QMessageBox.warning(self, "Decision Engine Not Configured", message)
        return False

    def _warn_optional_models(self) -> None:
        """Logs a warning for each optional model that is unset.

        Reviews still run without them; the engine skips that step. The same
        note is shown in the Review All confirmation and the Classify One
        result dialog.
        """
        note = optional_models_note()
        if note:
            for line in note.splitlines():
                logger.warning(line)

    def is_busy(self) -> bool:
        """Returns True while a background job or embedding is running.

        Callers such as the main window can use this to avoid closing the app
        mid-job.
        """
        return self._job_running() or self.is_embedding()

    def _job_running(self) -> bool:
        """True while a review, refresh or other one-at-a-time job runs."""
        return self._thread is not None

    def is_embedding(self) -> bool:
        """True while ticket embedding runs in the background."""
        return self._embed_thread is not None

    def request_shutdown(self) -> bool:
        """Prepares the tab for the window closing.

        If no job is running, returns True: the tab can be destroyed now.
        Otherwise cancels the rest of a review batch (the ticket in progress
        still finishes), keeps the controls disabled, and returns False; `idle`
        is emitted once the job has ended and its thread has been joined. The
        job's own completion callback and dialogs are skipped.
        """
        self._closing = True
        self._cancel.set()
        self._embed_cancel.set()
        if not self.is_busy():
            return True
        message = "Closing after the current job finishes…"
        self.status_label.setText(message)
        self.busy_bar.setFormat(message)
        logger.warning(f"Decision Engine: {message}")
        return False

    def wait_for_job(self) -> None:
        """Blocks until the running job's thread has finished, then joins it.

        Last resort for when the application is quitting anyway (e.g. on
        aboutToQuit): it freezes the window for the rest of the job, but a
        running QThread is never destroyed.
        """
        self._cancel.set()
        self._embed_cancel.set()
        if self._embed_thread is not None:
            self._embed_thread.quit()
            self._embed_thread.wait()
            self._embed_thread = None
            self._embed_worker = None
        if self._thread is None:
            return
        self._thread.quit()  # the thread's loop exits once the worker returns
        self._thread.wait()
        self._thread = None
        self._worker = None
        self._on_done = self._on_error = None

    def _set_busy(self, busy: bool, label: str = "") -> None:
        """Switches the tab between busy and idle states.

        Enables or disables the buttons and ticket ID field, and shows or hides
        the busy bar.

        Args:
            busy: True to disable controls and show the busy bar.
            label: Text for the busy bar and, when busy, the status label.
        """

        for button in (
            self.refresh_button,
            self.batch_button,
            self.review_one_button,
            self.accuracy_button,
            self.weekly_review_button,
            self.ab_test_button,
            self.training_button,
        ):
            button.setEnabled(not busy)
        self.embed_button.setEnabled(not busy and not self.is_embedding())
        has_selection = self._selected_row() is not None
        for button in (self.mark_good_button, self.mark_bad_button):
            button.setEnabled(not busy and has_selection)
        # Opening a link or copying doesn't touch Mongo, so they work during a job too.
        self.zendesk_button.setEnabled(has_selection)
        self.copy_button.setEnabled(has_selection)
        self._enable_review_added(busy)
        self._enable_navigation()
        self.ticket_id_input.setEnabled(not busy)
        self.busy_bar.setVisible(busy)
        self.busy_bar.setFormat(label)
        if not busy:
            # A batch's progress UI goes with it.
            self._batch = None
            self.busy_bar.setRange(0, 0)
            self.batch_label.setVisible(False)
            self.stop_button.setVisible(False)
        if busy and label:
            self.status_label.setText(label)

    def _run(
        self,
        task: Callable[[], Any],
        description: str,
        on_done: Callable[[Any], None],
        on_error: Optional[Callable[[str], None]] = None,
    ) -> bool:
        """Runs `task` on a worker thread, then `on_done(result)` (or `on_error(message)`) here.

        Returns:
            False, without running anything, while another job runs or the tab is closing.
        """
        if self._closing or self._thread is not None:
            return False

        def with_connection_hint():
            try:
                return task()
            except ServerSelectionTimeoutError as e:
                raise RuntimeError(CONNECTION_HINT) from e

        self._set_busy(True, f"{description}…")
        self._on_done = on_done
        self._on_error = on_error
        self._thread = worker_thread(self)
        self._worker = ThreadWorker(with_connection_hint, description)
        self._worker.moveToThread(self._thread)
        self._worker.finished.connect(self._job_finished)
        self._worker.error.connect(self._job_failed)
        self._thread.started.connect(self._worker.run)
        self._thread.start()
        return True

    def _stop_thread(self) -> None:
        """Shuts down the current worker thread and returns to the idle state.

        Quits and waits for the thread, clears the thread and worker
        references, and re-enables the controls (unless the tab is closing).
        """
        self._thread.quit()
        self._thread.wait()
        self._thread = None
        self._worker = None
        if not self._closing:
            self._set_busy(False)

    def _job_finished(self, result: Any) -> None:
        """Handles a successful job.

        Stops the thread first, then invokes the stored `on_done` callback with
        the result. The order matters: the callback may start the next job.
        Completion is reported by each callback, not here, so refreshes and
        intermediate steps (like the pre-confirmation count) stay quiet.

        Args:
            result: The value returned by the task.
        """
        self._stop_thread()
        on_done, self._on_done, self._on_error = self._on_done, None, None
        if self._closing:
            # Don't start follow-up jobs or show dialogs while closing.
            self._emit_idle_when_done()
            return
        on_done(result)

    def _job_failed(self, message: str) -> None:
        """Handles a failed job.

        Stops the thread, discards the pending callback, updates the status
        label, and shows an error dialog.

        Args:
            message: The error text from the worker, shown to the user.
        """
        self._stop_thread()
        on_error, self._on_done, self._on_error = self._on_error, None, None
        if self._closing:
            logger.error(f"Decision Engine job failed while closing: {message}")
            self._emit_idle_when_done()
            return
        self.status_label.setText("Last action failed; see the log.")
        QMessageBox.critical(self, "Decision Engine Error", message)
        if on_error is not None:
            on_error(message)

    def _emit_idle_when_done(self) -> None:
        """Emits `idle` once neither the job nor embedding is running."""
        if not self.is_busy():
            self.idle.emit()

    # --- actions ---------------------------------------------------------------

    def refresh(self) -> None:
        """Reloads the awaiting count and recent reviews from MongoDB.

        Needs only MONGO_URI, so existing reviews can be inspected without the
        model settings. Does nothing, without showing a dialog, if it is unset. The load runs on a background thread and updates the status
        label and table via `_show_loaded`.
        """
        if not self._configured(READ_ENV, notify=False):
            return

        def load():
            tickets = self.collection_factory()
            return count_awaiting(tickets), recent_reviews(tickets)

        self._run(load, "Loading reviews from MongoDB", self._show_loaded)

    def _show_loaded(self, loaded) -> None:
        """Displays the result of a refresh.

        Args:
            loaded: A (awaiting_count, review_rows) tuple, as produced by the
                refresh task.
        """
        awaiting, rows = loaded
        status = f"{awaiting} ticket(s) awaiting review."
        if len(rows) >= REVIEW_ROW_LIMIT:
            status += f" Showing the newest {REVIEW_ROW_LIMIT:,} reviews."
        self.status_label.setText(status)
        self._rows = rows
        self._fill_table()

    def start_batch(self) -> None:
        """Begins a batch: lists the awaiting tickets, then asks how many to review.

        Snapshots the awaiting tickets off the GUI thread before asking, so the
        batch reviews exactly the tickets the user approved, not ones inserted
        or reopened while the dialog was open. Shows a warning and stops if
        the environment is not configured.
        """
        if not self._configured(REVIEW_ENV, notify=True):
            return
        self._run(
            lambda: persistence.awaiting_tickets(self.collection_factory()),
            "Finding tickets awaiting review",
            self._confirm_batch,
        )

    def _confirm_batch(self, awaiting: List[dict]) -> None:
        """Asks how many tickets to review (1 to all), then reviews the oldest that many.

        Args:
            awaiting: The tickets awaiting review when the list was taken,
                oldest first, as {"zendesk_id", "subject"}.
        """
        self.status_label.setText(f"{len(awaiting)} ticket(s) awaiting review.")
        if not awaiting:
            QMessageBox.information(
                self, "Nothing to Review", "No tickets are awaiting review."
            )
            return
        count = ask_batch_size(self, len(awaiting), optional_models_note())
        if not count:
            return
        chosen = awaiting[:count]
        zendesk_ids = [ticket["zendesk_id"] for ticket in chosen]
        self._warn_optional_models()
        self._cancel.clear()  # a Stop pressed during an earlier batch

        def review_chosen():
            tickets = self.collection_factory()

            def with_ticket_state(progress: dict) -> None:
                # Read here, on the batch's thread: a re-reviewed ticket's new
                # row must show its outcomes and Zendesk status like a refresh.
                if progress.get("review") is not None:
                    ticket = tickets.find_one(
                        {"zendesk_id": progress["zendesk_id"]},
                        {field: 1 for field in TICKET_STATE_FIELDS},
                    )
                    progress = {**progress, "ticket": ticket or {}}
                self._batch_relay.progress.emit(progress)

            return self.review_batch(
                tickets, list(zendesk_ids), on_progress=with_ticket_state
            )

        started = self._run(
            review_chosen,
            f"Reviewing {count} ticket(s)",
            self._batch_done,
        )
        if not started:
            return
        self._batch = {
            "total": count,
            "started": time.monotonic(),
            "subjects": {t["zendesk_id"]: t.get("subject") for t in chosen},
        }
        self.busy_bar.setRange(0, count)
        self.busy_bar.setValue(0)
        self.batch_label.setText(f"0 of {count} done · starting…")
        self.batch_label.setVisible(True)
        self.stop_button.setEnabled(True)
        self.stop_button.setVisible(True)

    def stop_batch(self) -> None:
        """Stops the batch after the ticket in progress, which is still saved."""
        if self._batch is None:
            return
        self._cancel.set()
        self.stop_button.setEnabled(False)
        self.batch_label.setText(
            self.batch_label.text().split(" · about")[0]
            + " · stopping after this ticket…"
        )
        logger.info("Decision Engine: batch will stop after the current ticket.")

    def _on_batch_progress(self, progress: dict) -> None:
        """After each ticket: moves the bar, updates the counts and time left,
        and adds a stored review to the Reviews table."""
        if self._batch is None:  # a late signal after the batch ended
            return
        self.busy_bar.setValue(progress["done"])
        status = batch_status(progress, time.monotonic() - self._batch["started"])
        if self._cancel.is_set() and not self._closing:
            status = status.split(" · about")[0] + " · stopping after this ticket…"
        self.batch_label.setText(status)
        review = progress.get("review")
        if review is not None:
            zendesk_id = progress["zendesk_id"]
            ticket = progress.get("ticket") or {}
            row = {
                "zendesk_id": zendesk_id,
                "subject": self._batch["subjects"].get(zendesk_id),
                **{field: ticket[field] for field in TICKET_STATE_FIELDS if field in ticket},
                "review": review,
            }
            self._rows = [row, *self._rows][:REVIEW_ROW_LIMIT]
            self._fill_table()

    def start_embed_tickets(self) -> None:
        """Checks the vector index and lists stale tickets, then asks before sending them to Voyage.

        The index is checked first so a mismatch stops the run before any
        paid call; only the tickets in the approved list are sent.
        """
        if self.is_embedding() or not self._configured(EMBED_ENV, notify=True):
            return
        try:
            cfg = embeddings.config()
        except embeddings.EmbeddingNotConfigured as e:
            self.status_label.setText(str(e))
            QMessageBox.warning(self, "Embeddings Not Configured", str(e))
            return

        def prepare():
            tickets = self.collection_factory()
            status = embeddings.index_status(tickets, cfg)  # raises VectorIndexMismatch
            return embeddings.plan_backfill(tickets, cfg), status

        self._run(
            prepare,
            "Finding tickets that need embeddings",
            lambda prepared: self._confirm_embed(cfg, *prepared),
        )

    def _confirm_embed(self, cfg, plan, status: str) -> None:
        """Asks before the paid calls, then embeds exactly the approved tickets."""
        if plan.embed:
            answer = QMessageBox.question(
                self,
                "Embed Tickets",
                embeddings.describe_plan(plan, cfg)
                + " Tickets with a current embedding are skipped.\n\nContinue?",
                QMessageBox.Yes | QMessageBox.Cancel,
                QMessageBox.Cancel,
            )
            if answer != QMessageBox.Yes:
                return

        def embed_and_index():
            tickets = self.collection_factory()
            summary = embeddings.run_backfill(
                tickets,
                cfg,
                plan,
                should_stop=self._embed_cancel.is_set,
                on_progress=lambda handled, total: self._embed_relay.progress.emit(
                    (handled, total)
                ),
            )
            # A VectorIndexMismatch here (the index changed in Atlas during the
            # run) propagates and is shown as an error, not a warning.
            try:
                embeddings.finish_index(tickets, cfg, summary, status)
            except search.SearchUnavailable as e:
                summary["index"] = f"skipped: {e}"
            return summary

        self._run_embed(embed_and_index, len(plan.embed))

    def _run_embed(self, task: Callable[[], dict], total: int) -> None:
        """Runs the embedding `task` on its own thread, beside any other job."""
        if self._closing or self.is_embedding():
            return

        def with_connection_hint():
            try:
                return task()
            except ServerSelectionTimeoutError as e:
                raise RuntimeError(CONNECTION_HINT) from e

        self._embed_cancel.clear()  # a Stop pressed during an earlier run
        self._embed_thread = worker_thread(self)
        self._embed_worker = ThreadWorker(with_connection_hint, "Embedding tickets")
        self._embed_worker.moveToThread(self._embed_thread)
        self._embed_worker.finished.connect(self._embed_finished)
        self._embed_worker.error.connect(self._embed_failed)
        self._embed_thread.started.connect(self._embed_worker.run)
        self._show_embedding(f"Embedding 0/{total} ticket(s) in the background…")
        self._embed_thread.start()

    def _show_embedding(self, text: str) -> None:
        """Shows embedding progress and Stop, or hides both when text is empty."""
        running = self.is_embedding()
        self.embed_label.setText(text)
        self.embed_label.setVisible(bool(text))
        self.embed_stop_button.setVisible(running)
        self.embed_stop_button.setEnabled(running and not self._embed_cancel.is_set())
        self.embed_button.setEnabled(not running and not self._job_running())

    def _on_embed_progress(self, progress) -> None:
        if not self.is_embedding() or self._embed_cancel.is_set():
            return
        handled, total = progress
        self._show_embedding(f"Embedding {handled}/{total} ticket(s) in the background…")

    def stop_embedding(self) -> None:
        """Stops embedding after the batch in flight is saved."""
        if not self.is_embedding():
            return
        self._embed_cancel.set()
        self._show_embedding("Stopping embedding after this batch is saved…")
        logger.warning("Embedding: stop requested; finishing the current batch.")

    def _stop_embed_thread(self) -> None:
        self._embed_thread.quit()
        self._embed_thread.wait()
        self._embed_thread = None
        self._embed_worker = None
        self._show_embedding("")

    def _embed_finished(self, summary: dict) -> None:
        self._stop_embed_thread()
        if self._closing:
            self._emit_idle_when_done()
            return
        self._embed_done(summary)

    def _embed_failed(self, message: str) -> None:
        self._stop_embed_thread()
        if self._closing:
            logger.error(f"Embedding failed while closing: {message}")
            self._emit_idle_when_done()
            return
        self.status_label.setText("Embedding failed; see the log.")
        QMessageBox.critical(self, "Decision Engine Error", message)

    def _embed_done(self, summary: dict) -> None:
        message = embeddings.format_summary(summary)
        logger.success(message.replace("\n", "; "))
        stopped = summary.get("cancelled")
        self.status_label.setText(
            "Embedding stopped; the rest is left for the next run."
            if stopped
            else "Embedding complete."
        )
        QMessageBox.information(
            self, "Embedding Stopped" if stopped else "Embedding Complete", message
        )

    def _batch_done(self, summary) -> None:
        """Logs and announces the batch's end, then refreshes the list.

        Args:
            summary: The batch's counts from run_review_batch, or None.
        """
        stopped_by_user = isinstance(summary, dict) and summary.get("cancelled")
        logger.success(
            "Review batch stopped." if stopped_by_user else "Review batch finished."
        )
        message = (
            "Batch stopped after the current ticket; the rest are still awaiting review."
            if stopped_by_user
            else "Review batch finished. Results are listed below."
        )
        if isinstance(summary, dict):
            message += "\n\n" + persistence.format_batch_summary(summary)
        if isinstance(summary, dict) and summary.get("stopped"):
            QMessageBox.warning(self, "Review Stopped Early", message)
        else:
            title = "Batch Stopped" if stopped_by_user else "Review Complete"
            QMessageBox.information(self, title, message)
        self.refresh()

    def start_review_one(self) -> None:
        """Classifies the ticket whose Zendesk ID is in the input field.

        Warns and stops if the field is empty or the environment is not
        configured. Triggered by the button or by pressing Enter in the field.
        """

        zendesk_id = self.ticket_id_input.text().strip()
        if not zendesk_id:
            QMessageBox.warning(
                self, "No Ticket ID", "Enter a Zendesk ticket ID to classify."
            )
            return
        if not self._configured(REVIEW_ENV, notify=True):
            return
        self._warn_optional_models()
        self._run(
            lambda: self.review_one(zendesk_id, self.collection_factory()),
            f"Classifying ticket {zendesk_id}",
            lambda result: self._review_one_done(zendesk_id, result),
        )

    def _review_one_done(self, zendesk_id: str, result: Optional[dict]) -> None:
        """Handles the outcome of a single-ticket review.

        If the result is None (ticket missing or not awaiting review), logs a
        warning and tells the user. Otherwise logs and shows the classification,
        clears the input field, and refreshes the list.

        Args:
            zendesk_id: The ID that was submitted.
            result: The engine's review dict, or None.
        """
        if result is None:
            message = (
                f"Ticket {zendesk_id} was not found, is not awaiting review, or was "
                "reassigned while it was classified (it stays awaiting review)."
            )
            logger.warning(message)
            QMessageBox.information(self, "Not Classified", message)
            return
        message = f"Ticket {zendesk_id}: {result['classification']} (confidence {result['confidence']})"
        if result.get("engine_error"):
            message += f"\n\nThe engine failed repeatedly, so the ticket was routed to manual triage: {result['engine_error']}"
        logger.success(message)
        note = optional_models_note()
        QMessageBox.information(
            self,
            "Ticket Classified",
            message + (f"\n\nWarning:\n{note}" if note else ""),
        )
        self.ticket_id_input.clear()
        self.refresh()

    # --- results ---------------------------------------------------------------

    def _fill_table(self) -> None:
        """Populates the table from the loaded review rows.

        Sorting is paused during the fill so rows don't reorder mid-insert.
        Numeric values are stored as numbers so they sort numerically, and each
        row's index into `self._rows` is saved on its first cell for later
        lookup. Afterwards the table is sorted newest-first by review time,
        columns are resized, and the detail pane is cleared.
        """

        # Read from the table, not self._rows, which may already hold new rows.
        selected = self.table.selectedItems()
        keep = (
            self.table.item(selected[0].row(), 0).data(REVIEW_ID_ROLE)
            if selected
            else None
        )
        # Sorting is paused while filling, or rows reorder mid-insert.
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(self._rows))
        faint = QBrush(QColor(theme.TEXT_FAINT))
        for row_index, row in enumerate(self._rows):
            handled = fully_handled(row)
            review = row["review"]
            reviewed_at = review.get("reviewed_at")
            values = [
                row.get("zendesk_id"),
                row.get("subject"),
                review.get("classification"),
                review.get("confidence"),
                f"{reviewed_at:%Y-%m-%d %H:%M}" if reviewed_at else "",
                review.get("model_version"),
                decision_label(row),
                "Yes" if row_added(row) else "No",
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem()
                # DisplayRole with the raw value so numbers sort numerically.
                item.setData(
                    QtCore.Qt.DisplayRole,
                    value if isinstance(value, (int, float)) else str(value or ""),
                )
                if handled:
                    item.setForeground(faint)
                if column == 0:
                    item.setData(QtCore.Qt.UserRole, row_index)
                    item.setData(REVIEW_ID_ROLE, str(review.get("review_id") or ""))
                self.table.setItem(row_index, column, item)
        self.table.setSortingEnabled(True)
        # Re-enabling sorting applies the header's current indicator, which
        # defaults to column 0; reset it to newest review first.
        self.table.sortItems(COLUMNS.index("Reviewed (UTC)"), QtCore.Qt.DescendingOrder)
        self.table.resizeColumnsToContents()
        # A long subject would push the other columns out of the list.
        subject = COLUMNS.index("Subject")
        self.table.setColumnWidth(
            subject, min(self.table.columnWidth(subject), SUBJECT_MAX_WIDTH)
        )
        # Rows added during a batch keep the review being read selected.
        if keep:
            for table_row in range(self.table.rowCount()):
                if self.table.item(table_row, 0).data(REVIEW_ID_ROLE) == keep:
                    self.table.selectRow(table_row)
                    # Reselecting the same row emits no signal; show its new data.
                    self.show_selected_review()
                    return
        self.detail.clear()
        self._enable_navigation()

    def show_selected_review(self) -> None:
        """Shows the selected row's full review in the detail pane.

        Looks the review up through the row index stored on the row's first
        cell, so it stays correct after the user re-sorts the table. Clears the
        pane when nothing is selected.
        """
        row = self._selected_row()
        enabled = row is not None and not self._job_running()
        self.mark_good_button.setEnabled(enabled)
        self.mark_bad_button.setEnabled(enabled)
        self.zendesk_button.setEnabled(row is not None)
        self.copy_button.setEnabled(row is not None)
        self._enable_review_added(self._job_running())
        self._enable_navigation()
        if row is None:
            self.detail.clear()
            return
        text = format_review(row)
        # Unchanged text is left alone, so the pane keeps its scroll position.
        if self.detail.toPlainText() != text:
            self.detail.setPlainText(text)

    def view_selected_on_zendesk(self) -> None:
        """Opens the selected review's ticket in Zendesk."""
        row = self._selected_row()
        if row is not None:
            zendesk_links.open_ticket(row.get("zendesk_id"))

    def copy_selected_review(self) -> None:
        """Copies the selected review, as shown in the detail pane, to the clipboard."""
        row = self._selected_row()
        if row is None:
            return
        QApplication.clipboard().setText(format_review(row))
        self.status_label.setText(f"Review of ticket {row.get('zendesk_id')} copied.")

    def _enable_review_added(self, busy: bool) -> None:
        """Enabled for a selected ticket not yet marked, while no job runs."""
        row = self._selected_row()
        self.review_added_button.setEnabled(
            not busy and row is not None and not row_added(row)
        )

    def mark_selected_review_added(self) -> None:
        """Asks for confirmation, then marks the selected ticket's review as
        added to Zendesk."""
        row = self._selected_row()
        if row is None or not self._configured(READ_ENV, notify=True):
            return
        zendesk_id, review_id = row["zendesk_id"], row["review"].get("review_id")
        answer = QMessageBox.question(
            self,
            REVIEW_ADDED_BUTTON,
            f"Has the review of ticket {zendesk_id} been manually added to Zendesk?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        self._run(
            lambda: mark_review_added(self.collection_factory(), zendesk_id, review_id),
            f"Marking ticket {zendesk_id}",
            lambda found: self._feedback_done(
                f"Ticket {zendesk_id}: review marked as added to Zendesk."
                if found
                else f"Ticket {zendesk_id} is no longer in MongoDB; nothing was marked."
            ),
        )

    def _enable_navigation(self) -> None:
        """Enables Previous and Next only when there is a review to move to.

        Moving only changes the selection, so both work while a job runs.
        """
        current = self._selected_index()
        self.previous_button.setEnabled(
            step_awaiting(self._rows, current, -1) is not None
        )
        self.next_button.setEnabled(step_awaiting(self._rows, current, 1) is not None)

    def select_awaiting(self, step: int) -> None:
        """Selects the nearest review awaiting a decision: step 1 is older,
        -1 newer."""
        target = step_awaiting(self._rows, self._selected_index(), step)
        if target is None:
            return
        # The table may be sorted by any column; find the row holding target.
        for table_row in range(self.table.rowCount()):
            item = self.table.item(table_row, 0)
            if item.data(QtCore.Qt.UserRole) == target:
                self.table.selectRow(table_row)
                self.table.scrollToItem(item)
                return

    def _selected_index(self) -> Optional[int]:
        """The index into `self._rows` of the selected table row, or None."""
        selected = self.table.selectedItems() if hasattr(self, "table") else []
        if not selected:
            return None
        return self.table.item(selected[0].row(), 0).data(QtCore.Qt.UserRole)

    def _selected_row(self) -> Optional[dict]:
        """The loaded row behind the selected table row, or None."""
        index = self._selected_index()
        return None if index is None else self._rows[index]

    # --- feedback ----------------------------------------------------------------

    def mark_selected_correct(self) -> None:
        """Records a manual "good" outcome for the selected review."""
        row = self._selected_row()
        if row is None or not self._configured(READ_ENV, notify=True):
            return
        zendesk_id, review_id = row["zendesk_id"], row["review"]["review_id"]
        self._run(
            lambda: feedback.record_outcome(
                self.collection_factory(), zendesk_id, review_id, "good"
            ),
            f"Recording outcome for ticket {zendesk_id}",
            lambda _: self._feedback_done(
                f"Ticket {zendesk_id}: review marked correct."
            ),
        )

    def mark_selected_misclassified(self) -> None:
        """Asks for the correct category and logs the selected review as misclassified."""
        row = self._selected_row()
        if row is None or not self._configured(READ_ENV, notify=True):
            return
        zendesk_id, review = row["zendesk_id"], row["review"]
        choices = [c for c in CLASSIFICATIONS if c != review.get("classification")]
        correct, ok = QInputDialog.getItem(
            self,
            "Mark Misclassified",
            f"Ticket {zendesk_id} was classified {review.get('classification')}. "
            "What should it have been?",
            choices,
            0,
            False,
        )
        if not ok:
            return
        misread, ok = QInputDialog.getText(
            self, "Mark Misclassified", "Which signal was misread? (optional)"
        )
        if not ok:
            return

        def log_it():
            tickets = self.collection_factory()
            log, _ = feedback.get_feedback_collections(tickets.database)
            return feedback.log_misclassification(
                tickets, log, zendesk_id, review["review_id"], correct, misread.strip()
            )

        self._run(
            log_it,
            f"Logging misclassification for ticket {zendesk_id}",
            lambda _: self._feedback_done(
                f"Ticket {zendesk_id}: logged as misclassified (should be {correct})."
            ),
        )

    def _feedback_done(self, message: str) -> None:
        logger.success(message)
        self.status_label.setText(message)
        self.refresh()

    def show_accuracy_report(self) -> None:
        """Shows accuracy per category from the recorded outcomes."""
        if not self._configured(READ_ENV, notify=True):
            return
        self._run(
            lambda: feedback.accuracy_report(self.collection_factory()),
            "Building accuracy report",
            lambda report: QMessageBox.information(
                self,
                "Accuracy Report",
                (
                    feedback.format_accuracy_report(report)
                    if report
                    else "No reviews yet."
                ),
            ),
        )

    def open_weekly_review(self) -> None:
        """Opens the weekly review of misclassification_log and lessons_learned."""
        if not self._configured(READ_ENV, notify=True):
            return
        # Kept on self: a dialog with no reference is garbage-collected and closes.
        self.weekly_review_dialog = WeeklyReviewDialog(
            self._run,
            lambda: feedback.get_feedback_collections(
                self.collection_factory().database
            ),
            self,
        )
        self.weekly_review_dialog.show()

    def export_training_data(self) -> None:
        """Saves judged answers as fine-tuning JSONL or few-shot Markdown
        (decision_engine.training_export), then says what was left out."""
        if not self._configured(READ_ENV, notify=True):
            return
        path, chosen = QFileDialog.getSaveFileName(
            self,
            "Export Training Data",
            "triage-examples.jsonl",
            ";;".join(TRAINING_FORMATS),
        )
        if not path:
            return
        # The chosen format wins: its extension is added if the name lacks it.
        fmt = TRAINING_FORMATS.get(chosen)

        def task():
            tickets = self.collection_factory()
            log, _ = feedback.get_feedback_collections(tickets.database)
            ab_runs, ab_results = ab_test.get_collections(tickets.database)
            return training_export.describe(
                training_export.export_training_data(
                    tickets, log, ab_runs, ab_results, path, fmt=fmt
                )
            )

        def done(message: str) -> None:
            self.status_label.setText(message)
            QMessageBox.information(
                self,
                "Training Data Exported",
                message
                + "\n\nOnly answers made with the current prompt are included (the command "
                "line's --include-older-prompts adds the rest). The files hold customer "
                "ticket text; review them before sharing.",
            )

        self._run(task, "Exporting training data", done)

    def open_ab_test(self) -> None:
        """Opens the A/B test window: two engine configurations compared blind."""
        if not self._configured(READ_ENV, notify=True):
            return
        if self.ab_test_dialog is not None and self.ab_test_dialog.isVisible():
            self.ab_test_dialog.raise_()
            return
        # Kept on self: a dialog with no reference is garbage-collected and closes.
        self.ab_test_dialog = ABTestDialog(
            self._run,
            lambda: ab_collections(self.collection_factory()),
            lambda: self._closing,
            self,
            ready=lambda: self._configured(AB_TEST_ENV, notify=True),
        )
        self.ab_test_dialog.show()
