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

import embeddings
from decision_engine import feedback, persistence, search
from decision_engine.guards import CLASSIFICATIONS
from ab_test_dialog import ab_collections, ABTestDialog
from lessons_dialog import WeeklyReviewDialog
from loguru import logger
from pymongo.collection import Collection
from pymongo.errors import ServerSelectionTimeoutError
from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSignal, QObject, QThread
from PyQt5.QtWidgets import (
    QAbstractItemView,
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
from utility import ThreadWorker, worker_thread
from widgets import button_row, Button, Card, hint, LineEdit, ProgressBar, use_si_scrollbars

# Reading reviews only touches Mongo; classifying also needs the primary
# model. The secondary and confirmation models are optional: when unset, the
# engine skips that step.
READ_ENV = ("MONGO_URI",)
REVIEW_ENV = READ_ENV + (
    "CHEAPER_INFERENCE_API_KEY",
    "CHEAPER_INFERENCE_API_URL",
    "CHEAPER_INFERENCE_PRIMARY_MODEL_NAME",
)
# Embedding tickets reads and writes Mongo and calls Voyage AI.
EMBED_ENV = READ_ENV + ("EMBEDDING_API_KEY",)
# Optional models, with what is lost when each is unset.
OPTIONAL_MODEL_ENV = {
    "CHEAPER_INFERENCE_SECONDARY_MODEL_NAME": "no fallback if the primary model call fails",
    "CHEAPER_INFERENCE_CONFIRMATION_MODEL_NAME": "manual-triage and low-confidence results won't be re-checked",
}
RECENT_REVIEW_LIMIT = 200
# Table data role on a row's first cell: its review's ID, to keep it selected.
REVIEW_ID_ROLE = QtCore.Qt.UserRole + 1
COLUMNS = [
    "Ticket",
    "Subject",
    "Classification",
    "Confidence",
    "Reviewed (UTC)",
    "Model",
]
CONNECTION_HINT = (
    "Could not reach MongoDB. Check that you are on the VPN, that its exit IP "
    "is on the Atlas IP access list, and that the cluster is not paused."
)


def missing_env(names=REVIEW_ENV) -> List[str]:
    """Returns the names of required environment variables that are not set.

    A variable counts as missing if it is unset or set to an empty string.

    Args:
        names: Environment variable names to check. Defaults to REVIEW_ENV,
            everything a classification run needs.

    Returns:
        The missing names, in the order given. An empty list means everything
        is configured.
    """
    return [name for name in names if not os.getenv(name)]


def optional_models_note() -> str:
    """Describes which optional models are unset and what that skips.

    Returns:
        One line per unset model, or "" when both are set.
    """
    return "\n".join(
        f"{name} is not set: {effect}."
        for name, effect in OPTIONAL_MODEL_ENV.items()
        if not os.getenv(name)
    )


def count_awaiting(tickets: Collection) -> int:
    """Counts tickets that are awaiting review.

    Args:
        tickets: The MongoDB tickets collection.

    Returns:
        The number of documents where `awaiting_review` is True.
    """
    return tickets.count_documents({"awaiting_review": True})


def recent_reviews(tickets: Collection, limit: int = RECENT_REVIEW_LIMIT) -> List[dict]:
    """Returns the newest reviews across all tickets, newest first.

    Each row is {"zendesk_id", "subject", "review"}, where review is one
    agent_reviews entry exactly as the engine stored it.
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
                "review": "$agent_reviews",
            }
        },
    ]
    return list(tickets.aggregate(pipeline))


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
        *[f"  - {signal}" for signal in review.get("signals_matched") or ["(none)"]],
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


def latest_outcome(row: dict) -> Optional[dict]:
    """The newest outcome recorded for this row's review, or None."""
    review_id = row["review"].get("review_id")
    matches = [o for o in row.get("outcomes") or [] if o.get("review_id") == review_id]
    return matches[-1] if matches else None


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
    """Carries batch progress from the worker thread to the tab (queued, as it lives on the GUI thread)."""

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

        self.embed_button = Button("Embed Tickets (Voyage AI)…")
        self.embed_button.setToolTip(
            "Stores a Voyage embedding on each new or changed ticket and creates the vector index."
        )
        self.embed_button.clicked.connect(self.start_embed_tickets)

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
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.setSortingEnabled(True)
        self.table.itemSelectionChanged.connect(self.show_selected_review)

        self.detail = use_si_scrollbars(QTextEdit())
        self.detail.setReadOnly(True)
        self.detail.setPlaceholderText(
            "Select a review to see its reasoning, note and draft message."
        )

        self.mark_good_button = Button("Mark Correct")
        self.mark_good_button.clicked.connect(self.mark_selected_correct)
        self.mark_bad_button = Button("Mark Misclassified…")
        self.mark_bad_button.clicked.connect(self.mark_selected_misclassified)
        self.accuracy_button = Button("Accuracy Report")
        self.accuracy_button.clicked.connect(self.show_accuracy_report)
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
        feedback_row = QHBoxLayout()
        feedback_row.addWidget(self.mark_good_button)
        feedback_row.addWidget(self.mark_bad_button)
        feedback_row.addStretch()
        feedback_row.addWidget(self.ab_test_button)
        feedback_row.addWidget(self.weekly_review_button)
        feedback_row.addWidget(self.accuracy_button)
        detail_pane = QWidget()
        detail_layout = QVBoxLayout(detail_pane)
        detail_layout.setContentsMargins(0, 0, 0, 0)
        detail_layout.addWidget(self.detail)
        detail_layout.addLayout(feedback_row)

        splitter = QSplitter(QtCore.Qt.Vertical)
        splitter.addWidget(self.table)
        splitter.addWidget(detail_pane)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        review_card = Card("Review Tickets")
        review_card.body.addWidget(intro)
        review_card.body.addLayout(status_row)
        review_card.body.addLayout(one_row)
        review_card.body.addLayout(button_row(self.batch_button, self.embed_button))
        review_card.body.addWidget(self.busy_bar)
        batch_row = QHBoxLayout()
        batch_row.addWidget(self.batch_label, stretch=1)
        batch_row.addWidget(self.stop_button)
        review_card.body.addLayout(batch_row)

        reviews_card = Card(f"Recent Reviews (newest {RECENT_REVIEW_LIMIT})")
        reviews_card.body.addWidget(splitter)

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
            names: The variables this operation needs (READ_ENV or REVIEW_ENV).
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
        """Returns True while a background job is running.

        Callers such as the main window can use this to avoid closing the app
        mid-job.
        """
        return self._thread is not None

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
        if self._thread is None:
            return
        self._cancel.set()
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
            self.embed_button,
            self.weekly_review_button,
            self.ab_test_button,
        ):
            button.setEnabled(not busy)
        has_selection = self._selected_row() is not None
        for button in (self.mark_good_button, self.mark_bad_button):
            button.setEnabled(not busy and has_selection)
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
            self.idle.emit()
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
            self.idle.emit()
            return
        self.status_label.setText("Last action failed; see the log.")
        QMessageBox.critical(self, "Decision Engine Error", message)
        if on_error is not None:
            on_error(message)

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
        self.status_label.setText(f"{awaiting} ticket(s) awaiting review.")
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
        started = self._run(
            lambda: self.review_batch(
                self.collection_factory(),
                list(zendesk_ids),
                on_progress=self._batch_relay.progress.emit,
            ),
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
            self.batch_label.text().split(" · about")[0] + " · stopping after this ticket…"
        )
        logger.info("Decision Engine: batch will stop after the current ticket.")

    def _on_batch_progress(self, progress: dict) -> None:
        """After each ticket: moves the bar, updates the counts and time left,
        and adds a stored review to Recent Reviews."""
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
            row = {
                "zendesk_id": zendesk_id,
                "subject": self._batch["subjects"].get(zendesk_id),
                "review": review,
            }
            self._rows = [row, *self._rows][:RECENT_REVIEW_LIMIT]
            self._fill_table()

    def start_embed_tickets(self) -> None:
        """Checks the vector index and lists stale tickets, then asks before sending them to Voyage.

        The index is checked first so a mismatch stops the run before any
        paid call; only the tickets in the approved list are sent.
        """
        if not self._configured(EMBED_ENV, notify=True):
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
                tickets, cfg, plan, should_stop=self._cancel.is_set
            )
            # A VectorIndexMismatch here (the index changed in Atlas during the
            # run) propagates and is shown as an error, not a warning.
            try:
                embeddings.finish_index(tickets, cfg, summary, status)
            except search.SearchUnavailable as e:
                summary["index"] = f"skipped: {e}"
            return summary

        self._run(
            embed_and_index, f"Embedding {len(plan.embed)} ticket(s)", self._embed_done
        )

    def _embed_done(self, summary: dict) -> None:
        message = embeddings.format_summary(summary)
        logger.success(message.replace("\n", "; "))
        QMessageBox.information(self, "Embedding Complete", message)

    def _batch_done(self, summary) -> None:
        """Logs and announces the batch's end, then refreshes the list.

        Args:
            summary: The batch's counts from run_review_batch, or None.
        """
        stopped_by_user = isinstance(summary, dict) and summary.get("cancelled")
        logger.success("Review batch stopped." if stopped_by_user else "Review batch finished.")
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
        for row_index, row in enumerate(self._rows):
            review = row["review"]
            reviewed_at = review.get("reviewed_at")
            values = [
                row.get("zendesk_id"),
                row.get("subject"),
                review.get("classification"),
                review.get("confidence"),
                f"{reviewed_at:%Y-%m-%d %H:%M}" if reviewed_at else "",
                review.get("model_version"),
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem()
                # DisplayRole with the raw value so numbers sort numerically.
                item.setData(
                    QtCore.Qt.DisplayRole,
                    value if isinstance(value, (int, float)) else str(value or ""),
                )
                if column == 0:
                    item.setData(QtCore.Qt.UserRole, row_index)
                    item.setData(REVIEW_ID_ROLE, str(review.get("review_id") or ""))
                self.table.setItem(row_index, column, item)
        self.table.setSortingEnabled(True)
        # Re-enabling sorting applies the header's current indicator, which
        # defaults to column 0; reset it to newest review first.
        self.table.sortItems(COLUMNS.index("Reviewed (UTC)"), QtCore.Qt.DescendingOrder)
        self.table.resizeColumnsToContents()
        # Rows added during a batch keep the review being read selected.
        if keep:
            for table_row in range(self.table.rowCount()):
                if self.table.item(table_row, 0).data(REVIEW_ID_ROLE) == keep:
                    self.table.selectRow(table_row)
                    return
        self.detail.clear()

    def show_selected_review(self) -> None:
        """Shows the selected row's full review in the detail pane.

        Looks the review up through the row index stored on the row's first
        cell, so it stays correct after the user re-sorts the table. Clears the
        pane when nothing is selected.
        """
        row = self._selected_row()
        enabled = row is not None and not self.is_busy()
        self.mark_good_button.setEnabled(enabled)
        self.mark_bad_button.setEnabled(enabled)
        if row is None:
            self.detail.clear()
            return
        self.detail.setPlainText(format_review(row))

    def _selected_row(self) -> Optional[dict]:
        """The loaded row behind the selected table row, or None."""
        selected = self.table.selectedItems() if hasattr(self, "table") else []
        if not selected:
            return None
        return self._rows[
            self.table.item(selected[0].row(), 0).data(QtCore.Qt.UserRole)
        ]

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
            ready=lambda: self._configured(REVIEW_ENV, notify=True),
        )
        self.ab_test_dialog.show()
