"""Decision Engine tab for the PyQt app.

Shows how many tickets are awaiting review, runs the decision engine's own
entry points (run_review_batch, process_one_ticket) on a background thread,
and lists recent reviews read straight from Mongo. The engine is used as-is;
nothing here changes how tickets are classified or stored.
"""

import os
from typing import Any, Callable, List, Optional

from loguru import logger
from pymongo.collection import Collection
from pymongo.errors import ServerSelectionTimeoutError
from PyQt5 import QtCore
from PyQt5.QtCore import QThread
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from decision_engine import persistence
from utility import ThreadWorker

REQUIRED_ENV = ("MONGO_URI", "CHEAPER_INFERENCE_API_KEY")
RECENT_REVIEW_LIMIT = 200
COLUMNS = ["Ticket", "Subject", "Classification", "Confidence", "Reviewed (UTC)", "Model"]
CONNECTION_HINT = (
    "Could not reach MongoDB. Check that you are on the VPN, that its exit IP "
    "is on the Atlas IP access list, and that the cluster is not paused."
)


def missing_env(names=REQUIRED_ENV) -> List[str]:
    return [name for name in names if not os.getenv(name)]


def count_awaiting(tickets: Collection) -> int:
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
        {"$project": {"_id": 0, "zendesk_id": 1, "subject": 1, "review": "$agent_reviews"}},
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
        f"Reviewed: {reviewed_at:%Y-%m-%d %H:%M} UTC · Model: {review.get('model_version')}"
        if reviewed_at
        else f"Model: {review.get('model_version')}",
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
    if review.get("leadership_recipient"):
        lines += ["", f"Leadership recipient: {review['leadership_recipient']}"]
    return "\n".join(lines)


class TriageTab(QWidget):
    """The Decision Engine tab.

    Every Mongo or model call runs on a QThread, so a slow or unreachable
    cluster never freezes the window. One job runs at a time.

    Args:
        collection_factory: Returns the tickets collection. Defaults to the
            engine's own connection so both share one client.
        review_batch: Reviews every awaiting ticket.
        review_one: Reviews one ticket by Zendesk ID; returns None if it is
            missing or not awaiting review.
    """

    def __init__(
        self,
        collection_factory: Callable[[], Collection] = persistence.get_tickets_collection,
        review_batch: Callable[[Collection], Any] = persistence.run_review_batch,
        review_one: Callable[[str, Collection], Optional[dict]] = persistence.process_one_ticket,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.collection_factory = collection_factory
        self.review_batch = review_batch
        self.review_one = review_one
        self._thread: Optional[QThread] = None
        self._worker: Optional[ThreadWorker] = None
        self._on_done: Optional[Callable[[Any], None]] = None
        self._rows: List[dict] = []
        self._loaded = False
        self._build_ui()

    # --- layout ---------------------------------------------------------------

    def _build_ui(self) -> None:
        intro = QLabel(
            "Classifies tickets awaiting review and records each result on the ticket "
            "in MongoDB. Nothing is written to Zendesk. Each ticket reviewed is a paid model call."
        )
        intro.setWordWrap(True)

        self.status_label = QLabel()
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh)
        status_row = QHBoxLayout()
        status_row.addWidget(self.status_label, stretch=1)
        status_row.addWidget(self.refresh_button)

        self.review_all_button = QPushButton("Review All Awaiting Tickets")
        self.review_all_button.clicked.connect(self.start_review_all)

        self.ticket_id_input = QLineEdit()
        self.ticket_id_input.setPlaceholderText("Zendesk ticket ID")
        self.ticket_id_input.setClearButtonEnabled(True)
        self.ticket_id_input.returnPressed.connect(self.start_review_one)
        self.review_one_button = QPushButton("Classify One Ticket")
        self.review_one_button.clicked.connect(self.start_review_one)
        one_row = QHBoxLayout()
        one_row.addWidget(self.ticket_id_input, stretch=1)
        one_row.addWidget(self.review_one_button)

        # Range 0..0 makes the bar an indeterminate "busy" indicator:
        # run_review_batch reports no progress.
        self.busy_bar = QProgressBar()
        self.busy_bar.setRange(0, 0)
        self.busy_bar.setVisible(False)

        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.setSortingEnabled(True)
        self.table.itemSelectionChanged.connect(self.show_selected_review)

        self.detail = QTextEdit()
        self.detail.setReadOnly(True)
        self.detail.setPlaceholderText("Select a review to see its reasoning, note and draft message.")

        splitter = QSplitter(QtCore.Qt.Vertical)
        splitter.addWidget(self.table)
        splitter.addWidget(self.detail)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        layout = QVBoxLayout(self)
        layout.addWidget(intro)
        layout.addLayout(status_row)
        layout.addWidget(self.review_all_button)
        layout.addLayout(one_row)
        layout.addWidget(self.busy_bar)
        layout.addWidget(QLabel(f"Recent reviews (newest {RECENT_REVIEW_LIMIT})"))
        layout.addWidget(splitter, stretch=1)

        self._set_busy(False)

    def showEvent(self, event) -> None:
        # Load on first view rather than at startup, so opening the app never
        # waits on Mongo.
        super().showEvent(event)
        if not self._loaded:
            self._loaded = True
            self.refresh()

    # --- background jobs ------------------------------------------------------

    def _configured(self, notify: bool) -> bool:
        missing = missing_env()
        if not missing:
            return True
        message = f"Not configured: set {', '.join(missing)} (see .env.example) and restart the app."
        self.status_label.setText(message)
        if notify:
            logger.error(message)
            QMessageBox.warning(self, "Decision Engine Not Configured", message)
        return False

    def _set_busy(self, busy: bool, label: str = "") -> None:
        for button in (self.refresh_button, self.review_all_button, self.review_one_button):
            button.setEnabled(not busy)
        self.ticket_id_input.setEnabled(not busy)
        self.busy_bar.setVisible(busy)
        self.busy_bar.setFormat(label)
        if busy and label:
            self.status_label.setText(label)

    def _run(self, task: Callable[[], Any], description: str, on_done: Callable[[Any], None]) -> None:
        def with_connection_hint():
            try:
                return task()
            except ServerSelectionTimeoutError as e:
                raise RuntimeError(CONNECTION_HINT) from e

        self._set_busy(True, f"{description}…")
        self._on_done = on_done
        self._thread = QThread(self)
        self._worker = ThreadWorker(with_connection_hint, description)
        self._worker.moveToThread(self._thread)
        self._worker.finished.connect(self._job_finished)
        self._worker.error.connect(self._job_failed)
        self._thread.started.connect(self._worker.run)
        self._thread.start()

    def _stop_thread(self) -> None:
        self._thread.quit()
        self._thread.wait()
        self._thread = None
        self._worker = None
        self._set_busy(False)

    def _job_finished(self, result: Any) -> None:
        # The thread is stopped before the callback so the callback can
        # start the next job.
        self._stop_thread()
        on_done, self._on_done = self._on_done, None
        on_done(result)

    def _job_failed(self, message: str) -> None:
        self._stop_thread()
        self._on_done = None
        self.status_label.setText("Last action failed; see the log.")
        QMessageBox.critical(self, "Decision Engine Error", message)

    # --- actions ---------------------------------------------------------------

    def refresh(self) -> None:
        if not self._configured(notify=False):
            return

        def load():
            tickets = self.collection_factory()
            return count_awaiting(tickets), recent_reviews(tickets)

        self._run(load, "Loading reviews from MongoDB", self._show_loaded)

    def _show_loaded(self, loaded) -> None:
        awaiting, rows = loaded
        self.status_label.setText(f"{awaiting} ticket(s) awaiting review.")
        self._rows = rows
        self._fill_table()

    def start_review_all(self) -> None:
        if not self._configured(notify=True):
            return
        # Count first (off the GUI thread), then confirm with an up-to-date number.
        self._run(
            lambda: count_awaiting(self.collection_factory()),
            "Counting tickets awaiting review",
            self._confirm_review_all,
        )

    def _confirm_review_all(self, awaiting: int) -> None:
        self.status_label.setText(f"{awaiting} ticket(s) awaiting review.")
        if awaiting == 0:
            QMessageBox.information(self, "Nothing to Review", "No tickets are awaiting review.")
            return
        answer = QMessageBox.question(
            self,
            "Review All Awaiting Tickets",
            f"Send {awaiting} ticket(s) to the model for classification?\n\n"
            "Each ticket is a paid model call. Results are saved to MongoDB only.",
            QMessageBox.Yes | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )
        if answer != QMessageBox.Yes:
            return
        self._run(
            lambda: self.review_batch(self.collection_factory()),
            f"Reviewing {awaiting} ticket(s)",
            self._review_all_done,
        )

    def _review_all_done(self, _result) -> None:
        logger.success("Review batch finished.")
        QMessageBox.information(self, "Review Complete", "Review batch finished. Results are listed below.")
        self.refresh()

    def start_review_one(self) -> None:
        zendesk_id = self.ticket_id_input.text().strip()
        if not zendesk_id:
            QMessageBox.warning(self, "No Ticket ID", "Enter a Zendesk ticket ID to classify.")
            return
        if not self._configured(notify=True):
            return
        self._run(
            lambda: self.review_one(zendesk_id, self.collection_factory()),
            f"Classifying ticket {zendesk_id}",
            lambda result: self._review_one_done(zendesk_id, result),
        )

    def _review_one_done(self, zendesk_id: str, result: Optional[dict]) -> None:
        if result is None:
            message = f"Ticket {zendesk_id} was not found, or is not awaiting review."
            logger.warning(message)
            QMessageBox.information(self, "Not Classified", message)
            return
        logger.success(f"{zendesk_id}: {result['classification']} ({result['confidence']})")
        self.ticket_id_input.clear()
        self.refresh()

    # --- results ---------------------------------------------------------------

    def _fill_table(self) -> None:
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
                item.setData(QtCore.Qt.DisplayRole, value if isinstance(value, (int, float)) else str(value or ""))
                if column == 0:
                    item.setData(QtCore.Qt.UserRole, row_index)
                self.table.setItem(row_index, column, item)
        self.table.setSortingEnabled(True)
        # Re-enabling sorting applies the header's current indicator, which
        # defaults to column 0; reset it to newest review first.
        self.table.sortItems(COLUMNS.index("Reviewed (UTC)"), QtCore.Qt.DescendingOrder)
        self.table.resizeColumnsToContents()
        self.detail.clear()

    def show_selected_review(self) -> None:
        selected = self.table.selectedItems()
        if not selected:
            self.detail.clear()
            return
        row_index = self.table.item(selected[0].row(), 0).data(QtCore.Qt.UserRole)
        self.detail.setPlainText(format_review(self._rows[row_index]))
