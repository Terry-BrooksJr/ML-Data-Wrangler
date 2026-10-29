"""A/B testing of two decision-engine configurations, reviewed blind.

A run classifies a sample of tickets with configuration A (the current
engine by default) and configuration B (the one being considered); see
decision_engine.ab_test. Here a reviewer creates and classifies runs, then
reads each ticket's two answers side by side as "Output 1" and "Output 2",
in an order shuffled per ticket, and records which is better. Show Results
names the variants and tallies the verdicts.

Every Mongo and model call goes through the Decision Engine tab's job
runner, so it runs on a worker thread and never overlaps another tab job.
Nothing is written to the tickets collection.
"""

import json
import math
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

from ui import zendesk_links
from decision_engine import ab_test, pricing
from decision_engine.classify import EFFORT_LEVELS, EngineConfig, MAX_TOKENS_LIMIT
from decision_engine.identity import current_user
from loguru import logger
from pymongo.collection import Collection
from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSignal, QObject
from PyQt5.QtGui import QFontDatabase
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from ui.widgets import (
    Button,
    Card,
    CheckBox,
    hint,
    LineEdit,
    ProgressBar,
    use_si_scrollbars,
)

# run(task, description, on_done, on_error) -> False if another job is running.
Runner = Callable[
    [Callable[[], Any], str, Callable[[Any], None], Optional[Callable[[str], None]]],
    bool,
]
RUN_COLUMNS = ["Created (UTC)", "Name", "Tickets", "Status", "Reviewed"]
DEFAULT_SAMPLE_SIZE = 30
MAX_SAMPLE_SIZE = 1000
DEFAULT_EFFORT = "Model default"
# Fields a reviewer reads for each ticket; the transcript can be long.
BUSY_MESSAGE = "Another Decision Engine job is running. Try again when it finishes."


def reviewer_identity() -> str:
    """Who is reviewing: the computer login (decision_engine.identity)."""
    return current_user()


def format_output(output: Optional[dict]) -> str:
    """One variant's answer, without naming its models (the review is blind).

    Plain text, not HTML: the reasoning and draft can quote customer text.
    """
    if not output:
        return "Not classified yet."
    if "error" in output:
        attempts = output.get("attempts") or 1
        state = (
            f"Gave up after {attempts} failed attempts."
            if attempts >= ab_test.MAX_ATTEMPTS
            else f"Failed {attempts} time(s); Classify Pending tries again."
        )
        return (
            f"The engine failed on this ticket. {state}\n{format_cost(output)}\n"
            f"{output['error']}"
        )
    lines = [
        f"Classification: {output.get('classification')}",
        f"Confidence: {output.get('confidence')}",
        format_cost(output),
    ]
    if output.get("decision_only"):
        return "\n".join(lines + _classification_only(output))
    if output.get("confirmed_from"):
        first = output["confirmed_from"]
        lines.append(
            f"Re-checked: first answer was {first.get('classification')} "
            f"(confidence {first.get('confidence')})"
        )
    if output.get("leadership_recipient"):
        lines.append(f"Leadership recipient: {output['leadership_recipient']}")
    lines += [
        "",
        "Signals matched:",
        *[f"  - {signal}" for signal in output.get("signals_matched") or ["(none)"]],
        "",
        "Reasoning:",
        output.get("reasoning") or "(none)",
        "",
        "Internal note:",
        output.get("internal_note") or "(none)",
    ]
    if output.get("draft_message"):
        lines += ["", "Draft message (unsent):", output["draft_message"]]
    if output.get("guard_overrides"):
        lines += [
            "",
            "Overridden by code checks:",
            *[f"  - {o}" for o in output["guard_overrides"]],
        ]
    return "\n".join(lines)


def format_cost(output: dict) -> str:
    """One answer's cost: what was charged or estimated, over how many calls,
    and whether it was reused (so cost nothing here)."""
    if "cost_usd" not in output:
        return "Cost: not recorded"
    calls = output.get("model_calls") or 0
    detail = ", ".join(
        part
        for part in (
            output.get("cost_basis"),
            f"{calls} model call{'s' if calls != 1 else ''}" if calls else None,
        )
        if part
    )
    text = f"Cost: {pricing.format_usd(output.get('cost_usd'))}" + (
        f" ({detail})" if detail else ""
    )
    if output.get("reused_from"):
        text += "; reused from an earlier test, so nothing was spent here"
    return text


def _classification_only(output: dict) -> List[str]:
    """The rest of a decision model's answer: probabilities, and what the
    code checks changed, since there is no reasoning, note or draft."""
    lines = [
        "",
        "Classification only: this answer comes from a model that picks a "
        "category without writing reasoning, a note or a draft.",
    ]
    probabilities = output.get("probabilities") or {}
    if probabilities:
        lines += ["", "Probabilities:"]
        for name, probability in sorted(
            probabilities.items(), key=lambda item: -(item[1] or 0)
        ):
            lines.append(f"  - {name}: {(probability or 0):.0%}")
    if output.get("leadership_recipient"):
        lines += ["", f"Leadership recipient: {output['leadership_recipient']}"]
    if output.get("internal_note"):
        lines += ["", "Internal note (added by code checks):", output["internal_note"]]
    if output.get("guard_overrides"):
        lines += [
            "",
            "Overridden by code checks:",
            *[f"  - {o}" for o in output["guard_overrides"]],
        ]
    return lines


def _payload_transcript(payload: Optional[str]) -> Optional[List[dict]]:
    """The transcript in a saved model payload (build_ticket_payload: a
    preamble, a blank line, then JSON), or None if there is none."""
    if not payload or "\n\n" not in payload:
        return None
    try:
        return json.loads(payload.split("\n\n", 1)[1]).get("transcript")
    except ValueError:
        return None


def format_ticket(result: Optional[dict]) -> str:
    """A result's ticket as both sides were sent it: subject, status and
    transcript, oldest entry first.

    The transcript comes from the saved payload; results saved before that
    kept it in ticket.transcript. Comment bodies are shown as sent, inside
    their <customer_text> markers.
    """
    if not result:
        return ""
    ticket = result.get("ticket") or {}
    lines = [
        f"Ticket {ticket.get('zendesk_id') or result.get('ticket_id')}: "
        f"{ticket.get('subject') or '(subject removed)'}",
        f"Status: {ticket.get('status') or '-'}",
    ]
    transcript = _payload_transcript(result.get("payload"))
    if transcript is None:
        transcript = ticket.get("transcript")
    if transcript is None:
        lines += [
            "",
            "The ticket text was removed when this test was finalized; the full "
            "ticket is in the tickets collection.",
        ]
    for entry in transcript or []:
        kind = "Internal note" if entry.get("is_internal_note") else "Comment"
        when = entry.get("timestamp")
        lines += [
            "",
            f"— {kind} by {entry.get('author') or 'unknown'} ({entry.get('role') or '-'})"
            + (
                f", {when:%Y-%m-%d %H:%M}"
                if hasattr(when, "strftime")
                else f", {when[:16].replace('T', ' ')}" if isinstance(when, str) else ""
            ),
            str(entry.get("body") or "").strip(),
        ]
    return "\n".join(lines)


def _live_prices() -> Dict[str, pricing.Price]:
    from decision_engine import decisions
    from decision_engine.classify import get_client

    return pricing.parse_listing(decisions.model_list(get_client(), refresh=True))


def runs_with_counts(runs: Collection, results: Collection) -> List[dict]:
    """The newest runs, each with "reviewed": how many of its tickets have a verdict."""
    listed = ab_test.list_runs(runs)
    reviewed = {
        row["_id"]: row["count"]
        for row in results.aggregate(
            [
                {"$match": {"preference": {"$ne": None}}},
                {"$group": {"_id": "$run_id", "count": {"$sum": 1}}},
            ]
        )
    }
    for run in listed:
        run["reviewed"] = reviewed.get(run["_id"], 0)
    return listed


def ab_collections(tickets: Collection) -> Tuple[Collection, Collection, Collection]:
    """(tickets, ab_tests, ab_results), the A/B collections beside tickets."""
    return (tickets, *ab_test.get_collections(tickets.database))


class ConfigForm(QWidget):
    """The models and settings of one variant."""

    def __init__(self, config: EngineConfig, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.primary = QLineEdit(config.primary)
        self.secondary = QLineEdit(config.secondary or "")
        self.secondary.setPlaceholderText("none")
        self.confirmation = QLineEdit(config.confirmation or "")
        self.confirmation.setPlaceholderText("none")
        self.arbiter = QLineEdit(config.arbiter or "")
        self.arbiter.setPlaceholderText("none: disagreements go to manual triage")
        self.max_tokens = QSpinBox()
        self.max_tokens.setRange(256, MAX_TOKENS_LIMIT)
        self.max_tokens.setSingleStep(1024)
        self.max_tokens.setValue(config.max_tokens)
        self.effort = QComboBox()
        self.effort.addItem(DEFAULT_EFFORT, None)
        for level in EFFORT_LEVELS:
            self.effort.addItem(level, level)
        self.effort.setCurrentIndex(max(0, self.effort.findData(config.effort)))
        self.effort.setToolTip(
            "Sent as output_config.effort. Leave at the model default for models "
            "other than Claude, which may reject it."
        )
        form = QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)
        form.addRow("Primary model", self.primary)
        form.addRow("Secondary model", self.secondary)
        form.addRow("Confirmation model", self.confirmation)
        form.addRow("Arbiter model", self.arbiter)
        form.addRow("Max tokens", self.max_tokens)
        form.addRow("Effort", self.effort)

    def config(self) -> EngineConfig:
        """Raises ValueError for an invalid configuration (no primary model)."""
        return EngineConfig(
            primary=self.primary.text().strip(),
            secondary=self.secondary.text().strip() or None,
            confirmation=self.confirmation.text().strip() or None,
            arbiter=self.arbiter.text().strip() or None,
            max_tokens=self.max_tokens.value(),
            effort=self.effort.currentData(),
        )


PRICE_COLUMNS = ["Model", "Input", "Output", "Cache read", "Cache write", "Source"]
# Price fields in column order (after Model), as pricing.Price names them.
_PRICE_FIELDS = ("input", "output", "cache_read", "cache_write")


class PriceTable(QTableWidget):
    """Prices per million tokens for the models in a test, editable.

    Starts from today's prices (live); a value typed in marks that model's
    prices "entered" and is kept if the model list changes.
    """

    def __init__(self, live: Dict[str, pricing.Price], parent=None) -> None:
        super().__init__(0, len(PRICE_COLUMNS), parent)
        use_si_scrollbars(self)
        self.live = live
        self.entered: Dict[str, Dict[str, str]] = {}
        self._filling = False
        self.setHorizontalHeaderLabels(PRICE_COLUMNS)
        self.verticalHeader().setVisible(False)
        self.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.setMaximumHeight(170)
        self.itemChanged.connect(self._edited)

    def show_models(self, models: List[str]) -> None:
        self._filling = True
        self.setRowCount(len(models))
        for row, model in enumerate(models):
            name = QTableWidgetItem(model)
            name.setFlags(name.flags() & ~QtCore.Qt.ItemIsEditable)
            self.setItem(row, 0, name)
            live = self.live.get(model)
            entered = self.entered.get(model, {})
            for column, field in enumerate(_PRICE_FIELDS, start=1):
                value = entered.get(field)
                if value is None:
                    number = getattr(live, field) if live else None
                    value = "" if number is None else f"{number:g}"
                self.setItem(row, column, QTableWidgetItem(value))
            source = (
                "entered" if entered else "live" if live else "not listed: enter it"
            )
            cell = QTableWidgetItem(source)
            cell.setFlags(cell.flags() & ~QtCore.Qt.ItemIsEditable)
            self.setItem(row, len(PRICE_COLUMNS) - 1, cell)
        self._filling = False

    def _edited(self, item: QTableWidgetItem) -> None:
        if self._filling or item.column() in (0, len(PRICE_COLUMNS) - 1):
            return
        model = self.item(item.row(), 0).text()
        values = self.entered.setdefault(model, {})
        values[_PRICE_FIELDS[item.column() - 1]] = item.text().strip()
        self._filling = True
        self.item(item.row(), len(PRICE_COLUMNS) - 1).setText("entered")
        self._filling = False

    def prices(self) -> Dict[str, pricing.Price]:
        """Model -> Price for every row with input and output prices.

        Raises:
            ValueError: a price isn't a number, or is negative.
        """
        prices = {}
        for row in range(self.rowCount()):
            model = self.item(row, 0).text()
            values = {}
            for column, field in enumerate(_PRICE_FIELDS, start=1):
                text = self.item(row, column).text().strip().lstrip("$")
                if not text:
                    values[field] = None
                    continue
                try:
                    values[field] = float(text)
                except ValueError:
                    raise ValueError(
                        f"The {PRICE_COLUMNS[column].lower()} price for {model} isn't a number."
                    )
                if not math.isfinite(values[field]) or values[field] < 0:
                    raise ValueError(
                        f"Prices for {model} must be finite numbers of 0 or more."
                    )
            if values["input"] is None or values["output"] is None:
                continue
            source = "entered" if model in self.entered else "live"
            prices[model] = pricing.Price(source=source, **values)
        return prices


class NewRunDialog(QDialog):
    """Asks for the two configurations, their prices, the sample size and a name.

    Args:
        live_prices: Today's prices from Cheaper Inference, for every listed
            model; may be empty if they couldn't be read.
    """

    def __init__(
        self,
        current: EngineConfig,
        parent: Optional[QWidget] = None,
        live_prices: Optional[Dict[str, pricing.Price]] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("New A/B Test")
        self.a_form = ConfigForm(current)
        self.b_form = ConfigForm(current)
        self.price_table = PriceTable(live_prices or {})
        for form in (self.a_form, self.b_form):
            for field in (form.primary, form.secondary, form.confirmation, form.arbiter):
                field.textChanged.connect(lambda _: self._refresh_prices())
        self._refresh_prices()
        price_card = Card("Prices (US dollars per million tokens)")
        price_card.body.addWidget(
            hint(
                "Today's prices from Cheaper Inference, discounts included"
                + ("" if live_prices else " (they couldn't be read: enter them)")
                + ". A call's actual charge is used whenever its reply reports one; "
                "these prices cost the rest. Edit any that are wrong."
            )
        )
        price_card.body.addWidget(self.price_table)
        a_card = Card("A: current engine")
        a_card.body.addWidget(self.a_form)
        b_card = Card("B: the configuration to try")
        b_card.body.addWidget(self.b_form)
        forms = QHBoxLayout()
        forms.addWidget(a_card)
        forms.addWidget(b_card)

        self.size = QSpinBox()
        self.size.setRange(1, MAX_SAMPLE_SIZE)
        self.size.setValue(DEFAULT_SAMPLE_SIZE)
        self.awaiting_only = CheckBox("Only tickets awaiting review")
        self.name = QLineEdit()
        self.name.setPlaceholderText("e.g. Opus 5.5 primary, Fable confirmation")
        sample = QFormLayout()
        sample.addRow("Tickets", self.size)
        sample.addRow("", self.awaiting_only)
        hours = int(ab_test.REUSE_WINDOW.total_seconds() // 3600)
        self.reuse = CheckBox(f"Reuse answers saved in the last {hours} hours")
        self.reuse.setChecked(True)
        self.reuse.setToolTip(
            "Where any earlier test asked the same models about the same ticket "
            "content, copy that answer instead of paying for the call again."
        )
        sample.addRow("", self.reuse)
        sample.addRow("Name", self.name)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Create")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(
            hint(
                "Each ticket in the sample is classified by both configurations: "
                "two or more paid model calls per ticket. Nothing is saved to the tickets "
                "until every verdict is in."
            )
        )
        layout.addLayout(forms)
        layout.addWidget(price_card)
        layout.addLayout(sample)
        layout.addWidget(buttons)

    def accept(self) -> None:
        try:
            a, b = self.configs()
        except ValueError as e:
            QMessageBox.warning(self, "Check the Settings", str(e))
            return
        if a == b:
            QMessageBox.warning(
                self, "Check the Settings", "A and B are the same; change B."
            )
            return
        try:
            self.price_table.prices()
        except ValueError as e:
            QMessageBox.warning(self, "Check the Prices", str(e))
            return
        super().accept()

    def configs(self) -> Tuple[EngineConfig, EngineConfig]:
        return self.a_form.config(), self.b_form.config()

    def _refresh_prices(self) -> None:
        models = []
        for form in (self.a_form, self.b_form):
            for field in (form.primary, form.secondary, form.confirmation, form.arbiter):
                model = field.text().strip()
                if model and model not in models:
                    models.append(model)
        self.price_table.show_models(models)

    def prices(self) -> Dict[str, pricing.Price]:
        """The prices to save with the test, for the models it uses."""
        return self.price_table.prices()


class ProgressRelay(QObject):
    """Carries classification progress from the worker thread (queued: it lives on the GUI thread)."""

    progress = pyqtSignal(int, int)


class ABTestDialog(QDialog):
    """Lists A/B runs, classifies them, and records blind verdicts ticket by ticket.

    Args:
        run: The tab's job runner (TriageTab._run).
        collections: Returns (tickets, ab_tests, ab_results); called on the worker thread.
        closing: Returns True once the app is closing; a classification
            then stops at the next ticket.
        ready: Returns True if the model settings are present, warning the
            user if not; checked before anything that calls a model.
        classify: Classifies one ticket, as decision_engine.classify_ticket (tests replace it).
        price_source: Returns today's prices (tests replace it).
    """

    def __init__(
        self,
        run: Runner,
        collections: Callable[[], Tuple[Collection, Collection, Collection]],
        closing: Callable[[], bool] = lambda: False,
        parent: Optional[QWidget] = None,
        ready: Callable[[], bool] = lambda: True,
        classify: Optional[Callable[..., dict]] = None,
        price_source: Optional[Callable[[], Dict[str, pricing.Price]]] = None,
    ) -> None:
        super().__init__(parent)
        self.run = run
        self.collections = collections
        self.closing = closing
        self.ready = ready
        # Today's prices for every listed model; runs on the worker thread.
        self.price_source = price_source or _live_prices
        self.classify = classify
        self._stop = threading.Event()
        self._relay = ProgressRelay(self)
        self._relay.progress.connect(self._on_progress)
        self.runs: List[dict] = []
        self.current: Optional[dict] = None  # the run being reviewed
        self.docs: List[dict] = []
        self.position = 0  # index into visible()
        self.classifying = False
        self.setWindowTitle("A/B Test")
        self.resize(1100, 820)
        self._build_ui()
        self.load_runs()

    # --- layout ---------------------------------------------------------------

    def _build_ui(self) -> None:
        self.run_table = use_si_scrollbars(QTableWidget(0, len(RUN_COLUMNS)))
        self.run_table.setHorizontalHeaderLabels(RUN_COLUMNS)
        self.run_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.run_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.run_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.run_table.verticalHeader().setVisible(False)
        self.run_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.run_table.setMaximumHeight(170)
        self.run_table.itemSelectionChanged.connect(self._update_buttons)
        self.run_table.itemDoubleClicked.connect(lambda _: self.open_selected())

        self.new_button = Button("New Test…", primary=True)
        self.new_button.clicked.connect(self.new_run)
        self.open_button = Button("Review")
        self.open_button.clicked.connect(self.open_selected)
        self.classify_button = Button("Classify Pending")
        self.classify_button.setToolTip(
            "Classify the selected run's tickets that are missing an answer (paid calls)."
        )
        self.classify_button.clicked.connect(self.classify_selected)
        self.results_button = Button("Show Results")
        self.results_button.setToolTip("Reveals which output was A and which was B.")
        self.results_button.clicked.connect(self.show_results)
        self.export_button = Button("Export…")
        self.export_button.setToolTip(
            "Save the selected run as CSV (one row per ticket) or JSON (everything kept)."
        )
        self.export_button.clicked.connect(self.export_selected)
        self.refresh_button = Button("Refresh")
        self.refresh_button.clicked.connect(self.load_runs)
        run_buttons = QHBoxLayout()
        for button in (self.new_button, self.open_button, self.classify_button):
            run_buttons.addWidget(button)
        run_buttons.addStretch()
        run_buttons.addWidget(self.results_button)
        run_buttons.addWidget(self.export_button)
        run_buttons.addWidget(self.refresh_button)

        self.progress_bar = ProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_label = hint("")
        self.progress_label.setVisible(False)
        self.stop_button = Button("Stop After This Ticket")
        self.stop_button.clicked.connect(self.stop)
        self.stop_button.setVisible(False)
        progress_row = QHBoxLayout()
        progress_row.addWidget(self.progress_label, stretch=1)
        progress_row.addWidget(self.stop_button)

        runs_card = Card("Runs")
        runs_card.body.addWidget(self.run_table)
        runs_card.body.addLayout(run_buttons)
        runs_card.body.addWidget(self.progress_bar)
        runs_card.body.addLayout(progress_row)

        self.position_label = QLabel("Open a run to review it.")
        self.only_differences = CheckBox("Only where they disagree")
        self.only_differences.toggled.connect(self._filter_changed)
        self.only_unreviewed = CheckBox("Only unreviewed")
        self.only_unreviewed.toggled.connect(self._filter_changed)
        self.prev_button = Button("◀ Previous")
        self.prev_button.clicked.connect(lambda: self._move(-1))
        self.next_button = Button("Next ▶")
        self.next_button.clicked.connect(lambda: self._move(1))
        nav = QHBoxLayout()
        nav.addWidget(self.position_label, stretch=1)
        nav.addWidget(self.only_differences)
        nav.addWidget(self.only_unreviewed)
        nav.addWidget(self.prev_button)
        nav.addWidget(self.next_button)
        self.zendesk_button = Button(zendesk_links.BUTTON_TEXT)
        self.zendesk_button.setToolTip(
            zendesk_links.BUTTON_TIP
            + " Zendesk shows the ticket as it is now, not as the models saw it."
        )
        self.zendesk_button.clicked.connect(self.view_current_on_zendesk)
        nav.addWidget(self.zendesk_button)

        mono = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        self.ticket_view = self._text_view("The ticket appears here.")
        self.output_views = []
        outputs = QSplitter(QtCore.Qt.Horizontal)
        for side in (1, 2):
            view = self._text_view(f"Output {side} appears here.")
            box = Card(f"Output {side}")
            box.body.addWidget(view)
            outputs.addWidget(box)
            self.output_views.append(view)
        splitter = QSplitter(QtCore.Qt.Vertical)
        splitter.addWidget(self.ticket_view)
        splitter.addWidget(outputs)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        self.ticket_view.document().setDefaultFont(mono)

        self.verdict_label = hint("")
        self.note_input = LineEdit("Note", "")
        self.note_input.setPlaceholderText("Why (optional)")
        self.reviewer = reviewer_identity()
        self.reviewer_label = hint(
            f"Reviewing as {self.reviewer} (your computer login)"
        )
        self.choice_buttons = {
            "1": Button("Output 1 Better"),
            "2": Button("Output 2 Better"),
            "tie": Button("Equally Good"),
            "both_bad": Button("Both Wrong"),
        }
        for key, button in self.choice_buttons.items():
            button.clicked.connect(lambda _=False, key=key: self.record(key))
        verdict_row = QHBoxLayout()
        for button in self.choice_buttons.values():
            verdict_row.addWidget(button)
        verdict_row.addStretch()
        verdict_row.addWidget(self.reviewer_label)
        review_card = Card("Blind Review")
        review_card.body.addLayout(nav)
        review_card.body.addWidget(splitter, stretch=1)
        review_card.body.addWidget(self.verdict_label)
        review_card.body.addWidget(self.note_input)
        review_card.body.addLayout(verdict_row)

        self.status_label = hint("")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(12)
        layout.addWidget(runs_card)
        layout.addWidget(self.status_label)
        layout.addWidget(review_card, stretch=1)
        self._update_buttons()

    @staticmethod
    def _text_view(placeholder: str) -> QTextEdit:
        view = use_si_scrollbars(QTextEdit())
        view.setReadOnly(True)
        view.setPlaceholderText(placeholder)
        return view

    # --- state ------------------------------------------------------------------

    def _selected_run(self) -> Optional[dict]:
        rows = {index.row() for index in self.run_table.selectedIndexes()}
        return self.runs[rows.pop()] if len(rows) == 1 else None

    def visible(self) -> List[dict]:
        """The current run's tickets that pass the filters."""
        docs = self.docs
        if self.only_differences.isChecked():
            docs = [d for d in docs if ab_test.differs(d)]
        if self.only_unreviewed.isChecked():
            docs = [d for d in docs if not d.get("preference")]
        return docs

    def _current_doc(self) -> Optional[dict]:
        docs = self.visible()
        return docs[self.position] if 0 <= self.position < len(docs) else None

    def view_current_on_zendesk(self) -> None:
        """Opens the ticket being reviewed in Zendesk."""
        doc = self._current_doc()
        if doc is not None:
            zendesk_links.open_ticket(doc["ticket_id"])

    def _update_buttons(self) -> None:
        selected = self._selected_run() is not None
        idle = not self.classifying
        self.new_button.setEnabled(idle)
        self.refresh_button.setEnabled(idle)
        for button in (
            self.open_button,
            self.classify_button,
            self.results_button,
            self.export_button,
        ):
            button.setEnabled(idle and selected)
        docs = self.visible()
        doc = self._current_doc()
        self.prev_button.setEnabled(self.position > 0)
        self.next_button.setEnabled(self.position < len(docs) - 1)
        self.zendesk_button.setEnabled(doc is not None)
        finalized = (self.current or {}).get("status") == ab_test.FINALIZED
        can_judge = doc is not None and ab_test.answered(doc) and idle and not finalized
        for button in self.choice_buttons.values():
            button.setEnabled(can_judge)

    def _busy(self) -> None:
        if not self.closing():  # closing: say nothing
            self.status_label.setText(BUSY_MESSAGE)

    # --- runs -------------------------------------------------------------------

    def load_runs(self) -> None:
        def task():
            _, runs, results = self.collections()
            return runs_with_counts(runs, results)

        if not self.run(task, "Loading A/B runs", self._show_runs, None):
            self._busy()

    def _show_runs(self, runs: List[dict]) -> None:
        selected = self._selected_run()
        self.runs = runs
        self.run_table.setRowCount(len(runs))
        for row, run in enumerate(runs):
            created = run.get("created_at")
            cells = [
                f"{created:%Y-%m-%d %H:%M}" if created else "",
                run.get("name", ""),
                str(run.get("ticket_count", "")),
                run.get("status", ""),
                str(run.get("reviewed", 0)),
            ]
            for column, text in enumerate(cells):
                self.run_table.setItem(row, column, QTableWidgetItem(text))
            if selected and run["_id"] == selected["_id"]:
                self.run_table.selectRow(row)
        if runs and self._selected_run() is None:
            self.run_table.selectRow(0)
        self.status_label.setText(f"{len(runs)} A/B run(s).")
        self._update_buttons()

    def new_run(self) -> None:
        """Reads today's prices (off the GUI thread), then asks for the test."""
        if not self.ready():
            return
        if not self.run(
            self.price_source,
            "Reading today's prices",
            self._ask_new_run,
            lambda _: self._ask_new_run({}),
        ):
            self._busy()

    def _ask_new_run(self, live_prices: Dict[str, pricing.Price]) -> None:
        dialog = NewRunDialog(EngineConfig.from_env(), self, live_prices)
        if dialog.exec_() != QDialog.Accepted:
            return
        a, b = dialog.configs()
        prices = dialog.prices()
        size = dialog.size.value()
        awaiting_only = dialog.awaiting_only.isChecked()
        reuse = dialog.reuse.isChecked()
        name = dialog.name.text().strip()
        reviewer = self.reviewer

        def task():
            tickets, runs, results = self.collections()
            ids = ab_test.sample_ticket_ids(tickets, size, awaiting_only=awaiting_only)
            run_id = ab_test.create_run(
                tickets,
                runs,
                results,
                a,
                b,
                ids,
                name,
                reviewer,
                reuse_answers=reuse,
                prices=prices,
            )
            run = runs.find_one({"_id": run_id})
            return run, ab_test.estimate_calls(runs, results, run_id)

        if not self.run(task, "Creating an A/B run", self._run_created, None):
            self._busy()

    def _run_created(self, created: Tuple[dict, dict]) -> None:
        run, estimate = created
        skipped = run.get("skipped") or []
        self.status_label.setText(
            f"Created A/B run {run['_id']} with {run['ticket_count']} ticket(s)"
            + (
                f"; left out {len(skipped)} too large to store "
                f"({', '.join(s['ticket_id'] for s in skipped)})"
                if skipped
                else ""
            )
            + "."
        )
        answer = QMessageBox.question(
            self,
            "Classify Now?",
            f"Classify the {run['ticket_count']} ticket(s) with both configurations now? "
            + ab_test.describe_estimate(estimate),
        )
        if answer == QMessageBox.Yes:
            self._classify(run["_id"])
        else:
            self.load_runs()

    def classify_selected(self) -> None:
        """Works out what classifying would cost (off the GUI thread), then asks."""
        run = self._selected_run()
        if run is None or not self.ready():
            return

        def task():
            _, runs, results = self.collections()
            return ab_test.estimate_calls(runs, results, run["_id"])

        if not self.run(
            task,
            "Checking what's left to classify",
            lambda estimate: self._confirm_classify(run, estimate),
            None,
        ):
            self._busy()

    def _confirm_classify(self, run: dict, estimate: dict) -> None:
        retry = False
        if estimate["interrupted"]:
            answer = QMessageBox.question(
                self,
                "Retry Interrupted Calls?",
                f"{estimate['interrupted']} call(s) in “{run.get('name')}” were interrupted "
                "(the app stopped mid-call) and may already have been charged. Retry them too?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            retry = answer == QMessageBox.Yes
            if retry:
                estimate = {
                    **estimate,
                    "answers": estimate["answers"] + estimate["interrupted"],
                    "calls": estimate["calls"] + estimate["interrupted"],
                }
        if not estimate["answers"]:
            QMessageBox.information(
                self, "Nothing to Classify", ab_test.describe_estimate(estimate)
            )
            return
        answer = QMessageBox.question(
            self,
            "Classify Pending Tickets?",
            f"Classify what's left in “{run.get('name')}”? "
            + ab_test.describe_estimate(estimate),
        )
        if answer == QMessageBox.Yes:
            self._classify(run["_id"], retry_interrupted=retry)

    def _classify(self, run_id: str, retry_interrupted: bool = False) -> None:
        self._stop.clear()
        extra = {"classify": self.classify} if self.classify else {}

        def task():
            _, runs, results = self.collections()
            try:
                return ab_test.run_ab_test(
                    run_id,
                    runs,
                    results,
                    on_progress=self._relay.progress.emit,
                    should_stop=lambda: self._stop.is_set() or self.closing(),
                    retry_interrupted=retry_interrupted,
                    **extra,
                )
            except ab_test.Stopped as e:
                logger.info(f"A/B run {run_id}: {e}")
                return None

        started = self.run(
            task,
            "Classifying an A/B run",
            lambda summary: self._classified(run_id, summary),
            lambda _: self._classified(run_id, None, failed=True),
        )
        if not started:
            self._busy()
            return
        self.classifying = True
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(True)
        self.progress_label.setText("Starting…")
        self.progress_label.setVisible(True)
        self.stop_button.setEnabled(True)
        self.stop_button.setVisible(True)
        self._update_buttons()

    def _on_progress(self, done: int, total: int) -> None:
        if not self.classifying:  # a late signal after the run ended
            return
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(done)
        text = f"{done} of {total} ticket(s) classified by both"
        if self._stop.is_set():
            text += " · stopping after this ticket…"
        self.progress_label.setText(text)

    def stop(self) -> None:
        self._stop.set()
        self.stop_button.setEnabled(False)
        self.progress_label.setText(
            self.progress_label.text() + " · stopping after this ticket…"
        )

    def _classified(
        self, run_id: str, summary: Optional[dict], failed: bool = False
    ) -> None:
        self.classifying = False
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        self.stop_button.setVisible(False)
        if failed:
            self.status_label.setText(
                "Classification stopped on an error; what finished is saved."
            )
        elif summary is None:
            self.status_label.setText(
                "Stopped. Classify Pending carries on where it left off."
            )
        else:
            self.status_label.setText(
                f"Classified {summary['classified']} of {summary['tickets']} ticket(s) with both."
            )
        self.load_runs_then_open(run_id)

    def load_runs_then_open(self, run_id: str) -> None:
        """Refreshes the run list, then opens run_id for review."""

        def opened(runs: List[dict]) -> None:
            self._show_runs(runs)
            for row, run in enumerate(runs):
                if run["_id"] == run_id:
                    self.run_table.selectRow(row)
                    self.open_selected()
                    return

        task = lambda: runs_with_counts(*self.collections()[1:])
        if not self.run(task, "Loading A/B runs", opened, None):
            self._busy()

    # --- review -----------------------------------------------------------------

    def open_selected(self) -> None:
        run = self._selected_run()
        if run is None:
            return

        # Each result holds the ticket as it was when the run was created:
        # what both variants were sent, not the ticket as it is now.
        def task():
            results = self.collections()[2]
            return list(results.find({"run_id": run["_id"]}).sort("ticket_id", 1))

        if not self.run(
            task, "Loading the run's tickets", lambda d: self._show_run(run, d), None
        ):
            self._busy()

    def _show_run(self, run: dict, docs: List[dict]) -> None:
        self.current, self.docs = run, docs
        unreviewed = [
            i for i, d in enumerate(self.visible()) if not d.get("preference")
        ]
        self.position = unreviewed[0] if unreviewed else 0
        self._show_current()
        if run.get("status") == ab_test.FINALIZED:
            self.status_label.setText(
                "This test is finalized: its verdicts are saved on the tickets and can't change."
            )
        else:
            # Judged earlier but not finalized (say, the app closed first).
            self._finalize_if_ready()

    def _finalize_if_ready(self) -> None:
        """Once every answered ticket has a verdict, saves the verdicts on the
        tickets and removes the test's working data (ab_test.finalize)."""
        if self.current is None or not ab_test.ready_to_finalize(
            self.current, self.docs
        ):
            return
        run_id = self.current["_id"]

        def task():
            tickets, runs, results = self.collections()
            return ab_test.finalize(tickets, runs, results, run_id)

        def done(outcome: dict) -> None:
            missing = outcome["tickets_missing"]
            self.status_label.setText(
                f"Every ticket is judged, so the test is finalized: verdicts saved on "
                f"{outcome['tickets_updated']} ticket(s)"
                + (f", {len(missing)} ticket(s) no longer exist" if missing else "")
                + "; its working data was removed."
            )
            self.load_runs_then_open(run_id)

        if not self.run(task, "Finalizing the A/B test", done, None):
            self._busy()

    def export_selected(self) -> None:
        run = self._selected_run()
        if run is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export A/B Test",
            f"ab-test-{run['_id']}.csv",
            "CSV, one row per ticket (*.csv);;JSON, everything kept (*.json)",
        )
        if not path:
            return

        def task():
            results = self.collections()[2]
            return ab_test.export_run(
                run, list(results.find({"run_id": run["_id"]})), path
            )

        def done(written) -> None:
            self.status_label.setText(f"Exported to {written}.")

        if not self.run(task, "Exporting the A/B test", done, None):
            self._busy()

    def _filter_changed(self) -> None:
        self.position = 0
        self._show_current()

    def _move(self, step: int) -> None:
        self.position = max(0, min(self.position + step, len(self.visible()) - 1))
        self._show_current()

    def _show_current(self) -> None:
        docs = self.visible()
        doc = self._current_doc()
        if self.current is None:
            self.position_label.setText("Open a run to review it.")
        elif doc is None:
            self.position_label.setText(
                f"{self.current.get('name')}: no tickets match the filters."
            )
        else:
            reviewed = sum(1 for d in self.docs if d.get("preference"))
            self.position_label.setText(
                f"{self.current.get('name')}: ticket {self.position + 1} of {len(docs)} "
                f"· {reviewed} of {len(self.docs)} reviewed"
            )
        self.ticket_view.setPlainText(format_ticket(doc) if doc else "")
        for side, view in enumerate(self.output_views, start=1):
            output = (
                (doc.get("outputs") or {}).get(ab_test.blind_choice(doc, side))
                if doc
                else None
            )
            view.setPlainText(format_output(output) if doc else "")
        preference = (doc or {}).get("preference")
        self.verdict_label.setText(self._describe_verdict(doc, preference))
        self.note_input.setText((preference or {}).get("note", ""))
        self._update_buttons()

    @staticmethod
    def _describe_verdict(doc: Optional[dict], preference: Optional[dict]) -> str:
        if not preference:
            return "No verdict yet."
        choice = preference["choice"]
        if choice in ab_test.VARIANTS:
            side = doc["order"].index(choice) + 1
            text = f"Your verdict: Output {side} better"
        else:
            text = f"Your verdict: {ab_test.PREFERENCES[choice]}"
        return text + (
            f" ({preference['reviewer']})" if preference.get("reviewer") else ""
        )

    def record(self, key: str) -> None:
        """Saves the verdict for the ticket shown, then moves to the next one."""
        doc = self._current_doc()
        if doc is None or self.current is None:
            return
        # Map the blind side back to its variant.
        choice = ab_test.blind_choice(doc, int(key)) if key in ("1", "2") else key
        reviewer = self.reviewer
        note = self.note_input.text().strip()
        run_id = self.current["_id"]

        version = doc.get("preference_version") or 0

        def task():
            results = self.collections()[2]
            ab_test.record_preference(
                results, run_id, doc["ticket_id"], choice, version, reviewer, note
            )
            return results.find_one({"run_id": run_id, "ticket_id": doc["ticket_id"]})

        # A conflict (another reviewer saved first) is shown by the tab's error
        # dialog; reloading then shows their verdict.
        if not self.run(
            task,
            "Saving the verdict",
            lambda saved: self._recorded(doc, saved),
            lambda _: self._reload_current(),
        ):
            self._busy()

    def _reload_current(self) -> None:
        """Reloads the run being reviewed, keeping the ticket shown."""
        if self.current is None:
            return
        run, shown = self.current, self._current_doc()

        def reloaded(docs: List[dict]) -> None:
            self.current, self.docs = run, docs
            visible = self.visible()
            ids = [d["_id"] for d in visible]
            self.position = (
                ids.index(shown["_id"]) if shown and shown["_id"] in ids else 0
            )
            self._update_reviewed_count()
            self._show_current()

        task = lambda: list(
            self.collections()[2].find({"run_id": run["_id"]}).sort("ticket_id", 1)
        )
        if not self.run(task, "Reloading the run", reloaded, None):
            self._busy()

    def _update_reviewed_count(self) -> None:
        """Shows the open run's reviewed count in the run table."""
        if self.current is None:
            return
        count = sum(1 for d in self.docs if d.get("preference"))
        column = RUN_COLUMNS.index("Reviewed")
        for row, run in enumerate(self.runs):
            if run["_id"] == self.current["_id"]:
                run["reviewed"] = count
                self.run_table.setItem(row, column, QTableWidgetItem(str(count)))

    def _recorded(self, doc: dict, saved: Optional[dict]) -> None:
        if saved is not None:
            index = next(i for i, d in enumerate(self.docs) if d["_id"] == doc["_id"])
            self.docs[index] = saved
        docs = self.visible()
        if saved in docs:
            # Unfiltered (or still matching): move on to the next ticket.
            self.position = min(docs.index(saved) + 1, max(len(docs) - 1, 0))
        else:
            # It left the "only unreviewed" list; the next one is now at this position.
            self.position = min(self.position, max(len(docs) - 1, 0))
        self.status_label.setText(f"Saved the verdict on ticket {doc['ticket_id']}.")
        self._update_reviewed_count()
        self._show_current()
        self._finalize_if_ready()

    def show_results(self) -> None:
        run = self._selected_run()
        if run is None:
            return

        def task():
            results = self.collections()[2]
            return ab_test.format_summary(
                ab_test.summarize(run, list(results.find({"run_id": run["_id"]})))
            )

        if not self.run(task, "Summarising the run", self._show_summary, None):
            self._busy()

    def _show_summary(self, text: str) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("A/B Results")
        dialog.resize(760, 640)
        view = self._text_view("")
        view.document().setDefaultFont(
            QFontDatabase.systemFont(QFontDatabase.FixedFont)
        )
        view.setPlainText(text)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(dialog.reject)
        layout = QVBoxLayout(dialog)
        layout.addWidget(view)
        layout.addWidget(buttons)
        self.summary_dialog = dialog  # kept so it isn't garbage-collected
        dialog.show()

    def closeEvent(self, event) -> None:
        # The classification thread belongs to the tab; ask it to stop at the
        # next ticket rather than running on with the window closed.
        if self.classifying:
            self._stop.set()
        super().closeEvent(event)
