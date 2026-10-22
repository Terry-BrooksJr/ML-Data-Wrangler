"""Windows for BERTopic runs: the training options and a run's results.

The text and numbers come from bertopic_runs; these classes only lay them
out. The barchart and intertopic charts are Plotly HTML and open in the
browser (they load Plotly from its CDN).
"""

import pathlib
import webbrowser
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import bertopic_model
import bertopic_runs
import topic_store
from loguru import logger
from PyQt5.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHeaderView,
    QMessageBox,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from widgets import button_row, Button, CheckBox, ComboBox, heading, hint, use_si_scrollbars


class BertopicOptionsDialog(QDialog):
    """Asks how to train BERTopic; options() returns the choices once accepted.

    Args:
        lda_runs: Saved LDA runs the new run can be compared with.
        outcome_counts: (outcome, tickets) from bertopic_model.outcome_counts,
            None for unset; one checkbox each. None hides the choice.
    """

    def __init__(
        self,
        lda_runs: List[dict],
        outcome_counts: Optional[Sequence[Tuple[Optional[str], int]]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Train BERTopic Model")

        self.setMinimumWidth(620)
        intro = hint(
            "Groups tickets by meaning, using the embeddings already stored on each ticket. "
            "Only tickets whose embedding matches their current text are used."
        )

        # 0 is shown as the special text and means bertopic_model's default.
        self.min_cluster_size = QSpinBox()
        self.min_cluster_size.setRange(0, 100_000)
        self.min_cluster_size.setSpecialValueText(
            f"Automatic (1% of tickets, at least {bertopic_model.MIN_CLUSTER_SIZE_FLOOR})"
        )
        self.nr_topics = QSpinBox()
        self.nr_topics.setRange(0, 1_000)
        self.nr_topics.setSpecialValueText("Keep every topic found")

        self.keep_outliers = CheckBox(
            "Keep outliers as topic -1 (don't move them to the nearest topic)"
        )

        self.compare_lda = ComboBox("Compare with LDA run")
        self.compare_lda.addItem("None", None)
        for run in lda_runs:
            self.compare_lda.addItem(topic_store.describe_run(run), run["run_id"])

        self.assign = CheckBox(
            f"Write each ticket's topic to its `{bertopic_model.ASSIGNMENT_FIELD}` field in MongoDB"
        )
        self.assign.setToolTip("The LDA `topic` field is left alone.")

        # Every outcome ticked trains on every ticket, as before; untick some to
        # find the themes inside the rest (e.g. only client_education).
        self.outcome_boxes: List[Tuple[Optional[str], CheckBox]] = []
        for outcome, count in outcome_counts or []:
            box = CheckBox(f"{outcome or 'No outcome set'} ({count})")
            box.setChecked(True)
            box.toggled.connect(self._outcomes_changed)
            self.outcome_boxes.append((outcome, box))

        form = QFormLayout()
        form.addRow("Minimum topic size:", self.min_cluster_size)
        form.addRow("Number of topics:", self.nr_topics)
        form.setSpacing(10)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(12)
        layout.addWidget(intro)
        layout.addLayout(form)
        layout.addWidget(self.compare_lda)
        layout.addWidget(self.keep_outliers)
        layout.addWidget(self.assign)
        if self.outcome_boxes:
            layout.addWidget(heading("Tickets by Zendesk outcome"))
            layout.addWidget(
                hint(
                    "Untick outcomes to leave their tickets out, e.g. keep only "
                    "client_education to see the themes inside it."
                )
            )
            for _, box in self.outcome_boxes:
                layout.addWidget(box)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.button(QDialogButtonBox.Ok).setText("Train")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

    def _outcomes_changed(self) -> None:
        # Training on no tickets can't work.
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(
            any(box.isChecked() for _, box in self.outcome_boxes)
        )

    def selected_outcomes(self) -> Optional[List[Optional[str]]]:
        """The ticked outcomes, or None when all are (every ticket, no filter)."""
        chosen = [o for o, box in self.outcome_boxes if box.isChecked()]
        return None if len(chosen) == len(self.outcome_boxes) else chosen

    def options(self) -> bertopic_runs.TrainOptions:
        return bertopic_runs.TrainOptions(
            min_cluster_size=self.min_cluster_size.value() or None,
            nr_topics=self.nr_topics.value() or None,
            keep_outliers=self.keep_outliers.isChecked(),
            compare_lda=self.compare_lda.currentData(),
            assign=self.assign.isChecked(),
            outcomes=self.selected_outcomes(),
        )


class BertopicResultsDialog(QDialog):
    """Shows a BERTopic run's topics, its c_v and diversity beside any LDA run it was compared with,
    and buttons for its charts.

    Args:
        run: A run from bertopic_runs.list_runs() or train_and_save().
        charts: bertopic_runs.chart_paths(run).
        subjects: Looks up the run's example ticket subjects for Export Topic
            List (bertopic_runs.example_subjects); None exports without them.
    """

    def __init__(
        self,
        run: dict,
        charts: dict,
        parent: Optional[QWidget] = None,
        subjects: Optional[Callable[[dict], Dict[str, str]]] = None,
    ) -> None:
        super().__init__(parent)
        self.run = run
        self.charts = charts
        self.subjects = subjects
        self.setWindowTitle(f"BERTopic Results · {bertopic_runs.describe_run(run)}")
        self.resize(760, 700)

        summary = hint(bertopic_runs.summary_text(run))

        rows = bertopic_runs.comparison_rows(run)
        self.comparison = use_si_scrollbars(QTableWidget(len(rows), 4))
        self.comparison.setHorizontalHeaderLabels(
            ["Model", "Topics", "c_v coherence", "Topic diversity"]
        )
        for r, row in enumerate(rows):
            for c, value in enumerate(row):
                self.comparison.setItem(r, c, QTableWidgetItem(value))
        self.comparison.setEditTriggers(QTableWidget.NoEditTriggers)
        self.comparison.verticalHeader().hide()
        self.comparison.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.comparison.setFixedHeight(
            self.comparison.horizontalHeader().height()
            + sum(self.comparison.rowHeight(r) for r in range(len(rows)))
            + 4
        )
        note = hint(
            "Both scored on this run's tickets and tokens."
            if run.get("comparison")
            else "Pick an LDA run when training to compare with it."
        )

        # Plain text: topic words come from customer tickets.
        self.topic_text = use_si_scrollbars(QTextEdit())
        self.topic_text.setReadOnly(True)
        self.topic_text.setPlainText(bertopic_runs.format_topics(run))

        self.chart_buttons = {}
        for name, label in (
            ("barchart_html", "Open Topic Word Chart"),
            ("intertopic_html", "Open Intertopic Distance Map"),
        ):
            button = Button(label)
            button.clicked.connect(lambda _, name=name: self.open_chart(name))
            if not charts.get(name):
                button.setEnabled(False)
                button.setText(
                    f"{label.removeprefix('Open ')} could not be created; see the log"
                )
            self.chart_buttons[name] = button

        self.export_button = Button("Export Topic List…")
        self.export_button.setToolTip(
            "A CSV of the topics, with each one's outcomes, words and example tickets, "
            "and blank columns to name and keep, merge or drop it."
        )
        self.export_button.clicked.connect(self.export_topic_list)
        self.export_button.setEnabled(bool(run.get("topics")))

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.close)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(10)
        layout.addWidget(summary)
        layout.addWidget(self.comparison)
        layout.addWidget(note)
        layout.addWidget(heading("Topics, largest first"))
        layout.addWidget(self.topic_text, stretch=1)
        layout.addLayout(button_row(*self.chart_buttons.values(), self.export_button))
        layout.addWidget(buttons)

    def open_chart(self, name: str) -> None:
        if self.charts.get(name):
            webbrowser.open(pathlib.Path(self.charts[name]).resolve().as_uri())

    def export_topic_list(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Topic List",
            f"topic_list_{self.run['run_id']}.csv",
            "CSV (*.csv)",
        )
        if not path:
            return
        subjects: Dict[str, str] = {}
        note = ""
        if self.subjects is not None:
            try:
                subjects = self.subjects(self.run)
            except Exception as e:  # Mongo unreachable: export the rest anyway
                logger.warning(f"Could not read example subjects from MongoDB: {e}")
                note = "\n\nExample subjects are missing: MongoDB could not be reached."
        try:
            with open(path, "w", encoding="utf-8", newline="") as f:
                f.write(bertopic_runs.topic_list_csv(self.run, subjects))
        except OSError as e:
            QMessageBox.critical(self, "Could not save", f"{path}: {e}")
            return
        logger.info(
            f"BERTopic: wrote the topic list for run {self.run['run_id']} to {path}"
        )
        QMessageBox.information(
            self,
            "Topic List Exported",
            f"Wrote {len(self.run.get('topics') or [])} topic(s) to\n{path}\n\n"
            "Fill in proposed_name and decision (keep, merge into N, drop)." + note,
        )
