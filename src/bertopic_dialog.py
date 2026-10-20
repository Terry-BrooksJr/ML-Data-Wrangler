"""Windows for BERTopic runs: the training options and a run's results.

The text and numbers come from bertopic_runs; these classes only lay them
out. The barchart and intertopic charts are Plotly HTML and open in the
browser (they load Plotly from its CDN).
"""

import pathlib
import webbrowser
from typing import List, Optional

import bertopic_model
import bertopic_runs
import topic_store
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class BertopicOptionsDialog(QDialog):
    """Asks how to train BERTopic; options() returns the choices once accepted.

    Args:
        lda_runs: Saved LDA runs the new run can be compared with.
    """

    def __init__(self, lda_runs: List[dict], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Train BERTopic Model")

        intro = QLabel(
            "Groups tickets by meaning, using the embeddings already stored on each ticket. "
            "Only tickets whose embedding matches their current text are used."
        )
        intro.setWordWrap(True)

        # 0 is shown as the special text and means bertopic_model's default.
        self.min_cluster_size = QSpinBox()
        self.min_cluster_size.setRange(0, 100_000)
        self.min_cluster_size.setSpecialValueText(
            f"Automatic (1% of tickets, at least {bertopic_model.MIN_CLUSTER_SIZE_FLOOR})"
        )
        self.nr_topics = QSpinBox()
        self.nr_topics.setRange(0, 1_000)
        self.nr_topics.setSpecialValueText("Keep every topic found")

        self.keep_outliers = QCheckBox(
            "Keep outliers as topic -1 (don't move them to the nearest topic)"
        )

        self.compare_lda = QComboBox()
        self.compare_lda.addItem("None", None)
        for run in lda_runs:
            self.compare_lda.addItem(topic_store.describe_run(run), run["run_id"])

        self.assign = QCheckBox(
            f"Write each ticket's topic to its `{bertopic_model.ASSIGNMENT_FIELD}` field in MongoDB "
            "(the LDA `topic` field is left alone)"
        )

        form = QFormLayout()
        form.addRow("Minimum topic size:", self.min_cluster_size)
        form.addRow("Number of topics:", self.nr_topics)
        form.addRow(self.keep_outliers)
        form.addRow("Compare with LDA run:", self.compare_lda)
        form.addRow(self.assign)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Train")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(intro)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def options(self) -> bertopic_runs.TrainOptions:
        return bertopic_runs.TrainOptions(
            min_cluster_size=self.min_cluster_size.value() or None,
            nr_topics=self.nr_topics.value() or None,
            keep_outliers=self.keep_outliers.isChecked(),
            compare_lda=self.compare_lda.currentData(),
            assign=self.assign.isChecked(),
        )


class BertopicResultsDialog(QDialog):
    """Shows a BERTopic run's topics, its c_v and diversity beside any LDA run it was compared with,
    and buttons for its charts.

    Args:
        run: A run from bertopic_runs.list_runs() or train_and_save().
        charts: bertopic_runs.chart_paths(run).
    """

    def __init__(
        self, run: dict, charts: dict, parent: Optional[QWidget] = None
    ) -> None:
        super().__init__(parent)
        self.charts = charts
        self.setWindowTitle(f"BERTopic Results · {bertopic_runs.describe_run(run)}")
        self.resize(760, 700)

        summary = QLabel(bertopic_runs.summary_text(run))
        summary.setWordWrap(True)

        rows = bertopic_runs.comparison_rows(run)
        self.comparison = QTableWidget(len(rows), 4)
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
        note = QLabel(
            "Both scored on this run's tickets and tokens."
            if run.get("comparison")
            else "Pick an LDA run when training to compare with it."
        )

        # Plain text: topic words come from customer tickets.
        self.topic_text = QTextEdit()
        self.topic_text.setReadOnly(True)
        self.topic_text.setPlainText(bertopic_runs.format_topics(run))

        self.chart_buttons = {}
        for name, label in (
            ("barchart_html", "Open Topic Word Chart"),
            ("intertopic_html", "Open Intertopic Distance Map"),
        ):
            button = QPushButton(label)
            button.clicked.connect(lambda _, name=name: self.open_chart(name))
            if not charts.get(name):
                button.setEnabled(False)
                button.setText(
                    f"{label.removeprefix('Open ')} could not be created; see the log"
                )
            self.chart_buttons[name] = button

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.close)

        layout = QVBoxLayout(self)
        layout.addWidget(summary)
        layout.addWidget(self.comparison)
        layout.addWidget(note)
        layout.addWidget(QLabel("<b>Topics, largest first</b>"))
        layout.addWidget(self.topic_text, stretch=1)
        for button in self.chart_buttons.values():
            layout.addWidget(button)
        layout.addWidget(buttons)

    def open_chart(self, name: str) -> None:
        if self.charts.get(name):
            webbrowser.open(pathlib.Path(self.charts[name]).resolve().as_uri())
