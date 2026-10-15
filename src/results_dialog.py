"""Window showing the results of an LDA training run.

Replaces the Gradio page the app used to start: the topics and the coherence
plot are shown in the app itself, and the interactive pyLDAvis chart opens in
the browser, sandboxed (chart_viewer).
"""

import pathlib
from typing import List, Optional, Tuple

from loguru import logger
from PyQt5 import QtCore
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

import chart_viewer

PLOT_WIDTH = 560


def format_topics(topics: List[Tuple[int, float, List[str]]]) -> str:
    """One numbered line per topic, most prevalent first."""
    if not topics:
        return "No topics were found."
    return "\n".join(
        f"{rank}. {share:.0%} of tickets: {', '.join(words)}"
        for rank, (_, share, words) in enumerate(topics, start=1)
    )


class ResultsDialog(QDialog):
    """Shows the best model's topics and the coherence-by-topic-count plot.

    Args:
        best_num_topics: Topic count of the most coherent model.
        topics: (topic id, share of tickets, top words), most prevalent first,
            as returned by LatentDirichletAllocator.get_topics().
        chart_path: The saved pyLDAvis HTML chart, or None if it failed.
        plot_path: The saved coherence PNG, or None if it failed.
    """

    def __init__(
        self,
        best_num_topics: int,
        topics: List[Tuple[int, float, List[str]]],
        chart_path: Optional[str],
        plot_path: Optional[str],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.chart_path = chart_path
        self.setWindowTitle("LDA Results")
        self.resize(700, 750)

        heading = QLabel(f"<b>Topics ({best_num_topics}, the most coherent model)</b>")

        # Plain text: topic words come from customer tickets.
        self.topic_text = QTextEdit()
        self.topic_text.setReadOnly(True)
        self.topic_text.setPlainText(format_topics(topics))

        self.plot_label = QLabel()
        self.plot_label.setAlignment(QtCore.Qt.AlignCenter)
        pixmap = QPixmap(plot_path) if plot_path else QPixmap()
        if pixmap.isNull():
            self.plot_label.setText("The coherence plot could not be created; see the log.")
        else:
            self.plot_label.setPixmap(
                pixmap.scaledToWidth(min(PLOT_WIDTH, pixmap.width()), QtCore.Qt.SmoothTransformation)
            )

        self.chart_button = QPushButton("Open Interactive Topic Chart")
        self.chart_button.clicked.connect(self.open_chart)
        if not chart_path:
            self.chart_button.setEnabled(False)
            self.chart_button.setText("Interactive topic chart could not be created; see the log")

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.close)

        layout = QVBoxLayout(self)
        layout.addWidget(heading)
        layout.addWidget(self.topic_text, stretch=1)
        layout.addWidget(QLabel("Coherence by number of topics:"))
        layout.addWidget(self.plot_label)
        layout.addWidget(self.chart_button)
        layout.addWidget(buttons)

    def open_chart(self) -> None:
        """Opens the interactive chart in the browser, sandboxed (see chart_viewer).

        Never as a file:// page: a chart is HTML with script, and one from a
        downloaded run may not be safe.
        """
        if not self.chart_path:
            return
        try:
            chart_viewer.open_chart(self.chart_path)
        except OSError as e:
            logger.error(f"Could not open the topic chart: {e}")
            QMessageBox.warning(self, "Chart Not Opened", f"Could not open the topic chart:\n{e}")
