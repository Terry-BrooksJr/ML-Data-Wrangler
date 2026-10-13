import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5.QtGui import QColor, QImage
from PyQt5.QtWidgets import QApplication

import results_dialog
from results_dialog import ResultsDialog, format_topics

TOPICS = [(3, 0.6, ["import", "fails"]), (0, 0.4, ["login", "sso"])]


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def plot(tmp_path):
    path = tmp_path / "coherence.png"
    image = QImage(800, 400, QImage.Format_RGB32)
    image.fill(QColor("white"))
    image.save(str(path))
    return str(path)


def test_format_topics_ranks_by_share():
    assert format_topics(TOPICS) == "1. 60% of tickets: import, fails\n2. 40% of tickets: login, sso"


def test_format_topics_empty():
    assert format_topics([]) == "No topics were found."


def test_dialog_shows_topics_and_scaled_plot(qapp, plot, tmp_path):
    dialog = ResultsDialog(2, TOPICS, str(tmp_path / "lda_model.html"), plot)

    assert "60% of tickets" in dialog.topic_text.toPlainText()
    assert dialog.plot_label.pixmap().width() == results_dialog.PLOT_WIDTH
    assert dialog.chart_button.isEnabled()


def test_dialog_handles_missing_outputs(qapp):
    dialog = ResultsDialog(2, TOPICS, None, None)

    assert "could not be created" in dialog.plot_label.text()
    assert not dialog.chart_button.isEnabled()


def test_open_chart_opens_file_uri(qapp, monkeypatch, tmp_path):
    opened = []
    monkeypatch.setattr(results_dialog.webbrowser, "open", opened.append)
    chart = tmp_path / "lda_model.html"

    ResultsDialog(2, TOPICS, str(chart), None).open_chart()

    assert opened == [chart.resolve().as_uri()]


def test_open_chart_without_chart_does_nothing(qapp, monkeypatch):
    opened = []
    monkeypatch.setattr(results_dialog.webbrowser, "open", opened.append)

    ResultsDialog(2, TOPICS, None, None).open_chart()

    assert opened == []
