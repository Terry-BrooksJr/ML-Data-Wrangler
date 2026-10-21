import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
import results_dialog
from PyQt5.QtGui import QColor, QImage
from PyQt5.QtWidgets import QApplication
from results_dialog import format_topics, ResultsDialog

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
    assert (
        format_topics(TOPICS)
        == "1. 60% of tickets: import, fails\n2. 40% of tickets: login, sso"
    )


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


def test_open_chart_uses_the_sandboxed_viewer_not_a_file_uri(
    qapp, monkeypatch, tmp_path
):
    opened = []
    monkeypatch.setattr(results_dialog.chart_viewer.webbrowser, "open", opened.append)
    chart = tmp_path / "lda_model.html"
    chart.write_text("<html></html>")

    ResultsDialog(2, TOPICS, str(chart), None).open_chart()

    [url] = opened
    assert url.startswith("http://127.0.0.1:") and url.endswith("/chart.html")
    assert not url.startswith("file:")


def test_open_chart_without_chart_does_nothing(qapp, monkeypatch):
    opened = []
    monkeypatch.setattr(results_dialog.chart_viewer.webbrowser, "open", opened.append)

    ResultsDialog(2, TOPICS, None, None).open_chart()

    assert opened == []


def test_unreadable_chart_warns_instead_of_crashing(qapp, monkeypatch, tmp_path):
    from PyQt5.QtWidgets import QMessageBox

    warned = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warned.append(a[2]))
    ResultsDialog(2, TOPICS, str(tmp_path / "gone.html"), None).open_chart()
    assert warned and "Could not open the topic chart" in warned[0]
