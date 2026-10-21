import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
import theme
import widgets
from PyQt5.QtGui import QPalette
from PyQt5.QtWidgets import QApplication, QGraphicsOpacityEffect


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def restore_style(qapp):
    font, palette = qapp.font(), qapp.palette()
    yield qapp
    qapp.setStyleSheet("")
    qapp.setFont(font)
    qapp.setPalette(palette)


def test_colours_come_from_siliconui():
    assert theme.WINDOW == "#1C191F" and theme.ACCENT == "#855198"


def test_stylesheet_keeps_the_log_pane_a_console():
    sheet = theme.stylesheet()
    assert f"QTextEdit#{theme.LOG_PANE}" in sheet and "monospace" in sheet
    # Exact-class selectors leave SiliconUI's self-painted subclasses alone.
    assert ".QPushButton {" in sheet and "\nQPushButton {" not in sheet


def test_apply_sets_font_palette_and_stylesheet(restore_style):
    theme.apply(restore_style)
    assert restore_style.styleSheet() == theme.stylesheet()
    assert restore_style.font().pixelSize() == theme.FONT_PIXELS
    assert restore_style.palette().color(QPalette.Window).name().lower() == theme.WINDOW.lower()


def test_importing_siui_leaves_scale_factor_alone_off_windows():
    if os.name != "nt":
        assert "QT_SCALE_FACTOR" not in os.environ or os.environ["QT_SCALE_FACTOR"] != "1"


def test_disabled_button_is_faded(qapp):
    button = widgets.Button("Go")
    effect = button.graphicsEffect()
    assert isinstance(effect, QGraphicsOpacityEffect) and effect.opacity() == 1.0
    button.setEnabled(False)
    assert effect.opacity() == widgets.DISABLED_OPACITY
    button.setEnabled(True)
    assert effect.opacity() == 1.0


def test_progress_bar_has_a_busy_mode(qapp):
    bar = widgets.ProgressBar()
    bar.setRange(0, 0)  # SiliconUI alone divides by zero here
    bar.setValue(0)
    assert bar._valueToProgress(bar.value()) == 1.0
    bar.setRange(0, 100)
    assert bar._valueToProgress(50) == 0.5


def test_checkbox_and_combo_keep_the_qt_api(qapp):
    box = widgets.CheckBox("Keep apart")
    seen = []
    box.toggled.connect(seen.append)
    box.setChecked(True)
    assert box.isChecked() and seen == [True]

    combo = widgets.ComboBox("Embeddings")
    combo.addItem("Voyage", "voyage")
    combo.addItem("spaCy", "spacy")
    combo.setCurrentIndex(1)
    assert combo.currentData() == "spacy"


def test_siliconui_helpers_are_parented_to_their_widget(qapp):
    # Parentless, Python could free them mid-deletion of the widget (a crash).
    bar = widgets.ProgressBar()
    assert bar.flash_timer.parent() is bar
    combo = widgets.ComboBox("Embeddings")
    edit = combo.lineEdit()
    assert edit._tooltip_redirect_event_filter.parent() is edit
