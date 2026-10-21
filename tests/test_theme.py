import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
import theme
from PyQt5.QtWidgets import QApplication, QMessageBox


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def restore_style(qapp):
    yield qapp
    qapp.setStyleSheet("")


@pytest.mark.parametrize(
    "value, expected",
    [
        ("dark", "dark"),
        (" Light ", "light"),
        ("NATIVE", "native"),
        ("", theme.DEFAULT_THEME),
        (None, theme.DEFAULT_THEME),
        ("solarized", None),
    ],
)
def test_theme_name(value, expected):
    assert theme.theme_name(value) == expected


def test_stylesheets_keep_the_log_pane_a_dark_console():
    dark, light, native = (theme.stylesheet(name) for name in theme.THEMES)
    assert dark != light and len(dark) > len(native)
    assert all(f"QTextEdit#{theme.LOG_PANE}" in s for s in (dark, light, native))
    assert native.strip().startswith(f"QTextEdit#{theme.LOG_PANE}")


def test_apply_uses_the_setting(restore_style, monkeypatch):
    monkeypatch.setenv("LASSO_THEME", "light")
    assert theme.apply(restore_style) == "light"
    assert restore_style.styleSheet() == theme.stylesheet("light")

    monkeypatch.delenv("LASSO_THEME")
    assert theme.apply(restore_style) == theme.DEFAULT_THEME


def test_apply_falls_back_on_an_unknown_theme(restore_style):
    assert theme.apply(restore_style, "solarized") == theme.DEFAULT_THEME
    assert restore_style.styleSheet() == theme.stylesheet(theme.DEFAULT_THEME)


def test_settings_dialog_refuses_an_unknown_theme(qapp, monkeypatch):
    from settings_dialog import SettingsDialog

    shown = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *args: shown.append(args[2]))
    dialog = SettingsDialog()
    dialog.fields["LASSO_THEME"].setText("solarized")
    assert not dialog._theme_ok() and "dark, light, native" in shown[0]
    dialog.fields["LASSO_THEME"].setText("Light")
    assert dialog._theme_ok()
