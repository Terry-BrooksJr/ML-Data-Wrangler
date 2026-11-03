"""ui.scaling: the Windows scale factor, set before app.py's early QApplication."""

import ctypes
import os
from types import SimpleNamespace

import pytest

from ui import scaling


@pytest.fixture
def windows(monkeypatch):
    """Runs as Windows at 144 DPI (150%), with no QT_SCALE_FACTOR set."""
    calls = []
    user32 = SimpleNamespace(
        SetProcessDPIAware=lambda: calls.append("aware"),
        GetDpiForSystem=lambda: 144,
    )
    monkeypatch.setattr(scaling.sys, "platform", "win32")
    monkeypatch.setattr(ctypes, "windll", SimpleNamespace(user32=user32), raising=False)
    monkeypatch.delenv("QT_SCALE_FACTOR", raising=False)
    return calls


def test_sets_the_factor_from_the_system_dpi(windows):
    scaling.use_windows_scale_factor()
    assert os.environ["QT_SCALE_FACTOR"] == "1.5"
    assert windows == ["aware"]


def test_keeps_the_users_factor(windows, monkeypatch):
    monkeypatch.setenv("QT_SCALE_FACTOR", "2")
    scaling.use_windows_scale_factor()
    assert os.environ["QT_SCALE_FACTOR"] == "2"
    assert windows == []


def test_falls_back_to_1_without_the_dpi_api(windows, monkeypatch):
    monkeypatch.setattr(ctypes, "windll", SimpleNamespace(user32=SimpleNamespace()))
    scaling.use_windows_scale_factor()
    assert os.environ["QT_SCALE_FACTOR"] == "1"


def test_does_nothing_off_windows(monkeypatch):
    monkeypatch.setattr(scaling.sys, "platform", "darwin")
    monkeypatch.delenv("QT_SCALE_FACTOR", raising=False)
    scaling.use_windows_scale_factor()
    assert "QT_SCALE_FACTOR" not in os.environ
