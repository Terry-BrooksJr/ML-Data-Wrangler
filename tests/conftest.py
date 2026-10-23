"""Setup shared by every test."""

import gc

import pytest


@pytest.fixture(autouse=True)
def close_windows_left_open():
    """Closes and deletes the windows a test opened and left open.

    SiliconUI's scroll bars and fades run on timers. A window left over from one
    test kept repainting during later ones, and once Python had freed part of it,
    a scroll bar's paintEvent crashed the whole run (a segfault in
    test_triage_tab, roughly every other run). Windows that existed before the
    test, such as app.py's main window, are left alone.
    """
    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance()
    before = set(app.topLevelWidgets()) if app else set()
    yield
    app = QApplication.instance()
    if app is None:
        return
    from PyQt5.QtCore import QCoreApplication, QEvent

    for widget in app.topLevelWidgets():
        if widget not in before:
            widget.close()
            widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    app.processEvents()
    gc.collect()
