"""Setup shared by every test."""

import gc

import pytest


@pytest.fixture(autouse=True)
def spend_guard_for_tests(request, tmp_path_factory, monkeypatch):
    """Keeps the PIN, state and audit log in a temporary folder (not the
    test's tmp_path, which some tests list), and lets paid calls
    (all stubbed in tests) through without a PIN.

    Tests marked real_spend_guard get the real checks; live_spend tests (real
    paid calls) get the real checks and the real secrets folder.
    """
    import threading

    import pin_dialog
    import spend_guard

    if request.node.get_closest_marker("live_spend"):
        return

    folder = tmp_path_factory.mktemp("spend_guard")
    monkeypatch.setenv(spend_guard.SECRETS_DIR_ENV, str(folder / "secrets"))
    monkeypatch.setenv(spend_guard.AUDIT_LOG_ENV, str(folder / "audit.log"))
    monkeypatch.delenv(spend_guard.PIN_FILE_ENV, raising=False)
    monkeypatch.setattr(spend_guard, "_local", threading.local())
    monkeypatch.setattr(spend_guard, "_cli_grant", None)
    monkeypatch.setattr(spend_guard, "_interactive_cli", True)
    monkeypatch.setattr(spend_guard, "_memory_state", {})
    if request.node.get_closest_marker("real_spend_guard"):
        return
    monkeypatch.setattr(spend_guard, "require", lambda service, what="": None)
    monkeypatch.setattr(
        pin_dialog,
        "ask_for_grant",
        lambda parent, operation, services: spend_guard.Grant(
            operation, frozenset(services)
        ),
    )


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
