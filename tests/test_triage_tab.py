import copy
import os
import time
from datetime import datetime, timedelta, timezone

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import mongomock
import pytest
from pymongo.errors import ServerSelectionTimeoutError
from PyQt5.QtWidgets import QApplication, QMessageBox

import triage_tab
from fixtures import CATEGORY_FIXTURES
from triage_tab import TriageTab, count_awaiting, format_review, recent_reviews

NOW = datetime(2026, 10, 13, 12, 0, tzinfo=timezone.utc)


def review(classification, minutes_ago, **extra):
    entry = {
        "review_id": f"r-{classification}-{minutes_ago}",
        "reviewed_at": NOW - timedelta(minutes=minutes_ago),
        "model_version": "test-model",
        "classification": classification,
        "confidence": 80,
        "signals_matched": ["signal a"],
        "reasoning": "because",
        "tag_applied": classification,
        "internal_note": "note",
        "draft_message": None,
        "leadership_recipient": None,
    }
    entry.update(extra)
    return entry


@pytest.fixture
def tickets():
    return mongomock.MongoClient().support_db.tickets


@pytest.fixture
def stored(tickets):
    """Two reviewed tickets (three reviews) and one awaiting review."""
    solved = copy.deepcopy(CATEGORY_FIXTURES["flag_solve"])
    solved["awaiting_review"] = False
    solved["agent_reviews"] = [review("manual_triage", 60), review("flag_solve", 5)]
    reengage = copy.deepcopy(CATEGORY_FIXTURES["flag_reengage"])
    reengage["awaiting_review"] = False
    reengage["agent_reviews"] = [review("flag_reengage", 30, draft_message="Any update?")]
    pending = copy.deepcopy(CATEGORY_FIXTURES["escalation_internal"])
    tickets.insert_many([solved, reengage, pending])
    return tickets


# --- queries ---------------------------------------------------------------


def test_count_awaiting(stored):
    assert count_awaiting(stored) == 1


def test_recent_reviews_flattens_newest_first(stored):
    rows = recent_reviews(stored)
    assert [row["review"]["classification"] for row in rows] == ["flag_solve", "flag_reengage", "manual_triage"]
    assert {"zendesk_id", "subject", "review"} <= set(rows[0])


def test_recent_reviews_respects_limit(stored):
    assert len(recent_reviews(stored, limit=2)) == 2


def test_format_review_includes_draft_only_when_present(stored):
    rows = {row["review"]["classification"]: row for row in recent_reviews(stored)}
    assert "Draft message (unsent):\nAny update?" in format_review(rows["flag_reengage"])
    assert "Draft message" not in format_review(rows["flag_solve"])


# --- widget ----------------------------------------------------------------


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setenv("MONGO_URI", "mongodb://test")
    monkeypatch.setenv("CHEAPER_INFERENCE_API_KEY", "test")


@pytest.fixture
def dialogs(monkeypatch):
    """Records message boxes instead of showing them; confirms by default."""
    shown = []

    def record(kind, answer=None):
        def box(*args, **kwargs):
            shown.append((kind, args[2] if len(args) > 2 else ""))
            return answer

        return box

    monkeypatch.setattr(QMessageBox, "information", record("information"))
    monkeypatch.setattr(QMessageBox, "warning", record("warning"))
    monkeypatch.setattr(QMessageBox, "critical", record("critical"))
    monkeypatch.setattr(QMessageBox, "question", record("question", QMessageBox.Yes))
    return shown


def wait_until_idle(qapp, tab, timeout=5.0):
    deadline = time.monotonic() + timeout
    while tab._thread is not None or not tab.refresh_button.isEnabled():
        qapp.processEvents()
        assert time.monotonic() < deadline, "background job did not finish"
        time.sleep(0.01)
    qapp.processEvents()


def make_tab(tickets, **engine):
    return TriageTab(collection_factory=lambda: tickets, **engine)


def test_refresh_fills_status_and_table(qapp, configured, stored):
    tab = make_tab(stored)
    tab.refresh()
    wait_until_idle(qapp, tab)

    assert tab.status_label.text() == "1 ticket(s) awaiting review."
    assert tab.table.rowCount() == 3
    classification_column = triage_tab.COLUMNS.index("Classification")
    shown = [tab.table.item(row, classification_column).text() for row in range(3)]
    assert shown == ["flag_solve", "flag_reengage", "manual_triage"]


def test_selecting_a_row_shows_its_detail(qapp, configured, stored):
    tab = make_tab(stored)
    tab.refresh()
    wait_until_idle(qapp, tab)

    tab.table.selectRow(0)
    assert "Classification:" in tab.detail.toPlainText()


def test_missing_env_blocks_actions(qapp, monkeypatch, dialogs, stored):
    monkeypatch.delenv("MONGO_URI", raising=False)
    monkeypatch.setenv("CHEAPER_INFERENCE_API_KEY", "test")
    calls = []
    tab = make_tab(stored, review_batch=lambda tickets: calls.append(tickets))

    tab.start_review_all()

    assert calls == []
    assert dialogs[0][0] == "warning"
    assert "MONGO_URI" in tab.status_label.text()


def test_review_all_confirms_then_runs_batch(qapp, configured, dialogs, stored):
    calls = []
    tab = make_tab(stored, review_batch=lambda tickets: calls.append(tickets))

    tab.start_review_all()
    wait_until_idle(qapp, tab)

    assert [kind for kind, _ in dialogs][:2] == ["question", "information"]
    assert "Send 1 ticket(s)" in dialogs[0][1]
    assert calls == [stored]


def test_review_all_cancel_does_not_run(qapp, configured, dialogs, monkeypatch, stored):
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Cancel)
    calls = []
    tab = make_tab(stored, review_batch=lambda tickets: calls.append(tickets))

    tab.start_review_all()
    wait_until_idle(qapp, tab)

    assert calls == []


def test_review_all_with_nothing_awaiting_skips_confirm(qapp, configured, dialogs, tickets):
    calls = []
    tab = make_tab(tickets, review_batch=lambda t: calls.append(t))

    tab.start_review_all()
    wait_until_idle(qapp, tab)

    assert calls == []
    assert dialogs == [("information", "No tickets are awaiting review.")]


def test_review_one_passes_trimmed_id(qapp, configured, dialogs, stored):
    calls = []

    def review_one(zendesk_id, tickets):
        calls.append(zendesk_id)
        return {"classification": "flag_solve", "confidence": 90}

    tab = make_tab(stored, review_one=review_one)
    tab.ticket_id_input.setText("  123  ")
    tab.start_review_one()
    wait_until_idle(qapp, tab)

    assert calls == ["123"]
    assert tab.ticket_id_input.text() == ""


def test_review_one_reports_not_awaiting(qapp, configured, dialogs, stored):
    tab = make_tab(stored, review_one=lambda zendesk_id, tickets: None)
    tab.ticket_id_input.setText("999")
    tab.start_review_one()
    wait_until_idle(qapp, tab)

    assert dialogs == [("information", "Ticket 999 was not found, or is not awaiting review.")]


def test_engine_error_is_shown_and_tab_recovers(qapp, configured, dialogs, stored):
    def failing_batch(tickets):
        raise KeyError("confidence")

    tab = make_tab(stored, review_batch=failing_batch)
    tab.start_review_all()
    wait_until_idle(qapp, tab)

    assert dialogs[-1][0] == "critical"
    assert tab.review_all_button.isEnabled()


def test_unreachable_mongo_shows_connection_hint(qapp, configured, dialogs):
    def unreachable():
        raise ServerSelectionTimeoutError("timed out")

    tab = TriageTab(collection_factory=unreachable)
    tab.refresh()
    wait_until_idle(qapp, tab)

    assert dialogs[-1][0] == "critical"
    assert "Atlas IP access list" in dialogs[-1][1]
