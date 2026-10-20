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
    for name in (*triage_tab.REVIEW_ENV, *triage_tab.OPTIONAL_MODEL_ENV):
        monkeypatch.setenv(name, "test")


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


def test_refresh_shows_no_dialog(qapp, configured, dialogs, stored):
    tab = make_tab(stored)
    tab.refresh()
    wait_until_idle(qapp, tab)

    assert dialogs == []


def test_refresh_needs_only_mongo(qapp, monkeypatch, dialogs, stored):
    for name in triage_tab.REVIEW_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MONGO_URI", "mongodb://test")
    tab = make_tab(stored)
    tab.refresh()
    wait_until_idle(qapp, tab)

    assert tab.table.rowCount() == 3
    assert dialogs == []


def test_review_needs_model_settings(qapp, configured, monkeypatch, dialogs, stored):
    monkeypatch.delenv("CHEAPER_INFERENCE_PRIMARY_MODEL_NAME")
    calls = []
    tab = make_tab(stored, review_batch=lambda tickets, ids: calls.append(tickets))

    tab.start_review_all()

    assert calls == []
    assert dialogs[0][0] == "warning"
    assert "CHEAPER_INFERENCE_PRIMARY_MODEL_NAME" in tab.status_label.text()


@pytest.mark.parametrize(
    "optional", ["CHEAPER_INFERENCE_SECONDARY_MODEL_NAME", "CHEAPER_INFERENCE_CONFIRMATION_MODEL_NAME"]
)
def test_review_runs_without_optional_models(qapp, configured, monkeypatch, dialogs, stored, optional):
    monkeypatch.delenv(optional, raising=False)
    calls = []
    tab = make_tab(stored, review_batch=lambda tickets, ids: calls.append(tickets))

    tab.start_review_all()
    wait_until_idle(qapp, tab)

    assert calls == [stored]
    assert f"{optional} is not set" in dialogs[0][1]


def test_no_optional_warning_when_all_set(qapp, configured, dialogs, stored):
    tab = make_tab(stored, review_batch=lambda tickets, ids: None)
    tab.start_review_all()
    wait_until_idle(qapp, tab)

    assert "Warning" not in dialogs[0][1]


def test_review_one_warns_about_unset_confirmation(qapp, configured, monkeypatch, dialogs, stored):
    monkeypatch.delenv("CHEAPER_INFERENCE_CONFIRMATION_MODEL_NAME")
    tab = make_tab(stored, review_one=lambda zendesk_id, tickets: {"classification": "manual_triage", "confidence": 40})
    tab.ticket_id_input.setText("123")
    tab.start_review_one()
    wait_until_idle(qapp, tab)

    assert "CHEAPER_INFERENCE_CONFIRMATION_MODEL_NAME is not set" in dialogs[0][1]


def test_missing_env_blocks_actions(qapp, configured, monkeypatch, dialogs, stored):
    monkeypatch.delenv("MONGO_URI")
    calls = []
    tab = make_tab(stored, review_batch=lambda tickets, ids: calls.append(tickets))

    tab.start_review_all()

    assert calls == []
    assert dialogs[0][0] == "warning"
    assert "MONGO_URI" in tab.status_label.text()


def test_review_all_confirms_then_runs_batch(qapp, configured, dialogs, stored):
    calls = []
    tab = make_tab(stored, review_batch=lambda tickets, ids: calls.append(tickets))

    tab.start_review_all()
    wait_until_idle(qapp, tab)

    # Exactly one confirm and one completion: the count and refresh are silent.
    assert dialogs == [
        ("question", dialogs[0][1]),
        ("information", "Review batch finished. Results are listed below."),
    ]
    assert "Send 1 ticket(s)" in dialogs[0][1]
    assert calls == [stored]


def test_review_all_cancel_does_not_run(qapp, configured, dialogs, monkeypatch, stored):
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Cancel)
    calls = []
    tab = make_tab(stored, review_batch=lambda tickets, ids: calls.append(tickets))

    tab.start_review_all()
    wait_until_idle(qapp, tab)

    assert calls == []


def test_review_all_with_nothing_awaiting_skips_confirm(qapp, configured, dialogs, tickets):
    calls = []
    tab = make_tab(tickets, review_batch=lambda t, ids: calls.append(t))

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
    assert dialogs == [("information", "Ticket 123: flag_solve (confidence 90)")]


def test_review_one_reports_not_awaiting(qapp, configured, dialogs, stored):
    tab = make_tab(stored, review_one=lambda zendesk_id, tickets: None)
    tab.ticket_id_input.setText("999")
    tab.start_review_one()
    wait_until_idle(qapp, tab)

    assert dialogs == [("information", "Ticket 999 was not found, or is not awaiting review.")]


def test_engine_error_is_shown_and_tab_recovers(qapp, configured, dialogs, stored):
    def failing_batch(tickets, ids):
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


# --- batch summary, engine errors and feedback --------------------------------


def test_review_all_shows_batch_summary(qapp, configured, dialogs, stored):
    summary = {"classified": 1, "manual_fallback": 0, "skipped": 0, "errored": 2, "stopped": None, "usage": {}}
    tab = make_tab(stored, review_batch=lambda tickets, ids: summary)
    tab.start_review_all()
    wait_until_idle(qapp, tab)

    assert dialogs[-1][0] == "information"
    assert "Engine errors (left in the queue to retry): 2" in dialogs[-1][1]


def test_review_all_warns_when_batch_stopped(qapp, configured, dialogs, stored):
    summary = {"classified": 0, "manual_fallback": 0, "skipped": 0, "errored": 0,
               "stopped": "AuthenticationError: bad key", "usage": {}}
    tab = make_tab(stored, review_batch=lambda tickets, ids: summary)
    tab.start_review_all()
    wait_until_idle(qapp, tab)

    assert dialogs[-1][0] == "warning"
    assert "Stopped early" in dialogs[-1][1]


def test_format_review_shows_overrides_engine_error_and_audit(stored):
    row = {
        "zendesk_id": "1",
        "subject": "s",
        "outcomes": [{"review_id": "r1", "outcome": "bad", "detection_method": "manual", "detail": "wrong"}],
        "review": review(
            "manual_triage", 1, review_id="r1",
            guard_overrides=["Downgraded from flag_solve"], engine_error="cut off",
            audit={"usage": {"input_tokens": 10, "output_tokens": 5}, "latency_ms": 12,
                   "request_id": "req_1", "prompt_version": "abc"},
        ),
    }
    text = format_review(row)
    assert "Overridden by code checks:\n  - Downgraded from flag_solve" in text
    assert "Engine error" in text
    assert "Outcome: bad (manual): wrong" in text
    assert "10 in / 5 out tokens" in text and "request req_1" in text


def test_feedback_buttons_need_a_selection(qapp, configured, stored):
    tab = make_tab(stored)
    tab.refresh()
    wait_until_idle(qapp, tab)
    assert not tab.mark_good_button.isEnabled()

    tab.table.selectRow(0)
    assert tab.mark_good_button.isEnabled() and tab.mark_bad_button.isEnabled()


def test_mark_correct_records_outcome(qapp, configured, dialogs, tickets, monkeypatch):
    from bson import ObjectId

    ticket = copy.deepcopy(CATEGORY_FIXTURES["flag_solve"])
    ticket["awaiting_review"] = False
    ticket["agent_reviews"] = [review("flag_solve", 5, review_id=ObjectId())]
    ticket["outcomes"] = []
    tickets.insert_one(ticket)
    tab = make_tab(tickets)
    tab.refresh()
    wait_until_idle(qapp, tab)

    tab.table.selectRow(0)
    tab.mark_selected_correct()
    wait_until_idle(qapp, tab)

    outcomes = tickets.find_one({"zendesk_id": ticket["zendesk_id"]})["outcomes"]
    assert [o["outcome"] for o in outcomes] == ["good"]


def test_mark_misclassified_logs_entry(qapp, configured, dialogs, tickets, monkeypatch):
    from bson import ObjectId
    from PyQt5.QtWidgets import QInputDialog

    ticket = copy.deepcopy(CATEGORY_FIXTURES["flag_solve"])
    ticket["awaiting_review"] = False
    ticket["agent_reviews"] = [review("flag_solve", 5, review_id=ObjectId())]
    tickets.insert_one(ticket)
    monkeypatch.setattr(QInputDialog, "getItem", lambda *a, **k: ("flag_reengage", True))
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("customer never confirmed", True))
    tab = make_tab(tickets)
    tab.refresh()
    wait_until_idle(qapp, tab)

    tab.table.selectRow(0)
    tab.mark_selected_misclassified()
    wait_until_idle(qapp, tab)

    entry = tickets.database.misclassification_log.find_one()
    assert entry["correct_classification"] == "flag_reengage"
    assert entry["misread_signal"] == "customer never confirmed"


def test_accuracy_report_dialog(qapp, configured, dialogs, stored):
    tab = make_tab(stored)
    tab.show_accuracy_report()
    wait_until_idle(qapp, tab)
    assert dialogs[-1][0] == "information" and "flag_solve" in dialogs[-1][1]


# --- confirmed snapshot (paid calls only for approved tickets) -------------------


def test_review_all_reviews_only_the_confirmed_snapshot(qapp, configured, stored, monkeypatch):
    late = copy.deepcopy(CATEGORY_FIXTURES["manual_triage"])

    def confirm_while_a_ticket_arrives(*args, **kwargs):
        stored.insert_one(late)  # e.g. the Mongo writer, while the dialog is open
        return QMessageBox.Yes

    monkeypatch.setattr(QMessageBox, "question", confirm_while_a_ticket_arrives)
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    calls = []
    tab = make_tab(stored, review_batch=lambda tickets, ids: calls.append(ids))

    tab.start_review_all()
    wait_until_idle(qapp, tab)

    assert calls == [[CATEGORY_FIXTURES["escalation_internal"]["zendesk_id"]]]


def test_default_batch_does_not_bill_tickets_added_after_confirmation(qapp, configured, stored, monkeypatch):
    from decision_engine import persistence

    late = copy.deepcopy(CATEGORY_FIXTURES["manual_triage"])
    classified = []

    def fake_process(zendesk_id, tickets, worker=None):
        classified.append(zendesk_id)
        return {"classification": "flag_solve", "confidence": 90}

    def confirm(*args, **kwargs):
        stored.insert_one(late)
        return QMessageBox.Yes

    monkeypatch.setattr(persistence, "process_one_ticket", fake_process)
    monkeypatch.setattr(QMessageBox, "question", confirm)
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    tab = make_tab(stored)

    tab.start_review_all()
    wait_until_idle(qapp, tab)

    assert classified == [CATEGORY_FIXTURES["escalation_internal"]["zendesk_id"]]


# --- shutdown ------------------------------------------------------------------


def test_request_shutdown_when_idle_allows_close(qapp, configured, stored):
    tab = make_tab(stored)
    assert tab.request_shutdown() is True


def test_request_shutdown_waits_for_running_job(qapp, configured, dialogs, stored):
    import threading

    release = threading.Event()
    done_callbacks, idle = [], []
    tab = make_tab(stored)
    tab.idle.connect(lambda: idle.append(True))
    tab._run(lambda: release.wait(5), "Slow job", done_callbacks.append)

    assert tab.request_shutdown() is False
    assert tab.is_busy()
    assert "Closing after the current job finishes" in tab.status_label.text()

    release.set()
    deadline = time.monotonic() + 5
    while not idle:
        qapp.processEvents()
        assert time.monotonic() < deadline
        time.sleep(0.01)

    assert not tab.is_busy()
    assert done_callbacks == []  # no follow-up jobs or dialogs while closing
    assert dialogs == []


def test_request_shutdown_cancels_rest_of_default_batch(qapp, configured, tickets, monkeypatch):
    import threading

    from decision_engine import persistence

    for category in ("flag_solve", "escalation_internal", "flag_reengage"):
        tickets.insert_one(copy.deepcopy(CATEGORY_FIXTURES[category]))
    started, release = threading.Event(), threading.Event()
    classified, idle = [], []

    def slow_process(zendesk_id, tickets, worker=None):
        classified.append(zendesk_id)
        started.set()
        release.wait(5)
        return {"classification": "flag_solve", "confidence": 90}

    monkeypatch.setattr(persistence, "process_one_ticket", slow_process)
    tab = make_tab(tickets)
    tab.idle.connect(lambda: idle.append(True))
    tab._run(lambda: tab.review_batch(tickets, persistence.awaiting_ids(tickets)), "Batch", lambda _: None)

    assert started.wait(5)
    assert tab.request_shutdown() is False
    release.set()
    deadline = time.monotonic() + 5
    while not idle:
        qapp.processEvents()
        assert time.monotonic() < deadline
        time.sleep(0.01)

    assert len(classified) == 1  # the in-flight ticket finished; the rest were cancelled


def test_wait_for_job_joins_thread(qapp, configured, stored):
    import threading

    release = threading.Event()
    tab = make_tab(stored)
    tab._run(lambda: release.wait(5), "Slow job", lambda _: None)
    threading.Timer(0.1, release.set).start()

    tab.wait_for_job()
    assert not tab.is_busy()


def test_no_new_jobs_after_shutdown_requested(qapp, configured, stored):
    tab = make_tab(stored)
    tab.request_shutdown()
    tab.refresh()
    assert not tab.is_busy()


# --- embeddings ----------------------------------------------------------------


def test_embed_tickets_asks_then_embeds_and_reports(qapp, configured, monkeypatch, dialogs, stored):
    import numpy as np

    from decision_engine import search

    monkeypatch.setenv("EMBEDDING_API_KEY", "k")
    monkeypatch.delenv("EMBEDDING_DIMENSIONS", raising=False)
    sent = []

    def fake_embed(texts, cfg, input_type="document"):
        sent.extend(texts)
        return np.zeros((len(texts), cfg.dimensions), dtype=np.float32), 5

    def no_search(*args):
        raise search.SearchUnavailable("no Atlas Search here")

    monkeypatch.setattr(triage_tab.embeddings, "embed", fake_embed)
    monkeypatch.setattr(search, "check_vector_index", no_search)
    tab = make_tab(stored)
    tab.start_embed_tickets()
    wait_until_idle(qapp, tab)

    kinds = [kind for kind, _ in dialogs]
    assert kinds == ["question", "information"]
    assert "3 ticket(s) need embeddings" in dialogs[0][1]
    assert "Embedded: 3" in dialogs[1][1] and "skipped: this MongoDB deployment has no Atlas Search" in dialogs[1][1]
    assert len(sent) == 3
    assert stored.count_documents({"embedding_meta.model": "voyage-3.5"}) == 3


def test_index_mismatch_stops_before_any_paid_call(qapp, configured, monkeypatch, dialogs, stored):
    from decision_engine import search

    monkeypatch.setenv("EMBEDDING_API_KEY", "k")

    def mismatch(*args):
        raise search.VectorIndexMismatch("tickets_vector indexes 512-dimension vectors")

    called = []
    monkeypatch.setattr(search, "check_vector_index", mismatch)
    monkeypatch.setattr(triage_tab.embeddings, "embed", lambda *a, **k: called.append(1))
    tab = make_tab(stored)
    tab.start_embed_tickets()
    wait_until_idle(qapp, tab)
    assert called == []
    assert not any(kind == "question" for kind, _ in dialogs)
    assert any("512-dimension" in text for _, text in dialogs)
    assert stored.count_documents({"embedding": {"$exists": True}}) == 0


def test_embed_tickets_needs_a_key(qapp, configured, monkeypatch, dialogs, stored):
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
    tab = make_tab(stored)
    tab.start_embed_tickets()
    assert dialogs and dialogs[0][0] == "warning" and "EMBEDDING_API_KEY" in dialogs[0][1]
    assert not tab.is_busy()


def test_embed_tickets_cancel_sends_nothing(qapp, configured, monkeypatch, dialogs, stored):
    from decision_engine import search

    monkeypatch.setenv("EMBEDDING_API_KEY", "k")
    monkeypatch.setattr(search, "check_vector_index", lambda *a: "missing")
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Cancel)
    called = []
    monkeypatch.setattr(triage_tab.embeddings, "run_backfill", lambda *a, **k: called.append(1))
    tab = make_tab(stored)
    tab.start_embed_tickets()
    wait_until_idle(qapp, tab)
    assert called == []
    assert stored.count_documents({"embedding": {"$exists": True}}) == 0


def test_index_changed_during_the_run_is_an_error_not_a_warning(qapp, configured, monkeypatch, dialogs, stored):
    import numpy as np

    from decision_engine import search

    monkeypatch.setenv("EMBEDDING_API_KEY", "k")
    monkeypatch.setattr(triage_tab.embeddings, "embed",
                        lambda texts, cfg, input_type="document": (np.zeros((len(texts), cfg.dimensions)), 0))
    monkeypatch.setattr(search, "check_vector_index", lambda *a: "missing")

    def changed(*args):
        raise search.VectorIndexMismatch("tickets_vector now indexes 256-dimension vectors")

    monkeypatch.setattr(search, "ensure_vector_index", changed)
    tab = make_tab(stored)
    tab.start_embed_tickets()
    wait_until_idle(qapp, tab)
    assert [kind for kind, _ in dialogs] == ["question", "critical"]
    assert "256-dimension" in dialogs[1][1]


# --- weekly review (misclassification_log and lessons_learned) ----------------------


@pytest.fixture
def logged(tickets):
    """One reviewed ticket logged twice as misclassified this week."""
    from bson import ObjectId

    from decision_engine import feedback

    ticket = copy.deepcopy(CATEGORY_FIXTURES["flag_solve"])
    ticket["awaiting_review"] = False
    ticket["agent_reviews"] = [review("flag_solve", 5, review_id=ObjectId())]
    tickets.insert_one(ticket)
    log, _ = feedback.get_feedback_collections(tickets.database)
    review_id = ticket["agent_reviews"][0]["review_id"]
    feedback.log_misclassification(tickets, log, ticket["zendesk_id"], review_id, "flag_reengage", "no reply yet")
    feedback.log_misclassification(tickets, log, ticket["zendesk_id"], review_id, "flag_reengage")
    return tickets


def open_review(qapp, tab):
    tab.open_weekly_review()
    dialog = tab.weekly_review_dialog
    wait_until_idle(qapp, tab)
    return dialog


def test_weekly_review_lists_the_weeks_entries(qapp, configured, dialogs, logged):
    tab = make_tab(logged)
    dialog = open_review(qapp, tab)
    assert dialog.log_table.rowCount() == 2
    assert dialog.log_table.item(0, 5).text() == "not reviewed"
    # Nothing reviewed yet: no summary can be drafted.
    assert not dialog.suggest_button.isEnabled() and not dialog.save_draft_button.isEnabled()

    dialog._shift_week(-7)
    wait_until_idle(qapp, tab)
    assert dialog.log_table.rowCount() == 0


def test_weekly_review_marks_drafts_and_approves(qapp, configured, dialogs, logged):
    log = logged.database.misclassification_log
    lessons = logged.database.lessons_learned
    tab = make_tab(logged)
    dialog = open_review(qapp, tab)
    dialog.reviewer_input.setText("lead")

    dialog.log_table.selectAll()
    dialog.mark_selected_reviewed()
    wait_until_idle(qapp, tab)
    assert log.count_documents({"reviewed": True, "reviewed_by": "lead"}) == 2
    assert dialog.log_table.item(0, 5).text() == "lead"

    dialog.suggest_summary()
    assert "should have been flag_reengage" in dialog.draft_input.toPlainText()
    dialog.save_draft()
    wait_until_idle(qapp, tab)
    draft = lessons.find_one()
    assert draft["approved"] is False and len(draft["based_on_log_ids"]) == 2
    assert dialog.lesson_table.rowCount() == 1 and dialog.lesson_table.item(0, 1).text() == "Draft"
    assert dialog.draft_input.toPlainText() == ""

    dialog.reviewer_input.setText("vp")
    dialog.lesson_table.selectRow(0)
    assert dialog.approve_button.isEnabled()
    dialog.approve_selected()
    wait_until_idle(qapp, tab)
    assert lessons.find_one()["approved_by"] == "vp"
    assert dialog.lesson_table.item(0, 1).text() == "Approved by vp"


def test_weekly_review_needs_a_reviewer_name(qapp, configured, dialogs, logged):
    tab = make_tab(logged)
    dialog = open_review(qapp, tab)
    dialog.reviewer_input.setText("")
    dialog.log_table.selectAll()
    dialog.mark_selected_reviewed()
    wait_until_idle(qapp, tab)
    assert dialogs[-1][0] == "warning"
    assert logged.database.misclassification_log.count_documents({"reviewed": True}) == 0


def test_weekly_review_recovers_from_an_error(qapp, configured, dialogs, logged, monkeypatch):
    from decision_engine import feedback

    tab = make_tab(logged)
    dialog = open_review(qapp, tab)

    def broken(*a, **k):
        raise RuntimeError("mongo down")

    monkeypatch.setattr(feedback, "misclassifications_for_week", broken)
    dialog.load()
    wait_until_idle(qapp, tab)
    assert dialogs[-1][0] == "critical"
    assert dialog.refresh_button.isEnabled() and "failed" in dialog.status_label.text()


def test_weekly_review_waits_for_another_job(qapp, configured, dialogs, logged):
    tab = make_tab(logged)
    dialog = open_review(qapp, tab)
    tab._thread = object()  # another job running
    try:
        dialog.load()
        assert "Another Decision Engine job" in dialog.status_label.text()
        assert dialog.refresh_button.isEnabled()
    finally:
        tab._thread = None
