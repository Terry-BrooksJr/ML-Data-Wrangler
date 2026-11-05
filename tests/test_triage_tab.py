import copy
import os
import time
from datetime import datetime, timedelta, timezone

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from ui.dialogs import ab_test_dialog
import mongomock
import pytest
from ui.tabs import triage_tab
from decision_engine import pricing
from decision_engine.zendesk_status import REVIEW_ADDED_KEY
from fixtures import CATEGORY_FIXTURES
from pymongo.errors import ServerSelectionTimeoutError
from PyQt5 import QtCore
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QApplication, QMessageBox
from ui import theme
from ui.tabs.triage_tab import count_awaiting, format_review, recent_reviews, TriageTab

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
    reengage["agent_reviews"] = [
        review("flag_reengage", 30, draft_message="Any update?")
    ]
    pending = copy.deepcopy(CATEGORY_FIXTURES["escalation_internal"])
    tickets.insert_many([solved, reengage, pending])
    return tickets


# --- queries ---------------------------------------------------------------


def test_count_awaiting(stored):
    assert count_awaiting(stored) == 1


def add_reviewed(tickets, zendesk_id, *reviews):
    """Another reviewed ticket, so a test has more than two rows."""
    ticket = copy.deepcopy(CATEGORY_FIXTURES["on_track"])
    ticket.update(zendesk_id=zendesk_id, awaiting_review=False, agent_reviews=list(reviews))
    tickets.insert_one(ticket)


def test_recent_reviews_shows_each_tickets_latest_review(stored):
    rows = recent_reviews(stored)
    assert [row["review"]["classification"] for row in rows] == ["flag_solve", "flag_reengage"]
    assert {"zendesk_id", "subject", "review", "earlier_reviews"} <= set(rows[0])
    # The ticket's older review is summarised, not given its own row.
    assert [(e["review_id"], e["classification"]) for e in rows[0]["earlier_reviews"]] == [
        ("r-manual_triage-60", "manual_triage")
    ]
    assert rows[1]["earlier_reviews"] == []


def test_recent_reviews_shows_a_rerun_once_it_is_the_newest(stored):
    zendesk_id = CATEGORY_FIXTURES["flag_reengage"]["zendesk_id"]
    stored.update_one(
        {"zendesk_id": zendesk_id},
        {"$push": {"agent_reviews": review("flag_reengage", 1, rerun_of="r-flag_reengage-30")}},
    )
    rows = {row["zendesk_id"]: row for row in recent_reviews(stored)}
    assert rows[zendesk_id]["review"]["rerun_of"] == "r-flag_reengage-30"
    assert [e["review_id"] for e in rows[zendesk_id]["earlier_reviews"]] == ["r-flag_reengage-30"]
    assert "(re-run with evidence)" not in format_review(rows[zendesk_id])
    assert "Earlier reviews of this ticket (newest first):" in format_review(rows[zendesk_id])


def test_recent_reviews_respects_limit(stored):
    assert len(recent_reviews(stored, limit=1)) == 1


def test_recent_reviews_returns_more_than_the_old_200(stored):
    for n in range(250):
        add_reviewed(stored, f"t{n}", review("on_track", 100 + n, review_id=f"t{n}"))
    assert len(recent_reviews(stored)) == 252


def test_a_batch_review_replaces_its_tickets_row():
    rows = [
        {"zendesk_id": "1", "review": {"review_id": "a", "classification": "manual_triage"},
         "earlier_reviews": [], "case": {"status": triage.OPEN}},
        {"zendesk_id": "2", "review": {"review_id": "b"}, "earlier_reviews": []},
    ]
    new = {"zendesk_id": "1", "review": {"review_id": "c", "classification": "flag_solve"}}
    replaced = triage_tab.replace_ticket_row(rows, new)
    assert [r["review"]["review_id"] for r in replaced] == ["c", "b"]
    assert [e["review_id"] for e in replaced[0]["earlier_reviews"]] == ["a"]
    assert replaced[0]["case"] == {"status": triage.OPEN}


def test_format_review_includes_draft_only_when_present(stored):
    rows = {row["review"]["classification"]: row for row in recent_reviews(stored)}
    assert "Draft message (unsent):\nAny update?" in format_review(
        rows["flag_reengage"]
    )
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

    def ask_batch_size(parent, awaiting, note):
        # Recorded like the other dialogs; answers "all of them".
        shown.append(
            ("question", f"{awaiting} awaiting" + (f"\n\nWarning:\n{note}" if note else ""))
        )
        return awaiting

    monkeypatch.setattr(triage_tab, "ask_batch_size", ask_batch_size)
    return shown


def wait_until_idle(qapp, tab, timeout=5.0):
    deadline = time.monotonic() + timeout
    while tab.is_busy() or not tab.refresh_button.isEnabled():
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
    assert tab.table.rowCount() == 2
    classification_column = triage_tab.COLUMNS.index("Classification")
    shown = [tab.table.item(row, classification_column).text() for row in range(2)]
    assert shown == ["flag_solve", "flag_reengage"]


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

    assert tab.table.rowCount() == 2
    assert dialogs == []


def test_review_needs_model_settings(qapp, configured, monkeypatch, dialogs, stored):
    monkeypatch.delenv("CHEAPER_INFERENCE_PRIMARY_MODEL_NAME")
    calls = []
    tab = make_tab(stored, review_batch=lambda tickets, ids, **kw: calls.append(tickets))

    tab.start_batch()

    assert calls == []
    assert dialogs[0][0] == "warning"
    assert "CHEAPER_INFERENCE_PRIMARY_MODEL_NAME" in tab.status_label.text()


@pytest.mark.parametrize(
    "optional",
    [
        "CHEAPER_INFERENCE_SECONDARY_MODEL_NAME",
        "CHEAPER_INFERENCE_CONFIRMATION_MODEL_NAME",
    ],
)
def test_review_runs_without_optional_models(
    qapp, configured, monkeypatch, dialogs, stored, optional
):
    monkeypatch.delenv(optional, raising=False)
    calls = []
    tab = make_tab(stored, review_batch=lambda tickets, ids, **kw: calls.append(tickets))

    tab.start_batch()
    wait_until_idle(qapp, tab)

    assert calls == [stored]
    assert f"{optional} is not set" in dialogs[0][1]


def test_no_optional_warning_when_all_set(qapp, configured, dialogs, stored):
    tab = make_tab(stored, review_batch=lambda tickets, ids, **kw: None)
    tab.start_batch()
    wait_until_idle(qapp, tab)

    assert "Warning" not in dialogs[0][1]


def test_review_one_warns_about_unset_confirmation(
    qapp, configured, monkeypatch, dialogs, stored
):
    monkeypatch.delenv("CHEAPER_INFERENCE_CONFIRMATION_MODEL_NAME")
    tab = make_tab(
        stored,
        review_one=lambda zendesk_id, tickets: {
            "classification": "manual_triage",
            "confidence": 40,
        },
    )
    tab.ticket_id_input.setText("123")
    tab.start_review_one()
    wait_until_idle(qapp, tab)

    assert "CHEAPER_INFERENCE_CONFIRMATION_MODEL_NAME is not set" in dialogs[0][1]


def test_missing_env_blocks_actions(qapp, configured, monkeypatch, dialogs, stored):
    monkeypatch.delenv("MONGO_URI")
    calls = []
    tab = make_tab(stored, review_batch=lambda tickets, ids, **kw: calls.append(tickets))

    tab.start_batch()

    assert calls == []
    assert dialogs[0][0] == "warning"
    assert "MONGO_URI" in tab.status_label.text()


def test_review_all_confirms_then_runs_batch(qapp, configured, dialogs, stored):
    calls = []
    tab = make_tab(stored, review_batch=lambda tickets, ids, **kw: calls.append(tickets))

    tab.start_batch()
    wait_until_idle(qapp, tab)

    # Exactly one confirm and one completion: the count and refresh are silent.
    assert dialogs == [
        ("question", dialogs[0][1]),
        ("information", "Review batch finished. Results are listed below."),
    ]
    assert dialogs[0][1].startswith("1 awaiting")
    assert calls == [stored]


def test_review_all_cancel_does_not_run(qapp, configured, dialogs, monkeypatch, stored):
    monkeypatch.setattr(triage_tab, "ask_batch_size", lambda *a: None)
    calls = []
    tab = make_tab(stored, review_batch=lambda tickets, ids, **kw: calls.append(tickets))

    tab.start_batch()
    wait_until_idle(qapp, tab)

    assert calls == []


def test_review_all_with_nothing_awaiting_skips_confirm(
    qapp, configured, dialogs, tickets
):
    calls = []
    tab = make_tab(tickets, review_batch=lambda t, ids, **kw: calls.append(t))

    tab.start_batch()
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

    assert dialogs == [
        (
            "information",
            "Ticket 999 was not found, is not awaiting review, or was reassigned while "
            "it was classified (it stays awaiting review).",
        )
    ]


def test_engine_error_is_shown_and_tab_recovers(qapp, configured, dialogs, stored):
    def failing_batch(tickets, ids, **kw):
        raise KeyError("confidence")

    tab = make_tab(stored, review_batch=failing_batch)
    tab.start_batch()
    wait_until_idle(qapp, tab)

    assert dialogs[-1][0] == "critical"
    assert tab.batch_button.isEnabled()


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
    summary = {
        "classified": 1,
        "manual_fallback": 0,
        "skipped": 0,
        "errored": 2,
        "stopped": None,
        "usage": {},
    }
    tab = make_tab(stored, review_batch=lambda tickets, ids, **kw: summary)
    tab.start_batch()
    wait_until_idle(qapp, tab)

    assert dialogs[-1][0] == "information"
    assert "Engine errors (left in the queue to retry): 2" in dialogs[-1][1]


def test_batch_rows_carry_the_tickets_zendesk_status(qapp, configured, dialogs, stored):
    # A ticket marked added before reviews were tracked is re-reviewed.
    zendesk_id = CATEGORY_FIXTURES["flag_reengage"]["zendesk_id"]
    stored.update_one({"zendesk_id": zendesk_id}, {"$set": {REVIEW_ADDED_KEY: True}})
    new_review = review("flag_solve", 0)
    progress = {
        "zendesk_id": zendesk_id,
        "outcome": "classified",
        "review": new_review,
        "done": 1,
        "total": 1,
        "summary": {},
    }
    tab = make_tab(stored, review_batch=lambda tickets, ids, on_progress=None: on_progress(progress))
    emitted = []
    tab._batch_relay.progress.connect(emitted.append)
    tab.start_batch()
    wait_until_idle(qapp, tab)

    assert emitted[0]["ticket"][REVIEW_ADDED_KEY] is True
    tab._batch = {"total": 1, "started": time.monotonic(), "subjects": {}}
    tab._on_batch_progress(emitted[0])
    assert triage_tab.row_added(tab._rows[0])
    added = triage_tab.COLUMNS.index("Added to Zendesk")
    newest = next(
        r for r in range(tab.table.rowCount())
        if tab.table.item(r, 0).data(QtCore.Qt.UserRole) == 0
    )
    assert tab.table.item(newest, added).text() == "Yes"


def test_review_all_warns_when_batch_stopped(qapp, configured, dialogs, stored):
    summary = {
        "classified": 0,
        "manual_fallback": 0,
        "skipped": 0,
        "errored": 0,
        "stopped": "AuthenticationError: bad key",
        "usage": {},
    }
    tab = make_tab(stored, review_batch=lambda tickets, ids, **kw: summary)
    tab.start_batch()
    wait_until_idle(qapp, tab)

    assert dialogs[-1][0] == "warning"
    assert "Stopped early" in dialogs[-1][1]


def test_format_review_shows_overrides_engine_error_and_audit(stored):
    row = {
        "zendesk_id": "1",
        "subject": "s",
        "outcomes": [
            {
                "review_id": "r1",
                "outcome": "bad",
                "detection_method": "manual",
                "detail": "wrong",
            }
        ],
        "review": review(
            "manual_triage",
            1,
            review_id="r1",
            guard_overrides=["Downgraded from flag_solve"],
            engine_error="cut off",
            audit={
                "usage": {"input_tokens": 10, "output_tokens": 5},
                "latency_ms": 12,
                "request_id": "req_1",
                "prompt_version": "abc",
            },
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


def test_review_added_asks_then_marks_the_ticket(qapp, configured, dialogs, stored):
    tab = make_tab(stored)
    tab.refresh()
    wait_until_idle(qapp, tab)
    assert not tab.review_added_button.isEnabled()

    tab.table.selectRow(0)
    zendesk_id = tab._selected_row()["zendesk_id"]
    assert "Review added to Zendesk: No" in tab.detail.toPlainText()
    tab.mark_selected_review_added()
    wait_until_idle(qapp, tab)

    assert dialogs[-1] == (
        "question",
        f"Has the review of ticket {zendesk_id} been manually added to Zendesk?",
    )
    assert stored.find_one({"zendesk_id": zendesk_id})[REVIEW_ADDED_KEY] is True
    # The refresh keeps the row selected and shows the change; it can't be
    # marked twice.
    assert tab._selected_row()["zendesk_id"] == zendesk_id
    assert "Review added to Zendesk: Yes" in tab.detail.toPlainText()
    assert not tab.review_added_button.isEnabled()


def test_copy_puts_the_whole_review_on_the_clipboard(qapp, configured, stored):
    tab = make_tab(stored)
    tab.refresh()
    wait_until_idle(qapp, tab)
    assert not tab.copy_button.isEnabled()

    tab.table.selectRow(0)
    assert tab.copy_button.isEnabled() and not tab.copy_button.icon().isNull()
    tab.copy_selected_review()

    assert QApplication.clipboard().text() == format_review(tab._selected_row())
    assert "copied" in tab.status_label.text()


def test_table_shows_decision_and_added_to_zendesk(qapp, configured, stored):
    add_reviewed(stored, "t-old", review("on_track", 90, review_id="t-old"))
    tab = make_tab(stored)
    tab.refresh()
    wait_until_idle(qapp, tab)
    first, second = tab._rows[0], tab._rows[1]
    stored.update_one(
        {"zendesk_id": first["zendesk_id"]},
        {"$set": {"outcomes": [{"review_id": first["review"]["review_id"], "outcome": "good"}]}},
    )
    stored.update_one(
        {"zendesk_id": second["zendesk_id"]},
        {"$set": {
            REVIEW_ADDED_KEY: True,
            "outcomes": [{"review_id": second["review"]["review_id"], "outcome": "bad"}],
        }},
    )
    tab.refresh()
    wait_until_idle(qapp, tab)

    def column(name):
        index = triage_tab.COLUMNS.index(name)
        return [tab.table.item(row, index).text() for row in range(3)]

    assert column("Decision") == ["Correct", "Misclassified", ""]
    assert column("Added to Zendesk") == ["No", "Yes", "No"]
    # Only the review with both a decision and the Zendesk flag is greyed out.
    faint = QColor(theme.TEXT_FAINT)
    greyed = [tab.table.item(row, 0).foreground().color() == faint for row in range(3)]
    assert greyed == [False, True, False]


def test_step_awaiting_skips_decided_and_added_rows():
    rows = [
        {"zendesk_id": "1", "review": {"review_id": "a"}},
        {"zendesk_id": "2", "review": {"review_id": "b"},
         "outcomes": [{"review_id": "b", "outcome": "good"}]},
        {"zendesk_id": "3", "review": {"review_id": "c"}, REVIEW_ADDED_KEY: True},
        {"zendesk_id": "4", "review": {"review_id": "d"},
         "outcomes": [{"review_id": "d", "outcome": "pending"}]},
        {"zendesk_id": "5", "review": {"review_id": "e"},
         "outcomes": [{"review_id": "e", "outcome": "bad"}]},
    ]
    assert triage_tab.step_awaiting(rows, None, 1) == 0
    assert triage_tab.step_awaiting(rows, None, -1) is None
    assert triage_tab.step_awaiting(rows, 0, 1) == 3
    assert triage_tab.step_awaiting(rows, 3, 1) is None
    assert triage_tab.step_awaiting(rows, 4, -1) == 3
    assert triage_tab.step_awaiting(rows, 3, -1) == 0


def test_owner_label_names_roster_staff():
    assert triage_tab.owner_label({"assignee": "1085778551"}) == "Rich Shupe"
    assert triage_tab.owner_label({"assignee": "999"}) == "999"
    assert triage_tab.owner_label({"assignee": None}) == triage_tab.UNASSIGNED
    assert triage_tab.owner_label({}) == triage_tab.UNASSIGNED


def test_row_matches_every_chosen_filter():
    row = {
        "assignee": "1085778551",
        "review": {"review_id": "a", "classification": "flag_solve"},
        "outcomes": [{"review_id": "a", "outcome": "good"}],
    }
    assert triage_tab.row_matches(row, {name: None for name in triage_tab.FILTERS})
    assert triage_tab.row_matches(
        row,
        {
            "Classification": "flag_solve",
            "Decision": "Correct",
            "Owner": "Rich Shupe",
            "Added to Zendesk": "No",
        },
    )
    assert not triage_tab.row_matches(row, {"Decision": triage_tab.NO_DECISION})
    assert not triage_tab.row_matches(row, {"Owner": "Chris Short"})
    assert triage_tab.row_matches(
        {"review": {"review_id": "b"}}, {"Decision": triage_tab.NO_DECISION}
    )


def test_filter_options_list_fixed_values_then_those_found():
    rows = [
        {"assignee": "999", "review": {"classification": "custom"}},
        {"assignee": None, "review": {"classification": "on_track"}},
        {"assignee": "1085778551", "review": {"classification": "on_track"}},
    ]
    assert triage_tab.filter_options("Owner", rows) == ["999", "Rich Shupe", "Unassigned"]
    assert triage_tab.filter_options("Classification", rows) == [
        *triage_tab.CLASSIFICATIONS,
        "custom",
    ]
    assert triage_tab.filter_options("Decision", []) == ["Correct", "Misclassified", "Pending"]


def test_filters_hide_rows_and_steer_navigation(qapp, configured, stored):
    add_reviewed(stored, "t-old", review("on_track", 90, review_id="t-old"))
    stored.update_one({"zendesk_id": "t-old"}, {"$set": {"assignee": "1085778551"}})
    tab = make_tab(stored)
    tab.refresh()
    wait_until_idle(qapp, tab)
    owner = tab.filter_combos["Owner"]
    assert [owner.itemText(i) for i in range(owner.count())] == [
        "All",
        "Rich Shupe",
        "Support Owner A",
    ]
    assert not tab.clear_filters_button.isEnabled()

    owner.setCurrentIndex(owner.findData("Rich Shupe"))
    visible = [r for r in range(tab.table.rowCount()) if not tab.table.isRowHidden(r)]
    assert len(visible) == 1
    owner_column = triage_tab.COLUMNS.index("Owner")
    assert tab.table.item(visible[0], owner_column).text() == "Rich Shupe"
    assert tab.filter_label.text() == "Showing 1 of 3 reviews."
    # Next goes straight to the only shown review, past newer hidden ones.
    tab.select_awaiting(1)
    assert tab._selected_row()["zendesk_id"] == "t-old"
    assert not tab.next_button.isEnabled() and not tab.previous_button.isEnabled()

    # A refresh keeps the choice.
    tab.refresh()
    wait_until_idle(qapp, tab)
    assert owner.currentData() == "Rich Shupe"
    assert sum(not tab.table.isRowHidden(r) for r in range(3)) == 1

    # Filtering the selected row out clears the selection.
    tab.filter_combos["Classification"].setCurrentIndex(
        tab.filter_combos["Classification"].findData("flag_solve")
    )
    assert all(tab.table.isRowHidden(r) for r in range(3))
    assert tab._selected_row() is None

    tab.clear_filters()
    assert not any(tab.table.isRowHidden(r) for r in range(3))
    assert not tab.filter_label.isVisible()


def test_next_and_previous_move_between_undecided_reviews(qapp, configured, stored):
    add_reviewed(stored, "t-old", review("on_track", 90, review_id="t-old"))
    tab = make_tab(stored)
    tab.refresh()
    wait_until_idle(qapp, tab)
    # The middle review is already added to Zendesk, so navigation skips it.
    stored.update_one(
        {"zendesk_id": tab._rows[1]["zendesk_id"]}, {"$set": {REVIEW_ADDED_KEY: True}}
    )
    tab.refresh()
    wait_until_idle(qapp, tab)
    newest, oldest = tab._rows[0]["review"]["review_id"], tab._rows[2]["review"]["review_id"]
    assert tab.next_button.isEnabled() and not tab.previous_button.isEnabled()

    # Sorting by another column doesn't change the order Next follows.
    tab.table.sortItems(triage_tab.COLUMNS.index("Classification"))
    tab.select_awaiting(1)
    assert tab._selected_row()["review"]["review_id"] == newest
    tab.select_awaiting(1)
    assert tab._selected_row()["review"]["review_id"] == oldest
    assert not tab.next_button.isEnabled() and tab.previous_button.isEnabled()
    tab.select_awaiting(-1)
    assert tab._selected_row()["review"]["review_id"] == newest


def test_review_added_marks_only_the_selected_review(qapp, configured, dialogs, stored):
    tab = make_tab(stored)
    tab.refresh()
    wait_until_idle(qapp, tab)
    # Row 0's ticket also has an earlier review, which isn't marked.
    earlier = tab._rows[0]["earlier_reviews"][0]["review_id"]
    tab.table.selectRow(0)
    tab.mark_selected_review_added()
    wait_until_idle(qapp, tab)

    added = triage_tab.COLUMNS.index("Added to Zendesk")
    by_review = {
        tab._rows[tab.table.item(r, 0).data(QtCore.Qt.UserRole)]["review"]["review_id"]: (
            tab.table.item(r, added).text()
        )
        for r in range(2)
    }
    assert by_review == {
        tab._rows[0]["review"]["review_id"]: "Yes",
        tab._rows[1]["review"]["review_id"]: "No",
    }
    assert not triage_tab.review_added(tab._rows[0], earlier)


def test_status_says_when_reviews_are_capped(qapp, configured, monkeypatch, stored):
    add_reviewed(stored, "t-old", review("on_track", 90, review_id="t-old"))
    monkeypatch.setattr(triage_tab, "REVIEW_ROW_LIMIT", 2)
    tab = make_tab(stored)
    tab._show_loaded((1, recent_reviews(stored, limit=2), 0))
    assert tab.status_label.text() == (
        "1 ticket(s) awaiting review. Showing the 2 most recently reviewed tickets."
    )


def test_review_added_answering_no_changes_nothing(
    qapp, configured, dialogs, stored, monkeypatch
):
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.No)
    tab = make_tab(stored)
    tab.refresh()
    wait_until_idle(qapp, tab)

    tab.table.selectRow(0)
    tab.mark_selected_review_added()
    wait_until_idle(qapp, tab)

    assert stored.count_documents({REVIEW_ADDED_KEY: True}) == 0


def test_mark_misclassified_logs_entry(qapp, configured, dialogs, tickets, monkeypatch):
    from bson import ObjectId
    from PyQt5.QtWidgets import QInputDialog

    ticket = copy.deepcopy(CATEGORY_FIXTURES["flag_solve"])
    ticket["awaiting_review"] = False
    ticket["agent_reviews"] = [review("flag_solve", 5, review_id=ObjectId())]
    tickets.insert_one(ticket)
    monkeypatch.setattr(
        QInputDialog, "getItem", lambda *a, **k: ("flag_reengage", True)
    )
    monkeypatch.setattr(
        QInputDialog, "getText", lambda *a, **k: ("customer never confirmed", True)
    )
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


def test_review_all_reviews_only_the_confirmed_snapshot(
    qapp, configured, stored, monkeypatch
):
    late = copy.deepcopy(CATEGORY_FIXTURES["manual_triage"])

    def confirm_while_a_ticket_arrives(parent, awaiting, note):
        stored.insert_one(late)  # e.g. the Mongo writer, while the dialog is open
        return awaiting

    monkeypatch.setattr(triage_tab, "ask_batch_size", confirm_while_a_ticket_arrives)
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    calls = []
    tab = make_tab(stored, review_batch=lambda tickets, ids, **kw: calls.append(ids))

    tab.start_batch()
    wait_until_idle(qapp, tab)

    assert calls == [[CATEGORY_FIXTURES["escalation_internal"]["zendesk_id"]]]


def test_default_batch_does_not_bill_tickets_added_after_confirmation(
    qapp, configured, stored, monkeypatch
):
    from decision_engine import persistence

    late = copy.deepcopy(CATEGORY_FIXTURES["manual_triage"])
    classified = []

    def fake_process(zendesk_id, tickets, worker=None):
        classified.append(zendesk_id)
        return {"classification": "flag_solve", "confidence": 90}

    def confirm(parent, awaiting, note):
        stored.insert_one(late)
        return awaiting

    monkeypatch.setattr(persistence, "process_one_ticket", fake_process)
    monkeypatch.setattr(triage_tab, "ask_batch_size", confirm)
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    tab = make_tab(stored)

    tab.start_batch()
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


def test_request_shutdown_cancels_rest_of_default_batch(
    qapp, configured, tickets, monkeypatch
):
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
    tab._run(
        lambda: tab.review_batch(tickets, persistence.awaiting_ids(tickets)),
        "Batch",
        lambda _: None,
    )

    assert started.wait(5)
    assert tab.request_shutdown() is False
    release.set()
    deadline = time.monotonic() + 5
    while not idle:
        qapp.processEvents()
        assert time.monotonic() < deadline
        time.sleep(0.01)

    assert (
        len(classified) == 1
    )  # the in-flight ticket finished; the rest were cancelled


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


def test_embed_tickets_asks_then_embeds_and_reports(
    qapp, configured, monkeypatch, dialogs, stored
):
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
    assert (
        "Embedded: 3" in dialogs[1][1]
        and "skipped: this MongoDB deployment has no Atlas Search" in dialogs[1][1]
    )
    assert len(sent) == 3
    assert stored.count_documents({"embedding_meta.model": "voyage-3.5"}) == 3


def test_index_mismatch_stops_before_any_paid_call(
    qapp, configured, monkeypatch, dialogs, stored
):
    from decision_engine import search

    monkeypatch.setenv("EMBEDDING_API_KEY", "k")

    def mismatch(*args):
        raise search.VectorIndexMismatch("tickets_vector indexes 512-dimension vectors")

    called = []
    monkeypatch.setattr(search, "check_vector_index", mismatch)
    monkeypatch.setattr(
        triage_tab.embeddings, "embed", lambda *a, **k: called.append(1)
    )
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
    assert (
        dialogs and dialogs[0][0] == "warning" and "EMBEDDING_API_KEY" in dialogs[0][1]
    )
    assert not tab.is_busy()


def test_embed_tickets_cancel_sends_nothing(
    qapp, configured, monkeypatch, dialogs, stored
):
    from decision_engine import search

    monkeypatch.setenv("EMBEDDING_API_KEY", "k")
    monkeypatch.setattr(search, "check_vector_index", lambda *a: "missing")
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Cancel)
    called = []
    monkeypatch.setattr(
        triage_tab.embeddings, "run_backfill", lambda *a, **k: called.append(1)
    )
    tab = make_tab(stored)
    tab.start_embed_tickets()
    wait_until_idle(qapp, tab)
    assert called == []
    assert stored.count_documents({"embedding": {"$exists": True}}) == 0


def test_index_changed_during_the_run_is_an_error_not_a_warning(
    qapp, configured, monkeypatch, dialogs, stored
):
    import numpy as np
    from decision_engine import search

    monkeypatch.setenv("EMBEDDING_API_KEY", "k")
    monkeypatch.setattr(
        triage_tab.embeddings,
        "embed",
        lambda texts, cfg, input_type="document": (
            np.zeros((len(texts), cfg.dimensions)),
            0,
        ),
    )
    monkeypatch.setattr(search, "check_vector_index", lambda *a: "missing")

    def changed(*args):
        raise search.VectorIndexMismatch(
            "tickets_vector now indexes 256-dimension vectors"
        )

    monkeypatch.setattr(search, "ensure_vector_index", changed)
    tab = make_tab(stored)
    tab.start_embed_tickets()
    wait_until_idle(qapp, tab)
    assert [kind for kind, _ in dialogs] == ["question", "critical"]
    assert "256-dimension" in dialogs[1][1]


def test_embedding_runs_beside_reviewing_and_stops_safely(
    qapp, configured, monkeypatch, dialogs, stored
):
    import threading

    import numpy as np
    from decision_engine import search

    monkeypatch.setenv("EMBEDDING_API_KEY", "k")
    monkeypatch.setattr(search, "check_vector_index", lambda *a: "missing")
    monkeypatch.setattr(search, "ensure_vector_index", lambda *a: "created")
    # One ticket per batch, so Stop lands between batches.
    monkeypatch.setattr(triage_tab.embeddings, "batch_chars", lambda cfg: 1)
    release, first_sent = threading.Event(), threading.Event()
    sent = []

    def slow_embed(texts, cfg, input_type="document"):
        sent.extend(texts)
        first_sent.set()
        release.wait(5)
        return np.zeros((len(texts), cfg.dimensions), dtype=np.float32), 1

    monkeypatch.setattr(triage_tab.embeddings, "embed", slow_embed)
    tab = make_tab(stored)
    tab.refresh()
    wait_until_idle(qapp, tab)
    tab.start_embed_tickets()
    try:
        deadline = time.monotonic() + 5
        while not first_sent.is_set():
            qapp.processEvents()
            assert time.monotonic() < deadline
            time.sleep(0.01)

        # Reviewing still works while the first batch is in flight.
        assert tab.is_embedding() and not tab._job_running()
        tab.table.selectRow(0)
        assert tab.mark_good_button.isEnabled() and tab.refresh_button.isEnabled()
        assert not tab.embed_button.isEnabled() and not tab.embed_stop_button.isHidden()
        tab.stop_embedding()
    finally:
        # A running QThread must not be destroyed, even when an assert fails.
        release.set()
        wait_until_idle(qapp, tab)

    # The batch in flight was saved; the rest were never sent.
    assert len(sent) == 1
    assert stored.count_documents({"embedding": {"$exists": True}}) == 1
    assert dialogs[-1][0] == "information" and "Not started (cancelled): 2" in dialogs[-1][1]
    assert tab.embed_stop_button.isHidden() and tab.embed_button.isEnabled()


def test_request_shutdown_stops_embedding_after_the_batch_in_flight(
    qapp, configured, monkeypatch, dialogs, stored
):
    import threading

    import numpy as np
    from decision_engine import search

    monkeypatch.setenv("EMBEDDING_API_KEY", "k")
    monkeypatch.setattr(search, "check_vector_index", lambda *a: "missing")
    monkeypatch.setattr(search, "ensure_vector_index", lambda *a: "created")
    monkeypatch.setattr(triage_tab.embeddings, "batch_chars", lambda cfg: 1)
    release, first_sent = threading.Event(), threading.Event()
    sent = []

    def slow_embed(texts, cfg, input_type="document"):
        sent.extend(texts)
        first_sent.set()
        release.wait(5)
        return np.zeros((len(texts), cfg.dimensions), dtype=np.float32), 1

    monkeypatch.setattr(triage_tab.embeddings, "embed", slow_embed)
    tab = make_tab(stored)
    idle = []
    tab.idle.connect(lambda: idle.append(1))
    tab.start_embed_tickets()
    try:
        deadline = time.monotonic() + 5
        while not first_sent.is_set():
            qapp.processEvents()
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert not tab.request_shutdown()
    finally:
        release.set()
        deadline = time.monotonic() + 5
        while tab.is_busy():
            qapp.processEvents()
            assert time.monotonic() < deadline
            time.sleep(0.01)
        qapp.processEvents()

    # Shutdown let the batch in flight finish and save, then started no other.
    assert idle == [1]
    assert len(sent) == 1
    assert stored.count_documents({"embedding": {"$exists": True}}) == 1
    # Closing shows no completion dialog.
    assert not any(kind == "information" for kind, _ in dialogs)


# --- weekly review (misclassification_log and lessons_learned) ----------------------


@pytest.fixture
def logged(tickets):
    """One ticket with two reviews, each logged as misclassified this week."""
    from bson import ObjectId
    from decision_engine import feedback

    ticket = copy.deepcopy(CATEGORY_FIXTURES["flag_solve"])
    ticket["awaiting_review"] = False
    ticket["agent_reviews"] = [
        review("flag_solve", 5, review_id=ObjectId()),
        review("flag_solve", 15, review_id=ObjectId()),
    ]
    tickets.insert_one(ticket)
    log, _ = feedback.get_feedback_collections(tickets.database)
    first, second = (r["review_id"] for r in ticket["agent_reviews"])
    feedback.log_misclassification(
        tickets, log, ticket["zendesk_id"], first, "flag_reengage", "no reply yet"
    )
    feedback.log_misclassification(
        tickets, log, ticket["zendesk_id"], second, "flag_reengage"
    )
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
    assert (
        not dialog.suggest_button.isEnabled()
        and not dialog.save_draft_button.isEnabled()
    )

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
    assert (
        dialog.lesson_table.rowCount() == 1
        and dialog.lesson_table.item(0, 1).text() == "Draft"
    )
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
    assert (
        logged.database.misclassification_log.count_documents({"reviewed": True}) == 0
    )


def test_weekly_review_recovers_from_an_error(
    qapp, configured, dialogs, logged, monkeypatch
):
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


# --- Run Batch: N tickets, live progress, stop after the current ticket ----------


@pytest.fixture
def queue(tickets):
    """Four tickets awaiting review, inserted out of order."""
    tickets.insert_many(
        [
            {"zendesk_id": z, "subject": f"Ticket {z}", "awaiting_review": True}
            for z in ("12", "10", "13", "11")
        ]
    )
    return tickets


def fake_review(classification="flag_solve", minutes_ago=0):
    return review(classification, minutes_ago, review_id=f"r-{time.monotonic_ns()}")


def test_run_batch_reviews_the_oldest_n(qapp, configured, dialogs, queue, monkeypatch):
    monkeypatch.setattr(triage_tab, "ask_batch_size", lambda parent, awaiting, note: 2)
    calls = []
    tab = make_tab(queue, review_batch=lambda tickets, ids, **kw: calls.append(ids))
    tab.start_batch()
    wait_until_idle(qapp, tab)
    assert calls == [["10", "11"]]


def test_batch_size_prompt_offers_one_to_all(qapp, monkeypatch):
    asked = []

    def get_int(parent, title, text, value, minimum, maximum, step):
        asked.append((value, minimum, maximum))
        return 3, True

    monkeypatch.setattr(triage_tab.QInputDialog, "getInt", get_int)
    assert triage_tab.ask_batch_size(None, 7, "") == 3
    assert asked == [(7, 1, 7)]  # defaults to all of them


def test_progress_updates_count_eta_and_rows_live(qapp, configured, stored, monkeypatch):
    tab = make_tab(stored)
    tab.refresh()
    wait_until_idle(qapp, tab)
    before = tab.table.rowCount()
    tab.table.selectRow(0)
    reading = tab._selected_row()["review"]["review_id"]

    tab._batch = {"total": 3, "started": time.monotonic() - 10, "subjects": {"77": "New one"}}
    tab.busy_bar.setRange(0, 3)
    summary = {"classified": 1, "manual_fallback": 0, "skipped": 0, "errored": 0}
    tab._on_batch_progress(
        {"zendesk_id": "77", "outcome": "classified", "review": fake_review(),
         "done": 1, "total": 3, "summary": summary}
    )

    assert tab.table.rowCount() == before + 1
    assert tab.busy_bar.value() == 1
    text = tab.batch_label.text()
    assert "1 of 3 done" in text and "1 classified" in text and "about 20 s left" in text
    # The review being read stays selected while rows arrive.
    assert tab._selected_row()["review"]["review_id"] == reading
    subjects = [tab.table.item(r, 1).text() for r in range(tab.table.rowCount())]
    assert "New one" in subjects


def test_stop_finishes_the_current_ticket_then_stops(qapp, configured, dialogs, queue, monkeypatch):
    import threading

    from decision_engine import persistence

    started, release, reviewed = threading.Event(), threading.Event(), []

    def slow_process(zendesk_id, tickets, worker=None):
        reviewed.append(zendesk_id)
        started.set()
        release.wait(5)  # the model call, still in flight when Stop is pressed
        return {**fake_review(), "zendesk_id": zendesk_id}

    monkeypatch.setattr(persistence, "process_one_ticket", slow_process)
    tab = make_tab(queue)
    tab.start_batch()
    try:
        deadline = time.monotonic() + 5
        while not started.is_set():
            qapp.processEvents()
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert not tab.stop_button.isHidden() and tab.stop_button.isEnabled()

        tab.stop_batch()
        assert not tab.stop_button.isEnabled()
        assert "stopping after this ticket" in tab.batch_label.text()
    finally:
        release.set()  # never leave the worker thread blocked
        wait_until_idle(qapp, tab)

    assert reviewed == ["10"]  # finished the one in flight, started no other
    assert "Batch stopped after the current ticket" in dialogs[-1][1]
    assert "Cancelled before starting: 3" in dialogs[-1][1]
    assert tab.stop_button.isHidden()


@pytest.mark.parametrize(
    "seconds, text",
    [(0, "0 s"), (44.6, "45 s"), (60, "1 min"), (200, "3 min 20 s"), (3900, "1 h 5 min"), (7200, "2 h")],
)
def test_format_duration(seconds, text):
    assert triage_tab.format_duration(seconds) == text


def test_batch_status_has_no_estimate_before_the_first_ticket():
    counts = {"classified": 0, "manual_fallback": 0, "skipped": 0, "errored": 0}
    assert triage_tab.batch_status({"done": 0, "total": 5, "summary": counts}, 3.0) == "0 of 5 done"


# --- A/B test: two configurations, reviewed blind ------------------------------------


def test_a_decision_answer_is_shown_blind_with_its_probabilities():
    text = ab_test_dialog.format_output(
        {
            "classification": "flag_solve",
            "confidence": 82,
            "decision_only": True,
            "probabilities": {"manual_triage": 0.18, "flag_solve": 0.82},
            "reasoning": "Classification only.",
            "model_version": "jev-1.13.0",
            "guard_overrides": [],
        }
    )
    assert text.splitlines()[:2] == ["Classification: flag_solve", "Confidence: 82"]
    assert "Classification only" in text
    assert text.index("flag_solve: 82%") < text.index("manual_triage: 18%")
    assert "jev" not in text and "Reasoning:" not in text


@pytest.fixture
def ab_tickets(tickets):
    tickets.insert_many([copy.deepcopy(t) for t in CATEGORY_FIXTURES.values()])
    return tickets


def ab_engine(payload, reengage_count=0, assignee=None, config=None, **kwargs):
    """B solves every ticket; A sends every ticket to manual triage."""
    classification = "flag_solve" if config.primary == "new-model" else "manual_triage"
    return {
        "classification": classification,
        "confidence": 80,
        "signals_matched": [],
        "reasoning": f"picked {classification}",
        "internal_note": "",
        "model_version": config.primary,
        "audit": {"usage": {"output_tokens": 5}},
    }


def open_ab_test(qapp, tab, monkeypatch, b_primary="new-model"):

    class Accepting(ab_test_dialog.NewRunDialog):
        def exec_(self):
            self.b_form.primary.setText(b_primary)
            self.size.setValue(3)
            return ab_test_dialog.QDialog.Accepted if self.configs()[0] != self.configs()[1] else 0

    monkeypatch.setattr(ab_test_dialog, "NewRunDialog", Accepting)
    tab.open_ab_test()
    dialog = tab.ab_test_dialog
    dialog.classify = ab_engine
    dialog.price_source = lambda: {
        "new-model": pricing.Price(input=10, output=20),
        "test": pricing.Price(input=1, output=2),
    }
    wait_until_idle(qapp, tab)
    return dialog


def test_ab_test_creates_classifies_and_reviews_blind(
    qapp, configured, dialogs, ab_tickets, monkeypatch, tmp_path
):
    from decision_engine import ab_test

    before = list(ab_tickets.find())
    tab = make_tab(ab_tickets)
    dialog = open_ab_test(qapp, tab, monkeypatch)
    assert dialog.run_table.rowCount() == 0

    dialog.new_run()  # creates, asks "classify now?" (yes), classifies, opens it
    for _ in range(4):
        wait_until_idle(qapp, tab)
    assert dialogs[-1][0] == "question"
    assert "6 answer(s) to get: at least 6 paid model call(s)" in dialogs[-1][1]
    assert dialog.run_table.rowCount() == 1
    assert dialog.run_table.item(0, 3).text() == "complete"
    assert "ticket 1 of 3" in dialog.position_label.text()
    assert list(ab_tickets.find()) == before  # the tickets are untouched

    # A fixed mix of orders: each side must show the variant its ticket's
    # stored order names, not always A first.
    runs, results = ab_test.get_collections(ab_tickets.database)
    assert runs.find_one()["prices"]["new-model"]["input"] == 10.0
    orders = [["B", "A"], ["A", "B"], ["B", "A"]]
    ids = sorted(d["ticket_id"] for d in results.find())
    for ticket_id, order in zip(ids, orders):
        results.update_one({"ticket_id": ticket_id}, {"$set": {"order": order}})
    # The review shows the snapshot taken at creation, not the live ticket.
    first_subject = ab_tickets.find_one({"zendesk_id": ids[0]})["subject"]
    ab_tickets.update_one({"zendesk_id": ids[0]}, {"$set": {"subject": "Edited later"}})
    dialog.open_selected()
    wait_until_idle(qapp, tab)

    answers = {"A": "Classification: manual_triage", "B": "Classification: flag_solve"}
    for position, order in enumerate(orders):
        dialog.position = position
        dialog._show_current()
        texts = [view.toPlainText() for view in dialog.output_views]
        assert [t.splitlines()[0] for t in texts] == [answers[v] for v in order]
        assert all("Cost: " in t for t in texts)
        assert not any("new-model" in t or "current" in t for t in texts)
    dialog.position = 0
    dialog._show_current()
    assert first_subject in dialog.ticket_view.toPlainText()
    assert "Edited later" not in dialog.ticket_view.toPlainText()

    # Output 1 on the first ticket is B's answer (order B, A).
    dialog.record("1")
    wait_until_idle(qapp, tab)
    saved = results.find_one({"ticket_id": ids[0]})["preference"]
    assert saved["choice"] == "B"
    # Recorded as the computer login, not a typed-in name.
    assert saved["reviewer"] == ab_test_dialog.reviewer_identity()
    assert not hasattr(dialog, "reviewer_input")
    assert "ticket 2 of 3" in dialog.position_label.text()
    reviewed_column = ab_test_dialog.RUN_COLUMNS.index("Reviewed")
    assert dialog.run_table.item(0, reviewed_column).text() == "1"

    # Another reviewer saved a verdict on ticket 2 meanwhile: ours is refused
    # with a message, and the reload shows theirs.
    results.update_one(
        {"ticket_id": ids[1]},
        {
            "$set": {
                "preference": {"choice": "tie", "reviewer": "other", "note": ""},
                "preference_version": 1,
            }
        },
    )
    dialog.record("2")
    for _ in range(2):
        wait_until_idle(qapp, tab)
    assert dialogs[-1][0] == "critical" and "changed by other" in dialogs[-1][1]
    assert results.find_one({"ticket_id": ids[1]})["preference"]["reviewer"] == "other"
    assert dialog.verdict_label.text() == "Your verdict: Both equally good (other)"
    assert dialog.run_table.item(0, reviewed_column).text() == "2"

    dialog.only_unreviewed.setChecked(True)
    assert len(dialog.visible()) == 1
    dialog.record("tie")
    wait_until_idle(qapp, tab)
    assert len(dialog.visible()) == 0

    # That was the last verdict: the test finalizes itself.
    for _ in range(3):
        wait_until_idle(qapp, tab)
    assert "finalized" in dialog.status_label.text()
    status_column = ab_test_dialog.RUN_COLUMNS.index("Status")
    assert dialog.run_table.item(0, status_column).text() == "finalized"
    verdicts = ab_tickets.find_one({"zendesk_id": ids[0]})["ab_test_verdicts"]
    assert [v["choice"] for v in verdicts] == ["B"]
    assert "payload" not in results.find_one({"ticket_id": ids[0]})
    dialog.only_unreviewed.setChecked(False)  # every ticket is reviewed now
    assert not any(b.isEnabled() for b in dialog.choice_buttons.values())
    assert "removed when this test was finalized" in dialog.ticket_view.toPlainText()

    dialog.show_results()
    wait_until_idle(qapp, tab)
    summary = dialog.summary_dialog.findChild(triage_tab.QTextEdit).toPlainText()
    assert "B better" in summary and "B: primary new-model" in summary

    export = tmp_path / "run.csv"
    monkeypatch.setattr(
        ab_test_dialog.QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (str(export), ""),
    )
    dialog.export_selected()
    wait_until_idle(qapp, tab)
    assert export.exists() and "B_classification" in export.read_text()
    assert str(export) in dialog.status_label.text()


def test_ab_test_waits_for_another_job(qapp, configured, dialogs, ab_tickets, monkeypatch):
    tab = make_tab(ab_tickets)
    dialog = open_ab_test(qapp, tab, monkeypatch)
    tab._thread = object()  # another job running
    try:
        dialog.load_runs()
        assert "Another Decision Engine job" in dialog.status_label.text()
    finally:
        tab._thread = None


def test_ab_test_needs_the_model_settings_to_classify(
    qapp, monkeypatch, dialogs, ab_tickets
):
    for name in triage_tab.REVIEW_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MONGO_URI", "mongodb://test")
    tab = make_tab(ab_tickets)
    dialog = open_ab_test(qapp, tab, monkeypatch)
    dialog.new_run()
    assert dialogs[-1][0] == "warning" and "Not configured" in dialogs[-1][1]
    assert dialog.run_table.rowCount() == 0


def test_the_price_table_starts_live_and_marks_edits(qapp, monkeypatch):
    from decision_engine.classify import EngineConfig

    shown = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: shown.append(a[2]))
    live = {"opus": pricing.Price(input=2.9, output=14.5, cache_read=0.145)}
    dialog = ab_test_dialog.NewRunDialog(EngineConfig("gpt"), None, live)
    dialog.b_form.primary.setText("opus")
    table = dialog.price_table
    assert [table.item(r, 0).text() for r in range(table.rowCount())] == ["gpt", "opus"]
    assert table.item(1, 1).text() == "2.9" and table.item(1, 5).text() == "live"
    assert table.item(0, 5).text() == "not listed: enter it"
    assert set(dialog.prices()) == {"opus"}

    table.item(0, 1).setText("0.065")
    table.item(0, 2).setText("0.325")
    assert table.item(0, 5).text() == "entered"
    prices = dialog.prices()
    assert prices["gpt"] == pricing.Price(input=0.065, output=0.325, source="entered")
    assert prices["opus"].source == "live"

    # An edit survives the model list changing.
    dialog.b_form.confirmation.setText("checker")
    assert dialog.prices()["gpt"].input == 0.065

    table.item(1, 2).setText("lots")
    dialog.accept()
    assert shown and "isn't a number" in shown[-1]


def test_each_side_shows_its_cost():
    text = ab_test_dialog.format_output(
        {
            "classification": "flag_solve",
            "confidence": 90,
            "cost_usd": 0.0123,
            "cost_basis": "billed",
            "model_calls": 2,
        }
    )
    assert "Cost: $0.01 (billed, 2 model calls)" in text
    reused = ab_test_dialog.format_output(
        {
            "classification": "flag_solve",
            "confidence": 90,
            "cost_usd": 0.0001,
            "cost_basis": "estimated",
            "model_calls": 1,
            "reused_from": {"run_id": "x"},
        }
    )
    assert "reused from an earlier test, so nothing was spent here" in reused


def test_entered_prices_must_be_finite(qapp, monkeypatch):
    from decision_engine.classify import EngineConfig

    shown = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: shown.append(a[2]))
    dialog = ab_test_dialog.NewRunDialog(EngineConfig("gpt"), None, {})
    dialog.b_form.primary.setText("other")  # A and B must differ
    dialog.price_table.item(0, 1).setText("inf")
    dialog.price_table.item(0, 2).setText("1")
    dialog.accept()
    assert shown and "finite" in shown[-1]


def test_the_ticket_view_reads_the_payload_both_sides_were_sent():
    from decision_engine.payload import build_ticket_payload

    ticket = CATEGORY_FIXTURES["flag_reengage"]
    result = {
        "ticket_id": ticket["zendesk_id"],
        "ticket": {"zendesk_id": ticket["zendesk_id"], "subject": "S", "status": "open"},
        "payload": build_ticket_payload(ticket),
    }
    text = ab_test_dialog.format_ticket(result)
    assert "S" in text and ticket["transcript"][0]["author"] in text
    finalized = {"ticket_id": "1", "ticket": {"status": "open"}}
    assert "removed when this test was finalized" in ab_test_dialog.format_ticket(finalized)


def test_interrupted_calls_are_retried_only_when_asked(
    qapp, configured, dialogs, ab_tickets, monkeypatch
):
    from decision_engine import ab_test

    tab = make_tab(ab_tickets)
    dialog = open_ab_test(qapp, tab, monkeypatch)
    runs, results = ab_test.get_collections(ab_tickets.database)
    ids = sorted(t["zendesk_id"] for t in ab_tickets.find())[:1]
    run_id = ab_test.create_run(
        ab_tickets,
        runs,
        results,
        *(ab_test.EngineConfig(m) for m in ("test", "new-model")),
        ids,
    )
    long_ago = datetime.now(timezone.utc) - timedelta(hours=2)
    results.update_one(
        {"run_id": run_id},
        {"$set": {"started": {"B": {"worker": "x", "at": long_ago}}}},
    )
    dialog.load_runs()
    wait_until_idle(qapp, tab)
    dialog.run_table.selectRow(0)
    retried = []
    monkeypatch.setattr(
        dialog, "_classify", lambda run_id, retry_interrupted=False: retried.append(retry_interrupted)
    )
    dialog.classify_selected()
    wait_until_idle(qapp, tab)
    questions = [text for kind, text in dialogs if kind == "question"]
    assert "may already have been charged" in questions[-2]
    assert retried == [True]  # the stub answers Yes


def test_reviewing_needs_the_arbiter_model():
    assert "CHEAPER_INFERENCE_ARBITER_MODEL_NAME" in triage_tab.REVIEW_ENV
    assert "CHEAPER_INFERENCE_ARBITER_MODEL_NAME" not in triage_tab.OPTIONAL_MODEL_ENV
    assert "CHEAPER_INFERENCE_ARBITER_MODEL_NAME" not in triage_tab.AB_TEST_ENV


@pytest.mark.parametrize("arbiter", [None, "   "])
def test_review_all_refuses_without_the_arbiter(
    qapp, configured, monkeypatch, dialogs, stored, arbiter
):
    if arbiter is None:
        monkeypatch.delenv("CHEAPER_INFERENCE_ARBITER_MODEL_NAME")
    else:
        monkeypatch.setenv("CHEAPER_INFERENCE_ARBITER_MODEL_NAME", arbiter)
    calls = []
    tab = make_tab(stored, review_batch=lambda tickets, ids, **kw: calls.append(tickets))

    tab.start_batch()
    wait_until_idle(qapp, tab)

    assert calls == []
    assert dialogs[0][0] == "warning"
    assert "CHEAPER_INFERENCE_ARBITER_MODEL_NAME" in tab.status_label.text()


def test_ab_test_does_not_need_the_arbiter(qapp, configured, monkeypatch, dialogs, ab_tickets):
    monkeypatch.delenv("CHEAPER_INFERENCE_ARBITER_MODEL_NAME")
    tab = make_tab(ab_tickets)
    dialog = open_ab_test(qapp, tab, monkeypatch)
    dialog.new_run()
    wait_until_idle(qapp, tab)
    assert not any(kind == "warning" for kind, _ in dialogs)
    assert dialog.run_table.rowCount() == 1


def test_review_text_names_each_trigger():
    text = triage_tab.format_review(
        {"zendesk_id": "1", "review": {"signals_matched": ["L6"]}}
    )
    assert "  - L6 — Genuine churn risk" in text


# --- manual-triage cases ---------------------------------------------------------

from decision_engine import evidence, triage  # noqa: E402
from test_triage_cases import JIRA_XML  # noqa: E402
from ui.dialogs.evidence_dialog import EvidenceDialog  # noqa: E402


@pytest.fixture
def with_case(stored):
    """A manual-triage case for the flag_solve ticket's first review. The
    table shows that ticket's newer flag_solve review, which carries it."""
    cases = triage.get_cases_collection(stored.database)
    cases.insert_one(
        {
            "zendesk_id": CATEGORY_FIXTURES["flag_solve"]["zendesk_id"],
            "review_id": "r-manual_triage-60",
            "status": triage.OPEN,
            "route": triage.GUARD_OVERRIDE,
            "answers": [],
            "arbitration": None,
            "open_questions": ["Has LRN-1234 been released?"],
            "trail": {"stored": "inline", "trail": {}},
            "evidence": [],
            "reruns": [],
        }
    )
    return cases


def select_classification(tab, classification):
    column = triage_tab.COLUMNS.index("Classification")
    for row in range(tab.table.rowCount()):
        if tab.table.item(row, column).text() == classification:
            tab.table.selectRow(row)
            return
    raise AssertionError(classification)


def test_a_case_is_shown_with_its_controls(qapp, configured, stored, with_case):
    tab = make_tab(stored)
    tab.refresh()
    wait_until_idle(qapp, tab)

    select_classification(tab, "flag_solve")
    text = tab.detail.toPlainText()
    assert "Manual-triage case (open): a code check downgraded the model's answer" in text
    assert "  - Has LRN-1234 been released?" in text
    assert "(none yet: use Add Evidence…)" in text
    assert tab.add_evidence_button.isEnabled()
    assert not tab.rerun_button.isEnabled()
    assert "Add Jira or Aha evidence" in tab.rerun_button.toolTip()

    select_classification(tab, "flag_reengage")
    assert not tab.add_evidence_button.isEnabled()


def test_evidence_and_a_rerun_from_the_tab(qapp, configured, monkeypatch, dialogs, stored, with_case):
    found = evidence.extract(evidence.JIRA, JIRA_XML, "export.xml")

    class Accepting:
        Accepted = EvidenceDialog.Accepted

        def __init__(self, *args):
            self.evidence = found

        def exec_(self):
            return EvidenceDialog.Accepted

    monkeypatch.setattr(triage_tab, "EvidenceDialog", Accepting)
    reruns = []
    monkeypatch.setattr(
        triage,
        "rerun_case",
        lambda tickets, cases, case_id: reruns.append(case_id)
        or {"classification": "flag_solve", "confidence": 90},
    )
    tab = make_tab(stored)
    tab.refresh()
    wait_until_idle(qapp, tab)
    select_classification(tab, "flag_solve")

    tab.add_evidence_to_selected()
    wait_until_idle(qapp, tab)

    case = with_case.find_one({})
    assert [e["data"]["issues"][0]["key"] for e in case["evidence"]] == ["LRN-1234"]
    select_classification(tab, "flag_solve")
    assert "Jira LRN-1234 (Released, Fixed), added by" in tab.detail.toPlainText()
    assert tab.rerun_button.isEnabled()

    tab.rerun_selected()
    wait_until_idle(qapp, tab)

    assert reruns == [case["_id"]]
    assert dialogs[-1][0] == "question" and "paid model call" in dialogs[-1][1]


def test_the_evidence_dialog_refuses_what_it_cant_parse(qapp, monkeypatch, dialogs):
    from ui.dialogs import evidence_dialog

    monkeypatch.setattr(
        evidence_dialog,
        "confirm_plain",
        lambda parent, title, text: dialogs.append(("question", text)) or True,
    )
    dialog = EvidenceDialog("90001")
    assert dialog.error_label.textFormat() == QtCore.Qt.PlainText
    dialog.load_file("broken.xml", b"<rss><channel>")
    dialog.check_and_add()
    assert dialog.evidence is None
    assert dialog.error_label.text().startswith("Can't add this: The XML isn't well-formed")

    dialog.paste_box.setPlainText(JIRA_XML)
    assert dialog.file_label.text() == "No file chosen."
    dialog.check_and_add()
    assert dialog.evidence.data["issues"][0]["key"] == "LRN-1234"
    assert dialog.evidence.source == "pasted"
    assert "Found: Jira LRN-1234" in dialogs[-1][1]


def test_rerun_is_disabled_while_another_rerun_holds_the_case(qapp, configured, stored, with_case):
    found = evidence.extract(evidence.JIRA, JIRA_XML, "export.xml")
    case = with_case.find_one({})
    triage.add_evidence(with_case, case["_id"], found.kind, found.data, found.source, found.sha256, "r")
    triage._claim(with_case, case["_id"], "another-worker")
    tab = make_tab(stored)
    tab.refresh()
    wait_until_idle(qapp, tab)

    select_classification(tab, "flag_solve")
    assert not tab.rerun_button.isEnabled()
    assert tab.rerun_button.toolTip() == "A re-run of this case is in progress."


def test_retry_trail_uploads_reports_a_failure_and_a_success(qapp, configured, monkeypatch, dialogs, stored, with_case):
    with_case.update_one({}, {"$set": {"trail.upload_error": "Couldn't upload"}})
    tab = make_tab(stored)
    tab.refresh()
    wait_until_idle(qapp, tab)
    select_classification(tab, "flag_solve")
    assert "its S3 upload failed: use Retry Trail Uploads" in tab.detail.toPlainText()

    def refuse(cases, tickets=None):
        raise triage.TrailUploadFailed("Couldn't upload s3://models/x.json (closed).")

    monkeypatch.setattr(triage, "upload_kept_trails", refuse)
    tab.retry_trail_uploads()
    wait_until_idle(qapp, tab)
    assert dialogs[-1][0] == "warning"
    assert "1 case(s) still have a trail kept in MongoDB" in dialogs[-1][1]

    monkeypatch.setattr(triage, "upload_kept_trails", lambda cases, tickets=None: 1)
    done = []
    monkeypatch.setattr(tab, "_feedback_done", done.append)
    tab.retry_trail_uploads()
    wait_until_idle(qapp, tab)
    assert done == ["Moved 1 reasoning trail(s) to S3."]


def test_a_reruns_failed_upload_is_shown():
    lines = triage_tab.format_case(
        {
            "status": triage.OPEN,
            "route": triage.ABSTAINED,
            "trail": {"stored": "s3", "bucket": "b", "key": "k"},
            "reruns": [
                {"classification": "manual_triage", "confidence": 40, "model": "arbiter",
                 "trail": {"stored": "inline", "trail": {}, "upload_error": "closed"}},
            ],
        }
    )
    assert "  Re-run 1's reasoning trail is kept in MongoDB (its S3 upload failed: use Retry Trail Uploads)" in lines


def test_a_rerun_opens_its_arbitration_results(qapp, configured, stored):
    zendesk_id = CATEGORY_FIXTURES["flag_reengage"]["zendesk_id"]
    stored.update_one(
        {"zendesk_id": zendesk_id},
        {"$push": {"agent_reviews": review("flag_solve", 1, rerun_of="r-flag_reengage-30")}},
    )
    tab = make_tab(stored)
    tab._results_for = zendesk_id  # as rerun_selected's done() sets it
    tab.refresh()
    wait_until_idle(qapp, tab)

    assert tab._results_for is None
    assert tab._results_dialog is not None and tab._results_dialog.isVisible()
    text = tab._results_dialog.text.toPlainText()
    assert "Yes: flag_reengage (confidence 80, test-model) became flag_solve" in text
    assert "Solve the ticket in Zendesk." in text
    tab._results_dialog.close()

    # The button is offered only for the re-run ticket's row.
    for row in range(tab.table.rowCount()):
        tab.table.selectRow(row)
        is_rerun = tab._selected_row()["zendesk_id"] == zendesk_id
        assert tab.results_button.isEnabled() == is_rerun


def test_an_older_open_case_stays_reachable_when_the_newest_is_resolved(
    qapp, configured, stored, with_case
):
    # The ticket's newest review re-ran a second case and resolved it; the
    # first case (with_case's) is still open.
    open_id = with_case.find_one({})["_id"]
    with_case.insert_one(
        {
            "zendesk_id": CATEGORY_FIXTURES["flag_solve"]["zendesk_id"],
            "review_id": "r-other",
            "status": triage.RESOLVED,
            "route": triage.GUARD_OVERRIDE,
            "open_questions": [],
            "trail": {"stored": "inline", "trail": {}},
            "evidence": [],
            "reruns": [{"review_id": "r-flag_solve-5", "classification": "flag_solve", "confidence": 90}],
        }
    )
    tab = make_tab(stored)
    tab.refresh()
    wait_until_idle(qapp, tab)
    select_classification(tab, "flag_solve")

    row = tab._selected_row()
    assert row["case"]["review_id"] == "r-other"
    assert triage_tab.actionable_case(row)["_id"] == open_id
    assert tab.add_evidence_button.isEnabled()
    text = tab.detail.toPlainText()
    assert "Manual-triage case (resolved)" in text
    assert "Another manual-triage case on this ticket is still open:" in text
    assert "  - Has LRN-1234 been released?" in text


def test_reviews_with_the_same_time_show_the_later_one(tickets):
    from bson import ObjectId

    earlier, later = ObjectId(), ObjectId()
    add_reviewed(
        tickets,
        "tie",
        # Stored older first, so only the tie-break puts the later review first.
        review("manual_triage", 5, review_id=earlier),
        review("flag_solve", 5, review_id=later),
    )
    (row,) = recent_reviews(tickets)
    assert row["review"]["review_id"] == later
    assert [e["review_id"] for e in row["earlier_reviews"]] == [earlier]


def test_reopening_results_deletes_the_previous_dialog(qapp, configured, stored):
    from PyQt5 import sip

    tab = make_tab(stored)
    row = {"zendesk_id": "1", "review": {"classification": "flag_solve", "rerun_of": "x"}}
    tab.show_results(row)
    first = tab._results_dialog
    tab.show_results(row)
    QApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
    assert sip.isdeleted(first)
    assert tab._results_dialog is not first and tab._results_dialog.isVisible()
    tab._results_dialog.close()
