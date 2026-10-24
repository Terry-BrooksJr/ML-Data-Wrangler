import json
import os
import stat
import sys
import threading

import pytest
import spend_guard

pytestmark = pytest.mark.real_spend_guard


def write_access(policy: dict) -> None:
    folder = spend_guard.ensure_secrets_dir()
    (folder / spend_guard.ACCESS_FILE).write_text(json.dumps(policy))


def the_pin() -> str:
    return spend_guard.ensure_pin().read_text().strip()


def audit_events() -> list:
    path = spend_guard.audit_path()
    return [json.loads(line)["event"] for line in path.read_text().splitlines()]


def test_pin_file_is_owner_only_and_eight_digits():
    path = spend_guard.ensure_pin()
    pin = path.read_text().strip()
    assert len(pin) == spend_guard.PIN_DIGITS and pin.isdigit()
    if sys.platform != "win32":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700


def test_pin_is_reused_until_it_expires():
    first = the_pin()
    assert the_pin() == first
    path = spend_guard.pin_path()
    old = path.stat().st_mtime - (spend_guard.DEFAULT_PIN_TTL_HOURS + 1) * 3600
    os.utime(path, (old, old))
    spend_guard.ensure_pin()
    assert path.stat().st_mtime > old + 3600
    assert audit_events().count("pin_rotated") == 2


@pytest.mark.skipif(sys.platform == "win32", reason="mode bits")
def test_folder_others_can_write_is_refused():
    folder = spend_guard.ensure_secrets_dir()
    folder.chmod(0o777)
    with pytest.raises(spend_guard.InsecureSecretsDir):
        spend_guard.ensure_pin()


def test_paid_call_without_grant_is_refused_in_the_app(monkeypatch):
    spend_guard.gui_mode()
    with pytest.raises(spend_guard.SpendNotAuthorized):
        spend_guard.require(spend_guard.CHEAPER_INFERENCE, "classify")
    assert "denied" in audit_events()


def test_grant_covers_only_its_thread_and_services():
    grant = spend_guard.authorize(the_pin(), "test", [spend_guard.VOYAGE])
    spend_guard.gui_mode()
    errors = []

    def other_thread():
        try:
            spend_guard.require(spend_guard.VOYAGE)
        except spend_guard.SpendNotAuthorized as e:
            errors.append(e)

    def task():
        spend_guard.require(spend_guard.VOYAGE, "embed")
        with pytest.raises(spend_guard.SpendNotAuthorized):
            spend_guard.require(spend_guard.OPENAI)
        thread = threading.Thread(target=other_thread)
        thread.start()
        thread.join()
        return "done"

    assert grant.run(task) == "done"
    assert grant.calls == 1 and len(errors) == 1
    with pytest.raises(spend_guard.SpendNotAuthorized):  # ended with the job
        spend_guard.require(spend_guard.VOYAGE)
    events = audit_events()
    assert {"pin_accepted", "grant_started", "paid_call", "grant_ended"} <= set(events)


def test_pin_is_never_written_to_the_audit_log():
    pin = the_pin()
    spend_guard.authorize(pin, "test")
    with pytest.raises(spend_guard.WrongPin):
        spend_guard.authorize("0", "test")
    assert pin not in spend_guard.audit_path().read_text()


def test_wrong_pins_lock_out_and_replace_the_pin():
    pin = the_pin()
    for remaining in range(spend_guard.MAX_BAD_PINS - 1, 0, -1):
        with pytest.raises(spend_guard.WrongPin) as wrong:
            spend_guard.check_pin("not it")
        assert wrong.value.remaining == remaining
    with pytest.raises(spend_guard.LockedOut):
        spend_guard.check_pin("not it")
    assert the_pin() != pin
    with pytest.raises(spend_guard.LockedOut):  # even the new, right PIN
        spend_guard.check_pin(the_pin())


def test_right_pin_resets_the_wrong_count():
    for _ in range(spend_guard.MAX_BAD_PINS - 1):
        with pytest.raises(spend_guard.WrongPin):
            spend_guard.check_pin("nope")
    spend_guard.check_pin(the_pin())
    with pytest.raises(spend_guard.WrongPin) as wrong:
        spend_guard.check_pin("nope")
    assert wrong.value.remaining == spend_guard.MAX_BAD_PINS - 1


def test_without_access_file_everyone_is_admin():
    assert spend_guard.role() == "admin"
    assert spend_guard.can(spend_guard.MANAGE_SECURITY)


def test_roles_from_access_file(monkeypatch):
    monkeypatch.setattr(spend_guard, "identity", lambda: "sam@box")
    write_access({"users": {"sam": "reviewer", "ana@box": "operator"}})
    assert spend_guard.role() == "reviewer"
    assert spend_guard.role("ana@box") == "operator"
    assert spend_guard.role("someone@else") == "viewer"
    spend_guard.require_permission(spend_guard.REVIEW, "review")
    with pytest.raises(spend_guard.PermissionDenied):
        spend_guard.authorize(the_pin(), "batch")
    with pytest.raises(spend_guard.PermissionDenied):
        spend_guard.require_permission(spend_guard.CHANGE_SETTINGS, "change settings")


@pytest.mark.parametrize(
    "policy", ["not json", json.dumps({"users": {"x": "superuser"}}), json.dumps([])]
)
def test_broken_access_file_grants_nothing(policy):
    folder = spend_guard.ensure_secrets_dir()
    (folder / spend_guard.ACCESS_FILE).write_text(policy)
    assert spend_guard.role() == "viewer"


def test_daily_call_limit():
    write_access({"default_role": "operator", "daily_paid_call_limit": 2})
    grant = spend_guard.authorize(the_pin(), "test")

    def task():
        spend_guard.require(spend_guard.VOYAGE)
        spend_guard.require(spend_guard.VOYAGE)
        with pytest.raises(spend_guard.DailyLimitReached):
            spend_guard.require(spend_guard.VOYAGE)

    grant.run(task)


def test_command_line_reads_the_pin_file(monkeypatch):
    monkeypatch.setenv(spend_guard.PIN_FILE_ENV, str(spend_guard.ensure_pin()))
    spend_guard.require(spend_guard.CHEAPER_INFERENCE, "classify")
    spend_guard.require(spend_guard.OPENAI, "embed")  # one grant per process
    assert audit_events().count("pin_accepted") == 1


def test_command_line_without_terminal_or_pin_file_is_refused(monkeypatch):
    monkeypatch.setattr(sys, "stdin", None)
    with pytest.raises(spend_guard.SpendNotAuthorized):
        spend_guard.require(spend_guard.CHEAPER_INFERENCE)


def test_classify_refuses_without_grant(monkeypatch):
    from decision_engine import classify

    spend_guard.gui_mode()
    monkeypatch.setattr(
        classify, "get_client", lambda: pytest.fail("the request was built")
    )
    with pytest.raises(spend_guard.SpendNotAuthorized):
        classify._classify_with("model", "{}")


def test_pin_dialog_asks_again_after_a_wrong_pin(monkeypatch):
    import pin_dialog

    answers = iter([("wrong", True), (the_pin(), True)])
    monkeypatch.setattr(
        pin_dialog.QInputDialog, "getText", lambda *args, **kwargs: next(answers)
    )
    grant = pin_dialog.ask_for_grant(
        None, "Reviewing 3 ticket(s)", [spend_guard.VOYAGE]
    )
    assert grant.services == {spend_guard.VOYAGE}
    assert audit_events().count("pin_rejected") == 1


def test_pin_dialog_refuses_a_viewer(monkeypatch):
    import pin_dialog

    write_access({"default_role": "viewer"})
    warnings = []
    monkeypatch.setattr(
        pin_dialog.QMessageBox, "warning", lambda *args: warnings.append(args[2])
    )
    monkeypatch.setattr(
        pin_dialog.QInputDialog,
        "getText",
        lambda *a, **k: pytest.fail("asked for a PIN"),
    )
    assert (
        pin_dialog.ask_for_grant(None, "Embedding tickets", [spend_guard.VOYAGE])
        is None
    )
    assert "viewer" in warnings[0]
