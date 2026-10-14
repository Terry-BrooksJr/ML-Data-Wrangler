import json
import os
import sys

import pytest

import settings


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "config_dir", lambda: tmp_path)
    for s in settings.SETTINGS:
        monkeypatch.delenv(s.name, raising=False)
    return tmp_path / "settings.json"


def test_save_round_trips_awkward_values(config, monkeypatch):
    uri = "mongodb://u:p'x\"é#\\@h/?a=1"
    settings.save({"MONGO_URI": uri})
    assert os.environ["MONGO_URI"] == uri

    monkeypatch.delenv("MONGO_URI")
    monkeypatch.setattr(settings, "is_frozen", lambda: True)
    settings.load()
    assert os.environ["MONGO_URI"] == uri


def test_blank_value_removes_setting(config):
    settings.save({"MONGO_DB": "other", "MONGO_URI": "mongodb://h"})
    settings.save({"MONGO_DB": "  ", "MONGO_URI": "mongodb://h"})
    assert "MONGO_DB" not in os.environ
    assert json.loads(config.read_text()) == {"MONGO_URI": "mongodb://h", "MONGO_DB": None}


def test_restart_needed_only_for_changed_import_time_values(config):
    assert settings.save({"LESSONS_COLLECTION": "x", "MONGO_DB": "y"}) == ["LESSONS_COLLECTION"]
    assert settings.save({"LESSONS_COLLECTION": "x", "MONGO_DB": "z"}) == []


def test_missing_required(config, monkeypatch):
    required = [s.name for s in settings.SETTINGS if s.required]
    assert settings.missing_required() == required
    for name in required:
        monkeypatch.setenv(name, "v")
    assert settings.missing_required() == []


def test_saved_settings_override_shell(config, monkeypatch):
    settings.save({"MONGO_DB": "from_settings"})
    monkeypatch.setenv("MONGO_DB", "from_shell")
    monkeypatch.setattr(settings, "is_frozen", lambda: True)
    settings.load()
    assert os.environ["MONGO_DB"] == "from_settings"


def test_failed_write_keeps_previous_settings(config, monkeypatch):
    settings.save({"MONGO_URI": "mongodb://old"})

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(os, "fsync", fail)
    with pytest.raises(OSError):
        settings.save({"MONGO_URI": "mongodb://new"})
    assert json.loads(config.read_text()) == {"MONGO_URI": "mongodb://old"}
    assert os.environ["MONGO_URI"] == "mongodb://old"
    assert [p.name for p in config.parent.iterdir()] == ["settings.json"]


@pytest.mark.skipif(os.name != "posix", reason="mode bits don't protect files on Windows; see the ACL test")
def test_settings_file_is_owner_only_on_posix(config):
    settings.save({"MONGO_URI": "mongodb://h"})
    assert oct(config.stat().st_mode & 0o777) == "0o600"


def test_folder_and_file_are_restricted_before_secrets_are_written(config, monkeypatch):
    restricted = []

    def record(path):
        contents = path.read_text() if path.is_file() else None
        restricted.append((path, contents))

    monkeypatch.setattr(settings, "_restrict_to_owner", record)
    settings.save({"MONGO_URI": "mongodb://secret"})
    assert restricted[0] == (config.parent, None)
    tmp, contents = restricted[1]
    assert tmp.parent == config.parent and contents == ""


def test_failed_acl_change_keeps_previous_settings(config, monkeypatch):
    settings.save({"MONGO_URI": "mongodb://old"})

    def fail(path):
        if path.is_file():
            raise OSError("access denied")

    monkeypatch.setattr(settings, "_restrict_to_owner", fail)
    with pytest.raises(OSError):
        settings.save({"MONGO_URI": "mongodb://new"})
    assert json.loads(config.read_text()) == {"MONGO_URI": "mongodb://old"}
    assert [p.name for p in config.parent.iterdir()] == ["settings.json"]


@pytest.mark.skipif(sys.platform != "win32", reason="Windows ACLs")
def test_windows_acl_is_owner_only(config):
    import subprocess

    settings.save({"MONGO_URI": "mongodb://h"})
    acl = subprocess.run(["icacls", str(config)], capture_output=True, text=True, check=True).stdout
    assert "OWNER RIGHTS" in acl
    assert "Users" not in acl and "Everyone" not in acl


def test_blanked_setting_stays_removed_over_shell_and_dotenv(config, monkeypatch):
    monkeypatch.setattr(settings, "is_frozen", lambda: True)
    monkeypatch.setenv("MONGO_DB", "from_shell")
    settings.save({"MONGO_DB": ""})
    assert "MONGO_DB" not in os.environ
    assert json.loads(config.read_text()) == {"MONGO_DB": None}

    monkeypatch.setenv("MONGO_DB", "from_shell")  # the next launch inherits it again
    settings.load()
    assert "MONGO_DB" not in os.environ

    # Saving again with the field still blank keeps the removal.
    settings.save({"MONGO_DB": ""})
    assert json.loads(config.read_text()) == {"MONGO_DB": None}


def test_never_set_blank_is_not_recorded(config):
    settings.save({"MONGO_URI": "mongodb://h"})
    assert json.loads(config.read_text()) == {"MONGO_URI": "mongodb://h"}


@pytest.mark.parametrize("contents", ["", "{not json", "[1, 2]", b"\xff\xfe"])
def test_corrupt_settings_file_is_set_aside(config, monkeypatch, contents):
    monkeypatch.setattr(settings, "is_frozen", lambda: True)
    config.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(contents, bytes):
        config.write_bytes(contents)
    else:
        config.write_text(contents)

    settings.load()

    assert settings.load_error and str(config) in settings.load_error
    assert not config.exists()
    [backup] = config.parent.glob("settings.json.corrupt-*")
    assert backup.read_bytes() == (contents if isinstance(contents, bytes) else contents.encode())

    settings.save({"MONGO_URI": "mongodb://h"})
    settings.load()
    assert settings.load_error is None
    assert os.environ["MONGO_URI"] == "mongodb://h"
