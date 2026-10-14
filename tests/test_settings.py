import json
import os

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
    assert oct(config.stat().st_mode & 0o777) == "0o600"

    monkeypatch.delenv("MONGO_URI")
    monkeypatch.setattr(settings, "is_frozen", lambda: True)
    settings.load()
    assert os.environ["MONGO_URI"] == uri


def test_blank_value_removes_setting(config):
    settings.save({"MONGO_DB": "other", "MONGO_URI": "mongodb://h"})
    settings.save({"MONGO_DB": "  ", "MONGO_URI": "mongodb://h"})
    assert "MONGO_DB" not in os.environ
    assert json.loads(config.read_text()) == {"MONGO_URI": "mongodb://h"}


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
