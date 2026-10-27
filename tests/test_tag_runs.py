import json
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import mongomock
import numpy as np
import pytest
from tags import tag_normalizer
from tags import tag_runs
from PyQt5.QtWidgets import QApplication, QInputDialog, QMessageBox
from tags.tag_normalizer import collect, group_tags, mapping, Previous, with_successors
from ui.tabs.tag_tab import TagTab
from test_tag_normalizer import fake_embed, VECTORS

VECTORS.update({"vehicle care": [0.97, 0.05, 0.1], "pw reset": [0.05, 0.98, 0.0]})


@pytest.fixture
def db():
    return mongomock.MongoClient().support_db


@pytest.fixture
def normalized(db):
    return db[tag_runs.normalized_collection()]


def doc(normalized, text):
    return normalized.find_one({"text": text}, {"_id": 0})


def replaced(normalized, text):
    return sorted(e["tag"] for e in doc(normalized, text)["replaces"])


# --- normalized_tags ------------------------------------------------------------------


def test_record_decisions_creates_one_document_per_successor(normalized):
    counts = tag_runs.record_decisions(
        normalized,
        {"oil_change": "car_maintenance", "tire_rotation": "car_maintenance",
         "car_maintenance": "car_maintenance", "refund": "refund"},
        "run-1",
    )
    assert counts["successors"] == 1 and counts["deprecated"] == 2
    car = doc(normalized, "car maintenance")
    assert car["tag"] == "car_maintenance" and car["run_ids"] == ["run-1"]
    assert replaced(normalized, "car maintenance") == ["oil_change", "tire_rotation"]
    assert doc(normalized, "refund") is None  # kept as itself, replaces nothing

    previous = tag_runs.previous_decisions(normalized)
    assert previous.successors == {"car maintenance": "car_maintenance"}
    assert previous.replaced == {"oil change": "car maintenance", "tire rotation": "car maintenance"}


def test_a_successor_merged_later_folds_in_without_chains(normalized):
    tag_runs.record_decisions(normalized, {"oil_change": "car_maintenance"}, "run-1")
    tag_runs.record_decisions(normalized, {"car_maintenance": "vehicle_care"}, "run-2")

    assert doc(normalized, "car maintenance") is None
    assert replaced(normalized, "vehicle care") == ["car_maintenance", "oil_change"]
    oil = next(e for e in doc(normalized, "vehicle care")["replaces"] if e["tag"] == "oil_change")
    assert oil["run_id"] == "run-1"  # keeps the run that first deprecated it


def test_latest_run_wins(normalized):
    tag_runs.record_decisions(normalized, {"oil_change": "car_maintenance", "tire_rotation": "car_maintenance"}, "run-1")
    # oil_change moves to another successor; tire_rotation is kept as itself.
    tag_runs.record_decisions(
        normalized, {"oil_change": "vehicle_care", "tire_rotation": "tire_rotation"}, "run-2"
    )
    assert replaced(normalized, "vehicle care") == ["oil_change"]
    # car_maintenance now replaces nothing, so it's gone.
    assert doc(normalized, "car maintenance") is None
    assert normalized.count_documents({"replaces.tag": "tire_rotation"}) == 0


def test_a_deprecated_tag_chosen_as_successor_is_released(normalized):
    tag_runs.record_decisions(normalized, {"oil_change": "car_maintenance", "tire_rotation": "car_maintenance"}, "run-1")
    tag_runs.record_decisions(normalized, {"lube_job": "oil_change"}, "run-2")
    assert replaced(normalized, "car maintenance") == ["tire_rotation"]
    assert replaced(normalized, "oil change") == ["lube_job"]


def test_spelling_variants_are_replaced_by_their_own_successor(normalized):
    tag_runs.record_decisions(normalized, {"Password-Reset": "password_reset"}, "run-1")
    assert replaced(normalized, "password reset") == ["Password-Reset"]
    # Same cleaned text: it's the successor's spelling, not a deprecation to pin.
    assert tag_runs.previous_decisions(normalized).replaced == {}


# --- grouping with earlier decisions --------------------------------------------------


def test_saved_successor_names_its_group_and_takes_new_synonyms():
    previous = Previous(successors={"vehicle care": "vehicle_care"})
    tags = with_successors(collect(["oil change", "car maintenance", "refund"]), previous)
    groups = group_tags(tags, fake_embed([t.text for t in tags]), 0.9, previous=previous)
    car = next(g for g in groups if len(g.members) > 1)
    assert car.canonical == "vehicle_care"
    mapped = mapping(groups)
    assert mapped["oil change"] == "vehicle_care" and "vehicle_care" not in mapped


def test_deprecated_tag_goes_to_its_successor_whatever_its_similarity():
    previous = Previous(successors={"password reset": "password_reset"}, replaced={"refund": "password reset"})
    tags = with_successors(collect(["refund", "oil change"]), previous)
    groups = group_tags(tags, fake_embed([t.text for t in tags]), 0.99, previous=previous)
    assert mapping(groups)["refund"] == "password_reset"


def test_two_saved_successors_never_merge():
    previous = Previous(successors={"car maintenance": "car_maintenance", "vehicle care": "vehicle_care"})
    tags = with_successors(collect(["oil change"]), previous)
    groups = group_tags(tags, fake_embed([t.text for t in tags]), 0.5, previous=previous)
    assert all(sum(t.known for t in g.members) <= 1 for g in groups)


# --- saving and reopening runs ------------------------------------------------------------


def run_parts(raw):
    tags = collect(raw)
    vectors = fake_embed([t.text for t in tags])
    groups = group_tags(tags, vectors, 0.9)
    return tags, vectors, groups, mapping(groups)


def test_save_writes_files_run_log_and_decisions(db, tmp_path):
    tags, vectors, groups, mapped = run_parts(["oil_change", "car_maintenance", "refund"])
    run = tag_runs.save(
        tags, vectors, groups, mapped, {"backend": "spacy", "threshold": 0.9}, {"renamed": [], "excluded": []},
        None, "tags.csv", database=db, base_dir=tmp_path,
    )
    folder = tmp_path / run["run_id"]
    for name in (tag_runs.TAGS_FILE, tag_runs.VECTORS_FILE, tag_runs.MAPPING_CSV, tag_runs.MAPPING_JSON, "run.json"):
        assert (folder / name).is_file()
    # A tied pair: the shorter tag names it.
    assert run["changes"] == {"car_maintenance": "oil_change"} and run["warnings"] == []
    logged = db[tag_runs.runs_collection()].find_one({"run_id": run["run_id"]})
    assert logged["method"] == tag_runs.METHOD and logged["source"] == "tags.csv"
    assert logged["normalized"]["deprecated"] == 1
    assert json.loads((folder / "run.json").read_text())["normalized"]["deprecated"] == 1

    reopened_tags, reopened_vectors = tag_runs.load_tags(run, tmp_path)
    assert [t.text for t in reopened_tags] == [t.text for t in tags]
    assert np.allclose(reopened_vectors, vectors)
    assert [r["run_id"] for r in tag_runs.known_runs(db[tag_runs.runs_collection()], tmp_path)] == [run["run_id"]]


def test_save_without_mongo_keeps_the_run_and_warns(tmp_path):
    tags, vectors, groups, mapped = run_parts(["oil change", "car maintenance"])
    run = tag_runs.save(tags, vectors, groups, mapped, {}, {}, None, database=None, base_dir=tmp_path)
    assert tag_runs.has_local_files(run, tmp_path)
    assert "MongoDB isn't set up" in run["warnings"][0]


def test_a_mongo_failure_becomes_a_warning(tmp_path):
    class Broken:
        def __getitem__(self, name):
            raise RuntimeError("no route to host")

    tags, vectors, groups, mapped = run_parts(["oil change", "car maintenance"])
    run = tag_runs.save(tags, vectors, groups, mapped, {}, {}, None, database=Broken(), base_dir=tmp_path)
    assert tag_runs.has_local_files(run, tmp_path) and len(run["warnings"]) == 2


def test_local_dir_refuses_a_path_run_id(tmp_path):
    with pytest.raises(ValueError):
        tag_runs.local_dir({"run_id": "../escape"}, tmp_path)


# --- the tab --------------------------------------------------------------------------------


@pytest.fixture
def qapp():
    return QApplication.instance() or QApplication([])


def wait_until_idle(qapp, tab, timeout=5.0):
    deadline = time.monotonic() + timeout
    while tab.is_busy() and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    qapp.processEvents()
    assert not tab.is_busy()


def make_tab(db, tmp_path, calls=None):
    def factory(backend):
        def embed(texts):
            (calls if calls is not None else []).append(list(texts))
            return fake_embed(texts)

        return embed

    return TagTab(factory, database_factory=lambda: db, storage=lambda: None, run_dir=tmp_path)


def test_tab_saves_then_next_find_uses_the_decisions(qapp, db, tmp_path, monkeypatch):
    shown = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a: shown.append(a[2]))
    tab = make_tab(db, tmp_path)
    tab.input.setPlainText("oil_change, car_maintenance")
    tab.start_find()
    wait_until_idle(qapp, tab)
    tab.tree.topLevelItem(0).setText(0, "vehicle_care")  # rename the group
    assert tab.save_button.isEnabled() and tab.save_run()
    wait_until_idle(qapp, tab)
    assert "Saved run" in shown[-1]
    assert replaced(db[tag_runs.normalized_collection()], "vehicle care") == ["car_maintenance", "oil_change"]

    calls = []
    later = make_tab(db, tmp_path, calls)
    later.input.setPlainText("tire rotation, refund")
    later.start_find()
    wait_until_idle(qapp, later)
    assert "vehicle care" in calls[0]  # the saved successor is embedded as an anchor
    assert later.mapping() == {"tire rotation": "vehicle_care", "refund": "refund"}


def test_tab_reopens_a_saved_run_without_embedding(qapp, db, tmp_path, monkeypatch):
    monkeypatch.setattr(QMessageBox, "information", lambda *a: None)
    tab = make_tab(db, tmp_path)
    tab.input.setPlainText("oil change, car maintenance, tire rotation")
    tab.start_find()
    wait_until_idle(qapp, tab)
    top = tab.tree.topLevelItem(0)
    tire = next(top.child(i) for i in range(top.childCount()) if top.child(i).text(0) == "tire rotation")
    from PyQt5 import QtCore

    tire.setCheckState(0, QtCore.Qt.Unchecked)
    expected = tab.mapping()
    tab.save_run()
    wait_until_idle(qapp, tab)

    calls = []
    other = make_tab(db, tmp_path, calls)
    monkeypatch.setattr(QInputDialog, "getItem", lambda *a: (a[3][0], True))
    other.show_previous_runs()
    wait_until_idle(qapp, other)  # lists the runs, then opens the one picked
    wait_until_idle(qapp, other)
    assert calls == [] and other.mapping() == expected


# --- S3 -------------------------------------------------------------------------------------

from test_model_storage import BUCKET, env, s3  # noqa: E402,F401  (fixtures)


def test_upload_then_download_on_another_computer(s3, db, tmp_path):
    from core import model_storage
    import shutil

    tags, vectors, groups, mapped = run_parts(["oil change", "car maintenance"])
    here = tmp_path / "here"
    run = tag_runs.save(
        tags, vectors, groups, mapped, {}, {}, None,
        database=db, storage=model_storage.config, base_dir=here,
    )
    assert run["warnings"] == []
    assert run["storage"]["prefix"] == f"topic-models/{tag_runs.STORAGE_SUBFOLDER}{run['run_id']}/"

    # Another computer: only the run log's record, no files.
    shutil.rmtree(here)
    elsewhere = tmp_path / "elsewhere"
    [record] = tag_runs.known_runs(db[tag_runs.runs_collection()], elsewhere)
    tag_runs.ensure_local(record, elsewhere, model_storage.config)
    reopened, reopened_vectors = tag_runs.load_tags(record, elsewhere)
    assert [t.text for t in reopened] == [t.text for t in tags]
    assert np.allclose(reopened_vectors, vectors)


def test_upload_failure_is_a_warning_with_the_retry_command(env, tmp_path, monkeypatch):
    from core import model_storage

    def fail(*args, **kwargs):
        raise RuntimeError("bucket unreachable")

    monkeypatch.setattr(model_storage, "upload_and_record", fail)
    tags, vectors, groups, mapped = run_parts(["oil change", "car maintenance"])
    run = tag_runs.save(tags, vectors, groups, mapped, {}, {}, None,
                        database=None, storage=model_storage.config, base_dir=tmp_path)
    assert "model_storage upload --tags" in run["warnings"][0]
