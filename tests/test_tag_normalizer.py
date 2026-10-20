import csv
import io
import json
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QMessageBox

import tag_normalizer
from tag_normalizer import clean_tag, collect, group_tags, mapping, parse_tags
from tag_tab import TagTab

# Hand-made 3-d vectors: a car axis, a login axis and a billing axis.
VECTORS = {
    "car maintenance": [1.0, 0.0, 0.0],
    "oil change": [0.95, 0.15, 0.0],
    "tire rotation": [0.93, 0.0, 0.2],
    "password reset": [0.0, 1.0, 0.0],
    "forgot password": [0.1, 0.97, 0.0],
    "refund": [0.0, 0.0, 1.0],
}


def fake_embed(texts):
    return np.asarray([VECTORS[t] for t in texts], dtype=np.float32)


# --- parsing ----------------------------------------------------------------------


@pytest.mark.parametrize("text, expected", [
    ("a, b c,d_e", ["a", "b c", "d_e"]),
    ("a\nb, c\r\n\n d ", ["a", "b", "c", "d"]),
    ('["oil change", "tire_rotation"]', ["oil change", "tire_rotation"]),
    ("['oil change', 'refund']", ["oil change", "refund"]),
    ('"quoted tag", \'single\'', ["quoted tag", "single"]),
    ("[not, valid json]", ["not", "valid json"]),
    ("﻿a,,  ,b", ["a", "b"]),
    ("", []),
    ('name,count\n"agent_ooo",812\n"language__en",301\n', ["agent_ooo", "language__en"]),
    ("x, 3\ny, 4", ["x", "y"]),  # numbers in the first row too: no header
    ("a, b\nc, d", ["a", "b", "c", "d"]),  # no count column: not a table
])
def test_parse_tags(text, expected):
    assert parse_tags(text) == expected


@pytest.mark.parametrize("raw, cleaned", [
    ("Car_Maintenance", "car maintenance"),
    ("car-maintenance", "car maintenance"),
    ("carMaintenance", "car maintenance"),
    ("SSOLogin", "sso login"),
    ("  Tire   Rotation!! ", "tire rotation"),
    ("api/rate-limit", "api rate limit"),
    ("___", ""),
])
def test_clean_tag(raw, cleaned):
    assert clean_tag(raw) == cleaned


def test_collect_merges_spellings_and_counts():
    tags = collect(["car_maintenance", "Car Maintenance", "car_maintenance", "refund", "!!"])
    assert [t.text for t in tags] == ["car maintenance", "refund"]
    assert tags[0].count == 3
    assert tags[0].spelling == "car_maintenance"


# --- grouping -----------------------------------------------------------------------


def grouped(raw, threshold=0.9):
    tags = collect(raw)
    return group_tags(tags, fake_embed([t.text for t in tags]), threshold)


def test_groups_synonyms_under_most_central_tag():
    groups = grouped(["oil change", "tire rotation", "car_maintenance", "password reset", "forgot password", "refund"])
    assert [[t.text for t in g.members] for g in groups] == [
        ["car maintenance", "oil change", "tire rotation"],
        ["password reset", "forgot password"],
        ["refund"],
    ]
    assert groups[0].canonical == "car_maintenance"  # original spelling, not the cleaned form
    assert groups[0].similarity[0] == pytest.approx(1.0)


def test_high_threshold_keeps_tags_apart():
    groups = grouped(["oil change", "car maintenance", "refund"], threshold=0.99)
    assert all(len(g.members) == 1 for g in groups)


def test_single_tag_and_zero_vector():
    tags = collect(["refund"])
    assert [g.canonical for g in group_tags(tags, fake_embed(["refund"]), 0.9)] == ["refund"]
    tags = collect(["xyzzy", "refund"])
    groups = group_tags(tags, np.array([[0.0, 0.0, 0.0], VECTORS["refund"]]), 0.9)
    assert sorted(len(g.members) for g in groups) == [1, 1]


def test_structured_tags_kept_apart_unless_asked():
    # Opposite values of one facet embed as near-identical.
    tags = collect(["sentiment__positive", "sentiment__negative", "positive_feedback", "refund"])
    vectors = np.array([[1.0, 0.0], [0.99, 0.1], [1.0, 0.01], [0.0, 1.0]])
    groups = group_tags(tags, vectors, 0.9)
    assert [[t.spelling for t in g.members] for g in groups if len(g.members) > 1] == [
        ["sentiment__positive", "positive_feedback"]]  # the taxonomy tag names the group
    merged = group_tags(tags, vectors, 0.9, merge_structured=True)
    assert max(len(g.members) for g in merged) == 3


def test_structured_tags_apart_in_large_groups():
    # Many near-identical free tags can't average a structured pair together.
    tags = collect([f"free{i}" for i in range(30)] + ["a__x", "a__y"])
    groups = group_tags(tags, np.ones((len(tags), 2)), 0.5)
    assert all(sum(t.structured for t in g.members) <= 1 for g in groups)


def test_tab_toggle_lets_structured_tags_merge(qapp):
    tab = TagTab(lambda backend: lambda texts: np.array([[1.0, 0.0], [0.99, 0.1]])[:len(texts)])
    tab.input.setPlainText("sentiment__positive, sentiment__negative")
    tab.start_find()
    wait_until_idle(qapp, tab)
    assert tab.tree.topLevelItemCount() == 0
    tab.keep_structured_apart.setChecked(False)
    assert tab.tree.topLevelItemCount() == 1


def test_mismatched_vectors_rejected():
    with pytest.raises(ValueError):
        group_tags(collect(["a", "b"]), np.zeros((1, 3)), 0.9)


def test_mapping_covers_every_spelling_and_honours_exclusions():
    groups = grouped(["car_maintenance", "Car Maintenance", "oil change", "tire rotation", "refund"])
    assert mapping(groups) == {
        "car_maintenance": "car_maintenance", "Car Maintenance": "car_maintenance",
        "oil change": "car_maintenance", "tire rotation": "car_maintenance", "refund": "refund",
    }
    assert mapping(groups, excluded={"tire rotation"})["tire rotation"] == "tire rotation"


def test_csv_and_json_output():
    groups = grouped(["car maintenance", "oil change", "refund"])
    mapped = mapping(groups)
    rows = list(csv.DictReader(io.StringIO(tag_normalizer.to_csv(groups, mapped))))
    assert {r["tag"]: r["canonical_tag"] for r in rows} == mapped
    data = json.loads(tag_normalizer.to_json(groups, mapped))
    assert data["mapping"] == mapped
    # A pair ties on centrality and uses, so the shorter tag names it.
    assert data["groups"] == [{"canonical": "oil change", "tags": ["oil change", "car maintenance"]}]


def test_cli_writes_mapping(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(tag_normalizer, "embedder", lambda backend: fake_embed)
    source = tmp_path / "tags.txt"
    source.write_text("oil_change, Car Maintenance, car_maintenance\nrefund\n")
    out = tmp_path / "mapping.json"
    assert tag_normalizer.main([str(source), "--backend", "spacy", "--threshold", "0.9", "--out", str(out)]) == 0
    assert json.loads(out.read_text())["mapping"]["oil_change"] == "Car Maintenance"  # more uses wins the tie
    assert "1 group(s) to merge" in capsys.readouterr().out


def test_cli_no_tags(tmp_path):
    source = tmp_path / "empty.txt"
    source.write_text(" , \n")
    assert tag_normalizer.main([str(source), "--backend", "spacy"]) == 1


# --- the tab ------------------------------------------------------------------------


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


def make_tab(calls):
    def factory(backend):
        def embed(texts):
            calls.append((backend, list(texts)))
            return fake_embed(texts)
        return embed
    return TagTab(factory)


def test_tab_groups_regroups_and_caches(qapp):
    calls = []
    tab = make_tab(calls)
    tab.input.setPlainText("oil change, car_maintenance, tire rotation, refund")
    assert tab.start_find()
    wait_until_idle(qapp, tab)
    assert tab.tree.topLevelItemCount() == 1  # refund has no match and is hidden
    top = tab.tree.topLevelItem(0)
    assert top.text(0) == "car_maintenance" and top.childCount() == 3

    tab.threshold.setValue(0.99)  # regroups from stored vectors
    assert tab.tree.topLevelItemCount() == 0
    tab.show_unmatched.setChecked(True)
    assert tab.tree.topLevelItemCount() == 4

    tab.input.setPlainText("oil change, car maintenance, refund, password reset")
    tab.start_find()
    wait_until_idle(qapp, tab)
    assert calls == [("voyage", ["oil change", "car maintenance", "tire rotation", "refund"]),
                     ("voyage", ["password reset"])]


def test_tab_rename_and_untick_shape_mapping(qapp):
    tab = make_tab([])
    tab.input.setPlainText("oil change, car maintenance, tire rotation")
    tab.start_find()
    wait_until_idle(qapp, tab)
    top = tab.tree.topLevelItem(0)
    top.setText(0, "vehicle care")
    tire = next(top.child(i) for i in range(top.childCount()) if top.child(i).text(0) == "tire rotation")
    tire.setCheckState(0, QtCore.Qt.Unchecked)
    assert tab.mapping() == {"car maintenance": "vehicle care", "oil change": "vehicle care",
                             "tire rotation": "tire rotation"}
    tab.threshold.setValue(0.89)  # same group, so the rename survives regrouping
    assert tab.tree.topLevelItem(0).text(0) == "vehicle care"


def test_tab_reports_embedding_error(qapp, monkeypatch):
    shown = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *args: shown.append(args[2]))

    def factory(backend):
        raise RuntimeError("EMBEDDING_API_KEY is not set")

    tab = TagTab(factory)
    tab.input.setPlainText("refund, oil change")
    tab.start_find()
    wait_until_idle(qapp, tab)
    assert "EMBEDDING_API_KEY" in shown[0]
    assert tab.find_button.isEnabled()


def test_tab_exports_zendesk_scripts(qapp, tmp_path, monkeypatch):
    from PyQt5.QtWidgets import QFileDialog
    shown = []
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *args: str(tmp_path))
    monkeypatch.setattr(QMessageBox, "information", lambda *args: shown.append(args[2]))
    tab = make_tab([])
    assert not tab.scripts_button.isEnabled()
    tab.input.setPlainText("oil_change, car_maintenance, tire_rotation")
    tab.start_find()
    wait_until_idle(qapp, tab)
    assert tab.scripts_button.isEnabled()
    tab.export_scripts()
    (kit,) = tmp_path.iterdir()
    assert kit.name.startswith("zendesk_retag_")
    assert sorted(row.split(",")[0] for row in (kit / "tag_changes.csv").read_text().split()[1:]) == [
        "oil_change", "tire_rotation"]
    assert "2 tag change(s)" in shown[0]
