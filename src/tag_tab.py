"""Tag Normalizer tab for the PyQt app.

Takes pasted tags or a file of them, embeds each distinct tag once on a
background thread, and shows the groups of tags that mean the same thing
(see tag_normalizer). Changing the similarity threshold regroups at once
from the stored vectors, with no new embedding call. Each group's tag can be
renamed, and any tag unticked to keep it out of its group, before the
mapping is exported as CSV or JSON. Nothing is written to MongoDB or Zendesk.
"""

import os
from datetime import datetime
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np
from loguru import logger
from PyQt5 import QtCore
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

import tag_normalizer
from tag_normalizer import SPACY, VOYAGE, Group, Tag
from utility import ThreadWorker, worker_thread

BACKEND_LABELS = {
    VOYAGE: "Voyage AI (best; sends the tags to Voyage)",
    SPACY: "spaCy (local and free; misses looser synonyms)",
}
COLUMNS = ["Tag", "Uses", "Similarity"]
# Item data role holding a member row's cleaned tag text.
TAG_ROLE = QtCore.Qt.UserRole


class TagTab(QWidget):
    """The Tag Normalizer tab.

    Embedding runs on a QThread (Voyage paces itself to the key's rate limit,
    so it can take minutes). A running QThread must not be destroyed, so the
    main window calls request_shutdown before closing and waits for `idle`.

    Signals:
        idle: Emitted when a job ends after request_shutdown.

    Args:
        embedder_factory: Returns the embedder for a backend name. Defaults
            to tag_normalizer.embedder.
    """

    idle = pyqtSignal()

    def __init__(self, embedder_factory: Callable[[str], tag_normalizer.Embedder] = tag_normalizer.embedder,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.embedder_factory = embedder_factory
        self._closing = False
        self._thread: Optional[QThread] = None
        self._worker: Optional[ThreadWorker] = None
        # Tags the running job is embedding, in order.
        self._missing: List[str] = []
        # Vectors already paid for, by (backend, cleaned tag).
        self._cache: Dict[Tuple[str, str], np.ndarray] = {}
        self._tags: List[Tag] = []
        self._vectors: Optional[np.ndarray] = None
        self._groups: List[Group] = []
        # Canonical names the user typed, keyed by the group's cleaned tags.
        self._renamed: Dict[FrozenSet[str], str] = {}
        # Cleaned tags the user unticked: left as they are.
        self._excluded: Set[str] = set()
        self._build_ui()

    # --- layout ---------------------------------------------------------------

    def _build_ui(self) -> None:
        intro = QLabel(
            "Finds tags that mean the same thing and maps each to one tag. Paste tags separated by "
            "commas or new lines (or a JSON list), or load a file (a CSV's first column is the tag). Rename a group by editing its tag; "
            "untick a tag to keep it out of its group."
        )
        intro.setWordWrap(True)

        self.input = QPlainTextEdit()
        self.input.setPlaceholderText("car_maintenance, oil change, Tire Rotation\npassword_reset\nforgot password")

        self.load_button = QPushButton("Load File…")
        self.load_button.clicked.connect(self.load_file)
        self.backend_combo = QComboBox()
        for backend, label in BACKEND_LABELS.items():
            self.backend_combo.addItem(label, backend)
        self.backend_combo.currentIndexChanged.connect(self._backend_changed)
        self.find_button = QPushButton("Find Similar Tags")
        self.find_button.clicked.connect(self.start_find)
        input_row = QHBoxLayout()
        input_row.addWidget(self.load_button)
        input_row.addWidget(self.backend_combo, stretch=1)
        input_row.addWidget(self.find_button)

        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.50, 0.99)
        self.threshold.setSingleStep(0.01)
        self.threshold.setDecimals(2)
        self.threshold.setValue(tag_normalizer.DEFAULT_THRESHOLDS[VOYAGE])
        self.threshold.setToolTip("Cosine similarity a tag needs (on average) to the rest of a group. "
                                  "Higher merges only closer synonyms.")
        self.threshold.valueChanged.connect(self.regroup)
        self.keep_structured_apart = QCheckBox("Keep facet__value tags apart")
        self.keep_structured_apart.setChecked(True)
        self.keep_structured_apart.setToolTip(
            "Never group two structured tags (sentiment__positive, sentiment__negative): they come from a "
            "fixed taxonomy and embed as near-identical. Free-form tags can still join them.")
        self.keep_structured_apart.toggled.connect(self.regroup)
        self.show_unmatched = QCheckBox("Show tags with no match")
        self.show_unmatched.toggled.connect(self._populate)
        self.summary_label = QLabel()
        options_row = QHBoxLayout()
        options_row.addWidget(QLabel("Merge at similarity ≥"))
        options_row.addWidget(self.threshold)
        options_row.addWidget(self.keep_structured_apart)
        options_row.addWidget(self.show_unmatched)
        options_row.addWidget(self.summary_label, stretch=1)

        self.busy_bar = QProgressBar()
        self.busy_bar.setRange(0, 0)
        self.busy_bar.setVisible(False)

        self.tree = QTreeWidget()
        self.tree.setColumnCount(len(COLUMNS))
        self.tree.setHeaderLabels(COLUMNS)
        self.tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.tree.itemChanged.connect(self._item_changed)

        self.export_button = QPushButton("Export Mapping…")
        self.export_button.clicked.connect(self.export_mapping)
        self.export_button.setEnabled(False)
        self.scripts_button = QPushButton("Export Zendesk Scripts…")
        self.scripts_button.setToolTip("Writes scripts that a person runs to find tickets with each old tag, "
                                       "remove it, then add the new tag. Nothing is sent to Zendesk from here.")
        self.scripts_button.clicked.connect(self.export_scripts)
        self.scripts_button.setEnabled(False)
        export_row = QHBoxLayout()
        export_row.addWidget(self.export_button)
        export_row.addWidget(self.scripts_button)

        layout = QVBoxLayout(self)
        layout.addWidget(intro)
        layout.addWidget(self.input, stretch=1)
        layout.addLayout(input_row)
        layout.addLayout(options_row)
        layout.addWidget(self.busy_bar)
        layout.addWidget(self.tree, stretch=3)
        layout.addLayout(export_row)

    def backend(self) -> str:
        return self.backend_combo.currentData()

    def _backend_changed(self) -> None:
        # Each backend's similarities sit on a different scale.
        self.threshold.setValue(tag_normalizer.DEFAULT_THRESHOLDS[self.backend()])

    # --- input ------------------------------------------------------------------

    def load_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load Tags", "", "Tag files (*.txt *.csv *.json);;All files (*)")
        if not path:
            return
        try:
            with open(path, encoding="utf-8-sig") as f:
                self.input.setPlainText(f.read())
        except (OSError, UnicodeDecodeError) as e:
            QMessageBox.critical(self, "Could not read file", f"{path}: {e}")

    # --- finding groups ---------------------------------------------------------

    def start_find(self) -> bool:
        """Embeds the tags not already embedded with this backend, then groups them.

        Returns:
            False if there were no tags, or a job is already running.
        """
        if self._closing or self._thread is not None:
            return False
        tags = tag_normalizer.collect(tag_normalizer.parse_tags(self.input.toPlainText()))
        if not tags:
            QMessageBox.information(self, "No tags", "Paste or load some tags first.")
            return False
        backend = self.backend()
        missing = [tag.text for tag in tags if (backend, tag.text) not in self._cache]

        def task():
            if not missing:
                return backend, tags, []
            return backend, tags, self.embedder_factory(backend)(missing)

        label = f"Embedding {len(missing)} tag(s) with {'Voyage AI' if backend == VOYAGE else 'spaCy'}"
        self._set_busy(True, f"{label}…")
        self._thread = worker_thread(self)
        self._worker = ThreadWorker(task, label)
        self._worker.moveToThread(self._thread)
        self._missing = missing
        self._worker.finished.connect(self._job_finished)
        self._worker.error.connect(self._job_failed)
        self._thread.started.connect(self._worker.run)
        self._thread.start()
        return True

    def _stop_thread(self) -> None:
        self._thread.quit()
        self._thread.wait()
        self._thread = None
        self._worker = None
        if not self._closing:
            self._set_busy(False)

    def _job_finished(self, result: Any) -> None:
        self._stop_thread()
        missing, self._missing = self._missing, []
        if self._closing:
            self.idle.emit()
            return
        backend, tags, vectors = result
        for text, vector in zip(missing, vectors):
            self._cache[(backend, text)] = np.asarray(vector)
        self.show_tags(tags, np.stack([self._cache[(backend, tag.text)] for tag in tags]))

    def _job_failed(self, message: str) -> None:
        self._stop_thread()
        if self._closing:
            self.idle.emit()
            return
        self.summary_label.setText("Embedding failed.")
        QMessageBox.critical(self, "Tag Normalizer", message)

    def show_tags(self, tags: List[Tag], vectors: np.ndarray) -> None:
        """Groups tags (one vector each) at the current threshold and shows them."""
        self._tags, self._vectors = tags, vectors
        self.regroup()

    def regroup(self) -> None:
        if self._vectors is None:
            return
        self._groups = tag_normalizer.group_tags(self._tags, self._vectors, self.threshold.value(),
                                                 merge_structured=not self.keep_structured_apart.isChecked())
        for group in self._groups:
            group.canonical = self._renamed.get(self._key(group), group.canonical)
        self._populate()

    @staticmethod
    def _key(group: Group) -> FrozenSet[str]:
        return frozenset(tag.text for tag in group.members)

    def _populate(self) -> None:
        self.tree.blockSignals(True)
        self.tree.clear()
        merged = 0
        for group in self._groups:
            is_merge = len(group.members) > 1 or len(group.members[0].spellings) > 1
            merged += is_merge
            if not is_merge and not self.show_unmatched.isChecked():
                continue
            top = QTreeWidgetItem([group.canonical, str(group.count), ""])
            top.setFlags(top.flags() | QtCore.Qt.ItemIsEditable)
            top.setToolTip(0, "Double-click to rename the tag this group maps to.")
            for tag, similarity in zip(group.members, group.similarity):
                child = QTreeWidgetItem([", ".join(tag.spellings), str(tag.count), f"{similarity:.2f}"])
                child.setData(0, TAG_ROLE, tag.text)
                child.setFlags(child.flags() | QtCore.Qt.ItemIsUserCheckable)
                child.setCheckState(0, QtCore.Qt.Unchecked if tag.text in self._excluded else QtCore.Qt.Checked)
                top.addChild(child)
            self.tree.addTopLevelItem(top)
            top.setExpanded(is_merge)
        self.tree.blockSignals(False)
        self.summary_label.setText(f"{len(self._tags)} distinct tag(s); {merged} group(s) to merge; "
                                   f"{len(self._groups) - merged} with no match.")
        for button in (self.export_button, self.scripts_button):
            button.setEnabled(bool(self._groups) and self._thread is None)

    def _group_for(self, item: QTreeWidgetItem) -> Optional[Group]:
        top = item if item.parent() is None else item.parent()
        name_key = frozenset(top.child(i).data(0, TAG_ROLE) for i in range(top.childCount()))
        return next((g for g in self._groups if self._key(g) == name_key), None)

    def _item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if column != 0:
            return
        group = self._group_for(item)
        if group is None:
            return
        if item.parent() is None:
            name = item.text(0).strip()
            if not name:
                self.tree.blockSignals(True)
                item.setText(0, group.canonical)
                self.tree.blockSignals(False)
                return
            group.canonical = name
            self._renamed[self._key(group)] = name
        else:
            text = item.data(0, TAG_ROLE)
            if item.checkState(0) == QtCore.Qt.Checked:
                self._excluded.discard(text)
            else:
                self._excluded.add(text)

    def mapping(self) -> Dict[str, str]:
        """{original tag: tag to use instead}, with the user's renames and unticked tags."""
        return tag_normalizer.mapping(self._groups, self._excluded)

    def export_mapping(self) -> None:
        path, chosen = QFileDialog.getSaveFileName(self, "Export Tag Mapping", "tag_mapping.csv",
                                                   "CSV (*.csv);;JSON (*.json)")
        if not path:
            return
        as_json = path.lower().endswith(".json") or (chosen.startswith("JSON") and not path.lower().endswith(".csv"))
        mapped = self.mapping()
        output = (tag_normalizer.to_json if as_json else tag_normalizer.to_csv)(self._groups, mapped)
        try:
            with open(path, "w", encoding="utf-8", newline="") as f:
                f.write(output)
        except OSError as e:
            QMessageBox.critical(self, "Could not save", f"{path}: {e}")
            return
        logger.info(f"Tag Normalizer: wrote {len(mapped)} tag(s) to {path}")

    def export_scripts(self) -> None:
        """Writes the Zendesk retag scripts into a new folder inside the one chosen."""
        parent = QFileDialog.getExistingDirectory(self, "Folder for Zendesk Retag Scripts")
        if not parent:
            return
        folder = os.path.join(parent, f"zendesk_retag_{datetime.now():%Y%m%d_%H%M%S}")
        try:
            changes, warnings = tag_normalizer.write_retag_scripts(self.mapping(), folder)
        except OSError as e:
            QMessageBox.critical(self, "Could not write scripts", str(e))
            return
        logger.info(f"Tag Normalizer: wrote Zendesk retag scripts for {len(changes)} tag change(s) to {folder}")
        text = (f"Wrote scripts for {len(changes)} tag change(s) to\n{folder}\n\n"
                "Review tag_changes.csv, then follow README.txt there. Nothing was sent to Zendesk.")
        if warnings:
            text += "\n\n" + "\n".join(warnings)
        QMessageBox.information(self, "Zendesk scripts written", text)

    # --- busy state and shutdown ------------------------------------------------

    def _set_busy(self, busy: bool, label: str = "") -> None:
        for widget in (self.load_button, self.find_button, self.backend_combo, self.export_button, self.scripts_button):
            widget.setEnabled(not busy)
        if not busy:
            self.export_button.setEnabled(bool(self._groups))
            self.scripts_button.setEnabled(bool(self._groups))
        self.busy_bar.setVisible(busy)
        self.busy_bar.setFormat(label)
        if busy:
            self.summary_label.setText(label)

    def is_busy(self) -> bool:
        return self._thread is not None

    def request_shutdown(self) -> bool:
        """Stops new jobs. Returns True if the tab can close now, else `idle` follows."""
        self._closing = True
        return not self.is_busy()

    def wait_for_job(self) -> None:
        """Blocks until a running job's thread finishes (last resort on quit)."""
        if self._thread is None:
            return
        self._thread.quit()
        self._thread.wait()
        self._thread = None
        self._worker = None
