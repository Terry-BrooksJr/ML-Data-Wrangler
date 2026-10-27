"""Tag Normalizer tab for the PyQt app.

Takes pasted tags or a file of them, embeds each distinct tag once on a
background thread, and shows the groups of tags that mean the same thing
(see tag_normalizer). Changing the similarity threshold regroups at once
from the stored vectors, with no new embedding call. The language-model
option instead asks a model that follows the tag normalization reference
(see tag_llm) and shows its flags for review. Each group's tag can be
renamed, and any tag unticked to keep it out of its group, before the
mapping is exported as CSV or JSON.

Save Run keeps the result the way LDA and BERTopic runs are kept (see
tag_runs): a folder here, an S3 upload when model storage is set up, a
record in tag_normalization_run_log, and its tag changes in normalized_tags.
Find Similar Tags reads normalized_tags back, so earlier successors and
deprecations carry over. Previous Runs… reopens a saved run as it was.
Nothing is ever written to Zendesk.
"""

import os
import pathlib
from datetime import datetime
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple

from embedding import embeddings
import numpy as np
from tags import tag_llm
from tags import tag_normalizer
from tags import tag_runs
from loguru import logger
from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSignal, QThread
from PyQt5.QtWidgets import (
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)
from tags.tag_normalizer import Group, LLM, Previous, SPACY, Tag, VOYAGE
from ui.utility import ThreadWorker, worker_thread
from ui.widgets import (
    Button,
    Card,
    CheckBox,
    ComboBox,
    hint,
    ProgressBar,
    use_si_scrollbars,
)

BACKEND_LABELS = {
    VOYAGE: "Voyage AI (best; sends the tags to Voyage)",
    SPACY: "spaCy (local and free; misses looser synonyms)",
    LLM: "Language model (follows the tag reference; sends the tags to Cheaper Inference)",
}
COLUMNS = ["Tag", "Uses", "Similarity"]
# The language model scores its own decisions instead of a similarity.
MODEL_COLUMNS = ["Tag", "Uses", "Confidence"]
SAVE_TOOLTIP = (
    "Saves this run (here, to S3 when set up, and to tag_normalization_run_log) "
    "and records its tag changes in normalized_tags for later runs."
)
# Item data role holding a member row's cleaned tag text.
TAG_ROLE = QtCore.Qt.UserRole


def _default_database():
    """The app's Mongo database, or None when MONGO_URI isn't set."""
    if not os.environ.get("MONGO_URI"):
        return None
    from decision_engine.persistence import get_database

    return get_database()


def _default_storage():
    from core import model_storage

    return model_storage.config()


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
        database_factory: Returns the Mongo database for normalized_tags and
            the run log, or None to work without them. Called on the worker thread.
        storage: Returns the model_storage.Config for uploads, or None.
        run_dir: Where runs are saved (default tag_runs.run_dir()).
        model_normalizer: Normalizes tags with the language model, given
            candidate pairs: called as model_normalizer(tags, candidates).
            Defaults to tag_llm.normalize.
    """

    idle = pyqtSignal()

    def __init__(
        self,
        embedder_factory: Callable[
            [str], tag_normalizer.Embedder
        ] = tag_normalizer.embedder,
        parent: Optional[QWidget] = None,
        database_factory: Callable[[], Any] = _default_database,
        storage: Callable[[], Any] = _default_storage,
        run_dir: Optional[pathlib.Path] = None,
        model_normalizer: Callable[..., tag_llm.Result] = tag_llm.normalize,
    ) -> None:
        super().__init__(parent)
        self.embedder_factory = embedder_factory
        self.database_factory = database_factory
        self.storage = storage
        self.run_dir = run_dir
        self.model_normalizer = model_normalizer
        # The language model's last result (decisions, flags); None for embeddings.
        self._model_result: Optional[tag_llm.Result] = None
        self._closing = False
        self._thread: Optional[QThread] = None
        self._worker: Optional[ThreadWorker] = None
        self._on_done: Optional[Callable[[Any], None]] = None
        # Earlier runs' decisions, read from normalized_tags on Find.
        self._previous: Optional[Previous] = None
        # Where the tags came from: a file name, or "pasted".
        self._source = "pasted"
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
        intro = hint(
            "Finds tags that mean the same thing and maps each to one tag. Paste tags separated by "
            "commas or new lines (or a JSON list), or load a file (a CSV's first column is the tag). Rename a group by editing its tag; "
            "untick a tag to keep it out of its group."
        )

        self.input = use_si_scrollbars(QPlainTextEdit())
        self.input.setPlaceholderText(
            "car_maintenance, oil change, Tire Rotation\npassword_reset\nforgot password"
        )

        self.load_button = Button("Load File…")
        self.load_button.clicked.connect(self.load_file)
        self.backend_combo = ComboBox("Embeddings")
        for backend, label in BACKEND_LABELS.items():
            self.backend_combo.addItem(label, backend)
        self.backend_combo.currentIndexChanged.connect(self._backend_changed)
        self.find_button = Button("Find Similar Tags", primary=True)
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
        self.threshold.setToolTip(
            "Cosine similarity a tag needs (on average) to the rest of a group. "
            "Higher merges only closer synonyms."
        )
        self.threshold.valueChanged.connect(self.regroup)
        self.keep_structured_apart = CheckBox("Keep facet__value tags apart")
        self.keep_structured_apart.setChecked(True)
        self.keep_structured_apart.setToolTip(
            "Never group two structured tags (sentiment__positive, sentiment__negative): they come from a "
            "fixed taxonomy and embed as near-identical. Free-form tags can still join them."
        )
        self.keep_structured_apart.toggled.connect(self.regroup)
        self.show_unmatched = CheckBox("Show tags with no match")
        self.show_unmatched.toggled.connect(self._populate)
        self.summary_label = hint("")
        options_row = QHBoxLayout()
        options_row.addWidget(QLabel("Merge at similarity ≥"))
        options_row.addWidget(self.threshold)
        options_row.addWidget(self.keep_structured_apart)
        options_row.addWidget(self.show_unmatched)
        options_row.addWidget(self.summary_label, stretch=1)

        self.busy_bar = ProgressBar()
        self.busy_bar.setRange(0, 0)
        self.busy_bar.setVisible(False)

        self.tree = use_si_scrollbars(QTreeWidget())
        self.tree.setColumnCount(len(COLUMNS))
        self.tree.setHeaderLabels(COLUMNS)
        self.tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.tree.itemChanged.connect(self._item_changed)

        self.export_button = Button("Export Mapping…")
        self.export_button.clicked.connect(self.export_mapping)
        self.export_button.setEnabled(False)
        self.scripts_button = Button("Export Zendesk Scripts…")
        self.scripts_button.setToolTip(
            "Writes scripts that a person runs to find tickets with each old tag, "
            "remove it, then add the new tag. Nothing is sent to Zendesk from here."
        )
        self.scripts_button.clicked.connect(self.export_scripts)
        self.scripts_button.setEnabled(False)
        self.runs_button = Button("Previous Runs…")
        self.runs_button.setToolTip(
            "Reopen a saved run, from MongoDB or this computer."
        )
        self.runs_button.clicked.connect(self.show_previous_runs)
        self.save_button = Button("Save Run", primary=True)
        self.save_button.setToolTip(SAVE_TOOLTIP)
        self.save_button.clicked.connect(self.save_run)
        self.save_button.setEnabled(False)
        # The model's flags: what it wants a person to decide.
        self.flags_view = use_si_scrollbars(QPlainTextEdit())
        self.flags_view.setReadOnly(True)
        self.flags_view.setMaximumHeight(140)
        self.flags_view.setVisible(False)
        export_row = QHBoxLayout()
        export_row.addWidget(self.runs_button)
        export_row.addStretch()
        export_row.addWidget(self.save_button)
        export_row.addWidget(self.export_button)
        export_row.addWidget(self.scripts_button)

        tags_card = Card("Tags")
        tags_card.body.addWidget(intro)
        self.input.setMaximumHeight(120)
        tags_card.body.addWidget(self.input)
        tags_card.body.addLayout(input_row)
        tags_card.body.addLayout(options_row)
        tags_card.body.addWidget(self.busy_bar)

        groups_card = Card("Groups")
        groups_card.body.addWidget(self.tree, stretch=1)
        groups_card.body.addWidget(self.flags_view)
        groups_card.body.addLayout(export_row)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 12, 0, 0)
        layout.setSpacing(12)
        layout.addWidget(tags_card)
        layout.addWidget(groups_card, stretch=1)

    def backend(self) -> str:
        return self.backend_combo.currentData()

    def _backend_changed(self) -> None:
        # The model maps tags itself: no threshold, and the reference's rules
        # (not this switch) keep structured tags apart.
        uses_vectors = self.backend() != LLM
        self.threshold.setEnabled(uses_vectors)
        self.keep_structured_apart.setEnabled(uses_vectors)
        if uses_vectors:
            # Each backend's similarities sit on a different scale.
            self.threshold.setValue(tag_normalizer.DEFAULT_THRESHOLDS[self.backend()])

    # --- input ------------------------------------------------------------------

    def load_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Tags", "", "Tag files (*.txt *.csv *.json);;All files (*)"
        )
        if not path:
            return
        try:
            with open(path, encoding="utf-8-sig") as f:
                self.input.setPlainText(f.read())
        except (OSError, UnicodeDecodeError) as e:
            QMessageBox.critical(self, "Could not read file", f"{path}: {e}")
            return
        self._source = os.path.basename(path)

    # --- finding groups ---------------------------------------------------------

    def start_find(self) -> bool:
        """Reads earlier decisions, embeds the tags not already embedded with this
        backend (saved successors included), then groups them.

        Returns:
            False if there were no tags, or a job is already running.
        """
        if self._closing or self._thread is not None:
            return False
        tags = tag_normalizer.collect(
            tag_normalizer.parse_tags(self.input.toPlainText())
        )
        if not tags:
            QMessageBox.information(self, "No tags", "Paste or load some tags first.")
            return False
        backend = self.backend()
        if backend == LLM:
            return self._start_model(tags)
        cached = {text for b, text in self._cache if b == backend}

        def task():
            previous = self._read_previous()
            everything = tag_normalizer.with_successors(tags, previous)
            missing = [tag.text for tag in everything if tag.text not in cached]
            vectors = self.embedder_factory(backend)(missing) if missing else []
            return backend, everything, previous, missing, vectors

        def done(result):
            backend, everything, previous, missing, vectors = result
            for text, vector in zip(missing, vectors):
                self._cache[(backend, text)] = np.asarray(vector)
            self._previous = previous
            self.show_tags(
                everything,
                np.stack([self._cache[(backend, tag.text)] for tag in everything]),
            )

        engine = "Voyage AI" if backend == VOYAGE else "spaCy"
        self._run(task, f"Embedding tags with {engine}", done)
        return True

    def _start_model(self, tags: List[Tag]) -> bool:
        """Asks the language model, with candidate pairs from Voyage (see tag_llm).

        Zendesk AI tags are never embedded or sent (tag_llm.split_ignored).
        Earlier decisions (normalized_tags) are not applied to model runs.
        """
        to_embed, _ = tag_llm.split_ignored(tags)
        missing = [t.text for t in to_embed if (VOYAGE, t.text) not in self._cache]
        known = {
            t.text: self._cache[(VOYAGE, t.text)]
            for t in to_embed
            if (VOYAGE, t.text) in self._cache
        }

        def task():
            new, candidates, note = [], [], None
            try:
                if missing:
                    new = self.embedder_factory(VOYAGE)(missing)
                vectors = {**known, **dict(zip(missing, new))}
                if len(to_embed) > 1:
                    candidates = tag_llm.candidate_pairs(
                        to_embed, np.stack([vectors[t.text] for t in to_embed])
                    )
            except (embeddings.EmbeddingNotConfigured, embeddings.ApiError) as e:
                new, note = [], f"No candidate pairs from Voyage: {e}"
            result = self.model_normalizer(tags, candidates=candidates)
            if note:
                result.warnings.append(note)
            return result, new

        def done(outcome):
            result, new = outcome
            for text, vector in zip(missing, new):
                self._cache[(VOYAGE, text)] = np.asarray(vector)
            self._previous = None
            self.show_model_result(tags, result)

        self._run(
            task, f"Asking the language model to normalize {len(tags)} tag(s)", done
        )
        return True

    def _read_previous(self) -> Optional[Previous]:
        """normalized_tags' decisions, or None (logged) when Mongo is off or unreachable."""
        try:
            database = self.database_factory()
            if database is None:
                return None
            return tag_runs.previous_decisions(
                database[tag_runs.normalized_collection()]
            )
        except Exception as e:
            logger.warning(
                f"Tag Normalizer: could not read {tag_runs.normalized_collection()} ({e}); "
                "grouping without earlier decisions."
            )
            return None

    def _run(
        self, task: Callable[[], Any], label: str, on_done: Callable[[Any], None]
    ) -> None:
        """Runs task on the tab's worker thread, then on_done(result) on the GUI thread."""
        self._set_busy(True, f"{label}…")
        self._on_done = on_done
        self._thread = worker_thread(self)
        self._worker = ThreadWorker(task, label)
        self._worker.moveToThread(self._thread)
        self._worker.finished.connect(self._job_finished)
        self._worker.error.connect(self._job_failed)
        self._thread.started.connect(self._worker.run)
        self._thread.start()

    def _stop_thread(self) -> None:
        self._thread.quit()
        self._thread.wait()
        self._thread = None
        self._worker = None
        if not self._closing:
            self._set_busy(False)

    def _job_finished(self, result: Any) -> None:
        self._stop_thread()
        on_done, self._on_done = self._on_done, None
        if self._closing:
            self.idle.emit()
            return
        on_done(result)

    def _job_failed(self, message: str) -> None:
        self._stop_thread()
        self._on_done = None
        if self._closing:
            self.idle.emit()
            return
        self.summary_label.setText("The job failed.")
        QMessageBox.critical(self, "Tag Normalizer", message)

    def show_tags(self, tags: List[Tag], vectors: np.ndarray) -> None:
        """Groups tags (one vector each) at the current threshold and shows them."""
        self._tags, self._vectors = tags, vectors
        self._show_model_notes(None)
        self.regroup()

    def show_model_result(self, tags: List[Tag], result: tag_llm.Result) -> None:
        """Shows the language model's checked groups, with its flags for review."""
        self._tags, self._vectors = tags, None
        self._groups = result.groups
        for group in self._groups:
            group.canonical = self._renamed.get(self._key(group), group.canonical)
        self._show_model_notes(result)
        self._populate()

    def _show_model_notes(self, result: Optional[tag_llm.Result]) -> None:
        self._model_result = result
        self.tree.setHeaderLabels(MODEL_COLUMNS if result else COLUMNS)
        notes = tag_llm.describe_flags(result) if result else ""
        self.flags_view.setPlainText(
            f"For review ({result.model}):\n{notes}" if notes else ""
        )
        self.flags_view.setVisible(bool(notes))
        self.save_button.setToolTip(
            "Language-model runs can't be saved yet: Export Mapping keeps the result."
            if result
            else SAVE_TOOLTIP
        )

    def regroup(self) -> None:
        if self._vectors is None:
            return
        self._groups = tag_normalizer.group_tags(
            self._tags,
            self._vectors,
            self.threshold.value(),
            merge_structured=not self.keep_structured_apart.isChecked(),
            previous=self._previous,
        )
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
        reasons = {
            d.get("tag"): d.get("reason")
            for d in (self._model_result.decisions if self._model_result else [])
            if d.get("reason")
        }
        for group in self._groups:
            is_merge = len(group.members) > 1 or len(group.members[0].spellings) > 1
            merged += is_merge
            if not is_merge and not self.show_unmatched.isChecked():
                continue
            top = QTreeWidgetItem([group.canonical, str(group.count), ""])
            top.setFlags(top.flags() | QtCore.Qt.ItemIsEditable)
            top.setToolTip(0, "Double-click to rename the tag this group maps to.")
            if group.members[0].known:
                top.setToolTip(
                    0,
                    "A successor saved by an earlier run (normalized_tags). "
                    "Double-click to rename it.",
                )
            for tag, similarity in zip(group.members, group.similarity):
                if tag.anchor_only:
                    # Not in this input: shown for context, never mapped.
                    child = QTreeWidgetItem(
                        [f"{tag.successor} (saved successor)", "–", ""]
                    )
                    child.setData(0, TAG_ROLE, tag.text)
                    child.setFlags(child.flags() & ~QtCore.Qt.ItemIsUserCheckable)
                    top.addChild(child)
                    continue
                spellings = ", ".join(s for s, uses in tag.spellings.items() if uses)
                child = QTreeWidgetItem(
                    [spellings, str(tag.count), f"{similarity:.2f}"]
                )
                child.setData(0, TAG_ROLE, tag.text)
                reason = next((reasons[s] for s in tag.spellings if s in reasons), None)
                if reason:
                    child.setToolTip(0, str(reason))
                child.setFlags(child.flags() | QtCore.Qt.ItemIsUserCheckable)
                child.setCheckState(
                    0,
                    (
                        QtCore.Qt.Unchecked
                        if tag.text in self._excluded
                        else QtCore.Qt.Checked
                    ),
                )
                top.addChild(child)
            self.tree.addTopLevelItem(top)
            top.setExpanded(is_merge)
        self.tree.blockSignals(False)
        distinct = sum(1 for tag in self._tags if not tag.anchor_only)
        carried = len(self._previous.successors) if self._previous else 0
        self.summary_label.setText(
            f"{distinct} distinct tag(s); {merged} group(s) to merge; "
            f"{len(self._groups) - merged} with no match."
            + (f" {carried} saved successor(s) considered." if carried else "")
        )
        for button in (self.export_button, self.scripts_button):
            button.setEnabled(bool(self._groups) and self._thread is None)
        self.save_button.setEnabled(self._can_save() and self._thread is None)

    def _group_for(self, item: QTreeWidgetItem) -> Optional[Group]:
        top = item if item.parent() is None else item.parent()
        name_key = frozenset(
            top.child(i).data(0, TAG_ROLE) for i in range(top.childCount())
        )
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

    # --- saved runs -------------------------------------------------------------

    def _can_save(self) -> bool:
        """Saved runs keep a vector per tag and regroup by embeddings when reopened,
        so a language-model result (no vectors for every tag) can't be saved yet."""
        return bool(self._groups) and self._model_result is None

    def save_run(self) -> bool:
        """Saves the run and its decisions on the worker thread (see tag_runs.save).

        Returns:
            False if there is nothing to save or a job is running.
        """
        if self._closing or self._thread is not None or not self._can_save():
            return False
        tags, vectors, groups = self._tags, self._vectors, self._groups
        mapped, previous, source = self.mapping(), self._previous, self._source
        backend = self.backend()
        settings = {
            "backend": backend,
            "embedding": tag_runs.embedding_info(backend),
            "threshold": round(self.threshold.value(), 2),
            "merge_structured": not self.keep_structured_apart.isChecked(),
        }
        edits = {
            "renamed": [
                {"tags": sorted(key), "canonical": name}
                for key, name in self._renamed.items()
            ],
            "excluded": sorted(self._excluded),
        }

        def task():
            return tag_runs.save(
                tags,
                vectors,
                groups,
                mapped,
                settings,
                edits,
                previous,
                source,
                database=self.database_factory(),
                storage=self.storage,
                base_dir=self.run_dir,
            )

        self._run(task, "Saving the run", self._saved)
        return True

    def _saved(self, run: dict) -> None:
        counts = run.get("counts") or {}
        text = (
            f"Saved run {run['run_id']} in {run['model_dir']}: {counts.get('tags_changed', 0)} "
            f"tag(s) mapped to {counts.get('groups_merged', 0)} successor(s)."
        )
        if run.get("storage"):
            text += f"\nUploaded to {run['storage']['url']}."
        if run.get("normalized") is not None:
            text += f"\n{tag_runs.normalized_collection()} updated; later runs will use these decisions."
        if run.get("warnings"):
            text += "\n\nWarning:\n" + "\n".join(run["warnings"])
        logger.success(f"Tag Normalizer: saved run {run['run_id']}")
        QMessageBox.information(self, "Run Saved", text)

    def show_previous_runs(self) -> bool:
        """Lists saved runs (Mongo and this computer) on the worker thread, then
        reopens the one picked, downloading it first if its files aren't here."""
        if self._closing or self._thread is not None:
            return False

        def task():
            try:
                database = self.database_factory()
            except Exception as e:
                logger.warning(
                    f"Tag Normalizer: MongoDB unreachable ({e}); listing this computer's runs."
                )
                database = None
            runs = (
                database[tag_runs.runs_collection()] if database is not None else None
            )
            return tag_runs.known_runs(runs, self.run_dir)

        self._run(task, "Listing saved runs", self._choose_run)
        return True

    def _choose_run(self, runs: List[dict]) -> None:
        if not runs:
            QMessageBox.information(
                self, "Previous Runs", "No saved Tag Normalizer runs were found."
            )
            return
        labels = [
            f"{tag_runs.describe_run(run)} · {tag_runs.where(run, self.run_dir)}"
            for run in runs
        ]
        label, ok = QInputDialog.getItem(
            self, "Previous Runs", "Run:", labels, 0, False
        )
        if not ok:
            return
        run = runs[labels.index(label)]

        def task():
            tag_runs.ensure_local(run, self.run_dir, self.storage)
            return run, *tag_runs.load_tags(run, self.run_dir)

        self._run(task, f"Opening run {run['run_id']}", self._restore_run)

    def _restore_run(self, result) -> None:
        """Shows a saved run with the settings and edits it was saved with."""
        run, tags, vectors = result
        settings = run.get("settings") or {}
        backend = settings.get("backend", VOYAGE)
        for widget in (self.backend_combo, self.threshold, self.keep_structured_apart):
            widget.blockSignals(True)
        index = self.backend_combo.findData(backend)
        if index >= 0:
            self.backend_combo.setCurrentIndex(index)
        self.threshold.setValue(
            float(settings.get("threshold", self.threshold.value()))
        )
        self.keep_structured_apart.setChecked(
            not settings.get("merge_structured", False)
        )
        for widget in (self.backend_combo, self.threshold, self.keep_structured_apart):
            widget.blockSignals(False)
            # Saved runs are embedding runs; the model option turns these off.
            widget.setEnabled(True)
        for tag, vector in zip(tags, vectors):
            self._cache[(backend, tag.text)] = np.asarray(vector)
        self.input.setPlainText(
            "\n".join(s for tag in tags for s, uses in tag.spellings.items() if uses)
        )
        self._source = run.get("source") or "pasted"
        self._renamed = tag_runs.renamed_of(run)
        self._excluded = set((run.get("edits") or {}).get("excluded", []))
        self._previous = tag_runs.previous_of(run)
        self.show_tags(tags, vectors)
        logger.info(f"Tag Normalizer: reopened run {run['run_id']}")

    def export_mapping(self) -> None:
        path, chosen = QFileDialog.getSaveFileName(
            self, "Export Tag Mapping", "tag_mapping.csv", "CSV (*.csv);;JSON (*.json)"
        )
        if not path:
            return
        as_json = path.lower().endswith(".json") or (
            chosen.startswith("JSON") and not path.lower().endswith(".csv")
        )
        mapped = self.mapping()
        extra = self._model_result.extra() if self._model_result else None
        output = (
            tag_normalizer.to_json(self._groups, mapped, extra)
            if as_json
            else tag_normalizer.to_csv(self._groups, mapped)
        )
        try:
            with open(path, "w", encoding="utf-8", newline="") as f:
                f.write(output)
        except OSError as e:
            QMessageBox.critical(self, "Could not save", f"{path}: {e}")
            return
        logger.info(f"Tag Normalizer: wrote {len(mapped)} tag(s) to {path}")

    def export_scripts(self) -> None:
        """Writes the Zendesk retag scripts into a new folder inside the one chosen."""
        parent = QFileDialog.getExistingDirectory(
            self, "Folder for Zendesk Retag Scripts"
        )
        if not parent:
            return
        folder = os.path.join(parent, f"zendesk_retag_{datetime.now():%Y%m%d_%H%M%S}")
        try:
            changes, warnings = tag_normalizer.write_retag_scripts(
                self.mapping(), folder
            )
        except OSError as e:
            QMessageBox.critical(self, "Could not write scripts", str(e))
            return
        logger.info(
            f"Tag Normalizer: wrote Zendesk retag scripts for {len(changes)} tag change(s) to {folder}"
        )
        text = (
            f"Wrote scripts for {len(changes)} tag change(s) to\n{folder}\n\n"
            "Review tag_changes.csv, then follow README.txt there. Nothing was sent to Zendesk."
        )
        if warnings:
            text += "\n\n" + "\n".join(warnings)
        QMessageBox.information(self, "Zendesk scripts written", text)

    # --- busy state and shutdown ------------------------------------------------

    def _set_busy(self, busy: bool, label: str = "") -> None:
        for widget in (
            self.load_button,
            self.find_button,
            self.backend_combo,
            self.export_button,
            self.scripts_button,
            self.save_button,
            self.runs_button,
        ):
            widget.setEnabled(not busy)
        if not busy:
            for button in (self.export_button, self.scripts_button):
                button.setEnabled(bool(self._groups))
            self.save_button.setEnabled(self._can_save())
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
