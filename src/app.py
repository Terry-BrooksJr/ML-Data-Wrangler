import json
import os
import pathlib
import sys
import time
from datetime import datetime
from typing import Any, Callable, List, Optional, Tuple, Union

import settings
import spend_guard
from loguru import logger
from PyQt5 import QtCore
from PyQt5.QtCore import (
    pyqtSignal,
    pyqtSlot,
    QCoreApplication,
    QMutex,
    QObject,
    QRect,
    QThread,
    QTimer,
)
from PyQt5.QtGui import QFont, QIcon, QKeySequence
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QScrollArea,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# Before the local imports, so everything sees the saved settings from the
# start. See settings.load() for which source wins.
settings.load()
settings.use_workspace()

import bertopic_model
import bertopic_runs
import chart_viewer
import en_core_web_lg
import macos
import model_storage
import pin_dialog
import tag_normalizer
import theme
import topic_store
from bertopic_dialog import BertopicOptionsDialog, BertopicResultsDialog
from decision_engine.persistence import get_database, reset_clients
from LDA_logic import LatentDirichletAllocator, stopwords
from results_dialog import ResultsDialog
from settings_dialog import SettingsDialog
from tag_tab import TagTab
from triage_tab import TriageTab
from utility import LogHighlighter, QTextEditStream, ThreadWorker, worker_thread
from widgets import button_row, Button, Card, hint, LineEdit, ProgressBar, use_si_scrollbars
from wrangler import DataWrangler, process_progress

nlp = en_core_web_lg.load()
stop_words: List[str] = stopwords.words("english")


class StatusRelay(QObject):
    """Carries status text from a worker thread to the GUI thread (queued, as the receiver lives there)."""

    message = pyqtSignal(str)


class MainWindow(QMainWindow, QDialog):
    """
    Represents the main application window for the Data Wrangler tool.
    This class initializes the user interface and manages interactions for data processing and model training.

    The `MainWindow` class sets up the main application window, including layouts, buttons, and logging.
    It provides methods for selecting files, processing data, and training models, while ensuring user inputs are validated.

    Args:
        windowName (str): The title of the main application window.

    Attributes:
        wrangler (DataWrangler): An instance of the DataWrangler class for managing ticket and comment data.
        allocator (LatentDirichletAllocator): An instance of the LatentDirichletAllocator for topic modeling.
        main_layout (QVBoxLayout): The main layout of the application.
        instruction_layout (QVBoxLayout): Layout for displaying instructions and warnings.
        data_entry_layout (QVBoxLayout): The Topic Modeling tab's cards.
        ticket_file_button (Button): Button for selecting the ticket file.
        comments_dir_button (Button): Button for selecting the comments directory.
        process_button (Button): Button for processing the data.
        train_model_button (Button): Button for training the model.
        num_topics_input (LineEdit): Input field for the number of topics.
        iterations_input (LineEdit): Input field for the number of iterations.
        passes_input (LineEdit): Input field for the number of passes.
        log_output (QTextEdit): Text area for displaying log output.

    Buttons, line edits and progress bars are PyQt-SiliconUI widgets (see widgets.py).
    """

    def __init__(self, windowName: str) -> None:
        """
        Initializes the main application window for the Data Wrangler tool.
        This constructor sets up the user interface, including layouts and components, and initializes necessary data handling objects.

        The `__init__` method configures the main window's title and size, creates instances of the data wrangler and allocator,
        and initializes the user interface components and logging. It establishes the layout for the application, ensuring
        that all elements are properly arranged for user interaction.

        Args:
            windowName (str): The title to be displayed in the main application window.

        Returns:
            None
        """
        super().__init__()
        self.wrangler: DataWrangler = DataWrangler()
        self.allocator: LatentDirichletAllocator = None
        self.pdialog: QProgressDialog = None
        # BERTopic training's own progress window (LDA uses self.pdialog).
        self.bertopic_pdialog: QProgressDialog = None
        self._allocator_error: str = ""
        self.run_thread: QThread = None
        self._run_job_callbacks = None
        self.mongo_thread: QThread = None
        self.process_thread: QThread = None
        self._process_total = 0
        # Set when the user closes the window while a job is running; the
        # window closes itself once every job has finished.
        self._close_pending = False
        self._base_title = windowName

        # Setting up the main application window
        self.setWindowTitle(windowName)
        self.setGeometry(100, 100, 1000, 860)

        # Main layout
        self.main_layout = QVBoxLayout()
        self.instruction_layout = QVBoxLayout()
        self.data_entry_layout = QVBoxLayout()

        # GUI Elements
        self.init_ui_components()
        self.init_menu()
        self.init_logging()

        # Set up the main widget and layout
        central_widget = QWidget()
        central_widget.setLayout(self.main_layout)
        self.setCentralWidget(central_widget)

    def init_ui_components(self):
        """
        Initializes the user interface components for the Data Wrangler application.
        This method sets up labels, buttons, input fields, and layouts to facilitate user interaction.

        The `init_ui_components` function creates and configures various UI elements, including instruction labels,
        buttons for file selection, and input fields for model parameters. It organizes these components into layouts
        and connects button actions to their respective functions, ensuring a cohesive user experience.

        Args:
            None

        Returns:
            None
        """
        # Instructions and warnings
        instruction_label = hint(
            "Choose the ticket file from the Zendesk Tickets API and the directory of "
            "JSON comments for each ticket, then Process Data. Write Tickets to MongoDB "
            "loads them into the decision engine's tickets collection (needs Mongo URI "
            "in Settings; Preferences on macOS)."
        )
        warning_label = hint(
            "Ticket and comment files must be in JSON format.", role="warning"
        )

        self.instruction_layout.addWidget(instruction_label)
        self.instruction_layout.addWidget(warning_label)

        # Buttons
        self.ticket_file_button = Button("Select Path to Ticket File")
        self.ticket_file = LineEdit("Ticket file")
        self.ticket_file.setPlaceholderText("Not chosen yet")

        self.comments_dir_button = Button("Select Path to Comments Directory")
        self.comments_dir = LineEdit("Comments")
        self.comments_dir.setPlaceholderText("Not chosen yet")

        self.process_button = Button("Process Data", primary=True)
        self.mongo_button = Button("Write Tickets to MongoDB")
        self.train_model_button = Button("Train Model", primary=True)
        # Saved runs survive a restart, so these work without processing data.
        self.previous_runs_button = Button("Show Previous Run…")
        self.assign_topics_button = Button("Assign Topics from Saved Run…")
        # BERTopic reads tickets and their embeddings from Mongo, not the files above.
        self.bertopic_button = Button("Train BERTopic Model…", primary=True)
        self.bertopic_runs_button = Button("Show Previous BERTopic Run…")

        self.comments_dir.setReadOnly(True)
        self.ticket_file.setReadOnly(True)

        self.ticket_file_button.clicked.connect(self.select_ticket_file)
        self.comments_dir_button.clicked.connect(self.select_comments_dir)
        self.process_button.clicked.connect(self.init_start_process)
        self.mongo_button.clicked.connect(self.start_mongo_worker)
        self.train_model_button.clicked.connect(self.start_allocator_worker)
        self.previous_runs_button.clicked.connect(self.show_previous_run)
        self.assign_topics_button.clicked.connect(self.start_assign_topics)
        self.bertopic_button.clicked.connect(self.start_bertopic_training)
        self.bertopic_runs_button.clicked.connect(self.show_previous_bertopic_run)
        self.train_model_button.setEnabled(False)

        # The training inputs, side by side
        self.form_layout = QHBoxLayout()
        self.num_topics_input = LineEdit("Number of Topics")
        self.iterations_input = LineEdit("Iterations")
        self.passes_input = LineEdit("Passes")

        for input_field in [
            self.num_topics_input,
            self.iterations_input,
            self.passes_input,
        ]:
            input_field.setClearButtonEnabled(True)
            input_field.setEnabled(False)
            self.form_layout.addWidget(input_field)

        disabled_notice = hint(
            "The next three inputs are disabled until data is processed."
        )

        # Log output display
        self.log_output = use_si_scrollbars(QTextEdit())
        self.log_output.setObjectName(theme.LOG_PANE)  # styled as a console
        self.log_output.setReadOnly(True)
        self.highlighter = LogHighlighter(self.log_output)
        sys.stdout = QTextEditStream(self.log_output)
        sys.stderr = QTextEditStream(self.log_output)
        self.log_output.setPlainText(
            f"Welcome, {os.getenv('USER', 'Learnosity Support Engineer')}!! Starting Lasso on {datetime.now().strftime('%A, %B %m, %Y')} at {datetime.now().strftime('%I:%M %p')}. \n Errors, logs and standard output will be show here... \n Well What Are You Waiting For?!?! Get to Work!"
        )
        # Arrange layout: one card per step, ticket data first.
        data_card = Card("Ticket Data")
        data_card.body.addLayout(self.instruction_layout)
        for field, button in [
            (self.ticket_file, self.ticket_file_button),
            (self.comments_dir, self.comments_dir_button),
        ]:
            row = QHBoxLayout()
            row.addWidget(field, stretch=1)
            row.addWidget(button)
            data_card.body.addLayout(row)
        data_card.body.addLayout(button_row(self.process_button, self.mongo_button))

        lda_card = Card("LDA Topic Model")
        lda_card.body.addWidget(disabled_notice)
        lda_card.body.addLayout(self.form_layout)
        lda_card.body.addLayout(
            button_row(
                self.train_model_button,
                self.previous_runs_button,
                self.assign_topics_button,
            )
        )

        bertopic_card = Card("Embedding Topics (BERTopic)")
        bertopic_card.body.addWidget(hint("Trains on the tickets in MongoDB."))
        bertopic_card.body.addLayout(
            button_row(self.bertopic_button, self.bertopic_runs_button)
        )

        self.data_entry_layout.setSpacing(12)
        self.data_entry_layout.addWidget(data_card)
        self.data_entry_layout.addWidget(lda_card)
        self.data_entry_layout.addWidget(bertopic_card)
        self.data_entry_layout.addStretch()

        # Topic modeling and the decision engine each get a tab; the log pane
        # sits below both so either tab's output stays visible. The topic tab
        # scrolls when the window is too short for its three cards.
        topic_page = QWidget()
        topic_page.setLayout(self.data_entry_layout)
        topic_tab = use_si_scrollbars(QScrollArea())
        topic_tab.setFrameShape(QFrame.NoFrame)
        topic_tab.setWidgetResizable(True)
        topic_tab.setWidget(topic_page)

        self.triage_tab = TriageTab()
        self.triage_tab.idle.connect(self._finish_pending_close)

        # spaCy tag grouping reuses the model already loaded for LDA.
        self.tag_tab = TagTab(
            lambda backend: (
                tag_normalizer.spacy_embedder(nlp)
                if backend == tag_normalizer.SPACY
                else tag_normalizer.embedder(backend)
            )
        )
        self.tag_tab.idle.connect(self._finish_pending_close)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.addTab(topic_tab, "Topic Modeling")
        self.tabs.addTab(self.triage_tab, "Decision Engine")
        self.tabs.addTab(self.tag_tab, "Tag Normalizer")

        log_card = Card("Log")
        log_card.body.addWidget(self.log_output)

        # Main layout arrangement
        self.main_layout.setContentsMargins(16, 8, 16, 16)
        self.main_layout.setSpacing(12)
        self.main_layout.addWidget(self.tabs, stretch=3)
        self.main_layout.addWidget(log_card, stretch=1)

    def init_menu(self) -> None:
        """Adds the Settings… action (the app menu's Preferences… on macOS)."""
        settings_action = QAction("Settings…", self)
        settings_action.setMenuRole(QAction.PreferencesRole)
        # The platform's Preferences shortcut where it has one; Windows defines
        # none, so fall back to Ctrl+, (Qt maps Ctrl to Command on macOS).
        standard = QKeySequence.keyBindings(QKeySequence.Preferences)
        settings_action.setShortcuts(standard or [QKeySequence("Ctrl+,")])
        settings_action.triggered.connect(self.open_settings)
        self.menuBar().addMenu("&File").addAction(settings_action)

    def open_settings(self) -> None:
        """Shows the settings dialog; saved values apply to the next action."""
        if SettingsDialog(self).exec_() != QDialog.Accepted:
            return
        # The Mongo and model clients are cached; rebuild them from the new values.
        reset_clients()
        if self.triage_tab._loaded and not self.triage_tab.is_busy():
            self.triage_tab.refresh()

    def prompt_for_missing_settings(self) -> None:
        """On launch, reports an unreadable settings file and offers the
        settings dialog if required settings are unset."""
        if settings.load_error:
            logger.error(settings.load_error)
            QMessageBox.warning(self, "Settings Not Loaded", settings.load_error)
        missing = settings.missing_required()
        if not missing:
            return
        answer = QMessageBox.question(
            self,
            "Setup Needed",
            settings.describe_missing(missing)
            + "\n\nTraining topic models works without these.\n\nOpen Settings now?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if answer == QMessageBox.Yes:
            self.open_settings()

    def confirm_file_change(self, type: str) -> None:

        warningMsg = QMessageBox(self, title="Update to File Selection!")
        warningMsg.setIcon("SP_FileIcon")
        warningMsg.setInformativeText(
            "You Are about To update the file path to a previously selected file"
        )
        warningMsg.setText("Would You Like to Continue?")
        warningMsg.setStandardButtons(QMessageBox.Yes | QMessageBox.Cancel)
        warningMsg.setDefaultButton(QMessageBox.Cancel)

        response = warningMsg.exec_()

        # Handle the user's response
        if response == QMessageBox.Yes:
            if type == "dir":
                self.select_comments_dir()
            else:
                self.select_ticket_file()

    def show_progress_bar(self, label: str = "Processing..."):
        """
        Displays a progress dialog with a progress bar to track the progress of long-running tasks.

        Args:
            label (str): The label to be displayed in the progress dialog.
        """
        # Create the progress dialog
        self.pdialog = QProgressDialog(label, "Cancel", 0, 100, self)
        self.pdialog.setWindowTitle("Progress")
        self.pdialog.setWindowModality(QtCore.Qt.WindowModal)
        self.pdialog.setMinimumDuration(0)  # Ensure it appears immediately
        self.pdialog.setAutoClose(False)  # Keep it open until we close it manually

        # Create the progress bar and add it to the dialog
        self.pbar = ProgressBar(self)
        self.pbar.setMaximum(100)
        self.pdialog.setBar(self.pbar)

        # Initialize the progress to 0
        self.pbar.setValue(0)

    def update_training_progress_bar(self, progress: int):
        """
        Updates the progress bar value in the progress dialog.

        Args:
            progress (int): The current progress value to be set.
        """
        if self.pdialog is not None:
            self.pdialog.setValue(progress)
            self.pbar.setValue(progress)

            # Close the dialog if progress is complete
            if progress >= 100:
                self.pdialog.close()

    def update_preprocess_progress(self, progress: int):
        self.update_training_progress_bar(progress // 2)

    def update_train_progress(self, progress: int):
        self.update_training_progress_bar(50 + progress // 2)

    def update_training_status(self, progress: str):
        """
        Updates the progress dialog label status in the progress dialog.

        Args:
            progress (str): The current status message to be shown.
        """
        if self.pdialog is not None:
            self.pdialog.setLabelText(progress)

    def start_allocator_worker(self):
        """
        Validates the training inputs, then preprocesses the corpus and trains the
        LDA models in a separate thread, updating the progress dialog as it goes.
        """
        number_of_topics = self.num_topics_input.text().strip()
        iterations = self.iterations_input.text().strip()
        passes = self.passes_input.text().strip()
        valid, error_message = LatentDirichletAllocator.validate_inputs(
            number_of_topics, iterations, passes
        )
        if not valid:
            logger.error(error_message)
            QMessageBox.warning(self, "Input Validation Error", error_message)
            return
        if self.run_thread is not None:
            QMessageBox.information(
                self, "Busy", "Wait for the current saved-run job to finish."
            )
            return
        if self._training_running():
            # The progress dialog closes at 100%, before charts are saved and the
            # model uploaded; a second job would replace the running thread.
            QMessageBox.information(
                self,
                "Busy",
                "Training is still saving the last model. Wait for it to finish.",
            )
            return

        self.train_model_button.setEnabled(False)
        iterations, passes, number_of_topics = (
            int(iterations),
            int(passes),
            int(number_of_topics),
        )
        # A run trained on the same tickets with the same settings would only
        # differ by its random seed, so offer it instead of retraining.
        dataset = topic_store.dataset_fingerprint(
            self.allocator.prelemma_corpus or [], self.allocator.document_ids
        )
        wanted = self.allocator.training_settings(number_of_topics, iterations, passes)
        preprocessing = topic_store.preprocessing_info(nlp, stop_words)
        self._start_run_job(
            lambda: topic_store.find_matching_run(
                self._known_runs(), dataset, wanted, preprocessing
            ),
            "Looking for a matching saved run",
            lambda match: self._train_or_reuse(
                match, iterations, passes, number_of_topics
            ),
            on_error=lambda message: self._train_or_reuse(
                None, iterations, passes, number_of_topics
            ),
        )

    def _training_running(self) -> bool:
        """True while the training thread (training, saving charts, uploading) is alive."""
        # self.thread shadows QObject.thread() only once training has started.
        thread = getattr(self, "thread", None)
        return isinstance(thread, QThread) and thread.isRunning()

    def _refresh_train_button(self) -> None:
        """Enables Train Model only with processed data and no training thread running."""
        self.train_model_button.setEnabled(
            self.allocator is not None
            and self.allocator.prelemma_corpus is not None
            and not self._training_running()
        )

    def _known_runs(self) -> List[dict]:
        """Saved runs from Mongo and disk; disk only if Mongo can't be reached. Worker thread."""
        try:
            return topic_store.known_runs(self._runs_collection())
        except Exception as e:
            logger.warning(
                f"Could not list runs from MongoDB ({e}); checking this computer's runs only."
            )
            return topic_store.known_runs()

    def _train_or_reuse(
        self, match, iterations: int, passes: int, number_of_topics: int
    ) -> None:
        """Trains, or, if a matching run exists and the user agrees, opens it instead."""
        if match is None:
            self._start_training(iterations, passes, number_of_topics)
            return
        box = QMessageBox(self)
        box.setWindowTitle("Matching Model Found")
        box.setIcon(QMessageBox.Question)
        where = (
            "on this computer"
            if topic_store.has_local_files(match)
            else (
                f"in {match['storage']['url']}"
                if match.get("storage")
                else "in MongoDB only (its files may be missing)"
            )
        )
        box.setText(
            f"Run {topic_store.describe_run(match)} was trained on the same "
            f"{(match.get('dataset') or {}).get('ticket_count', '?')} tickets with the same settings and preprocessing. "
            f"It is saved {where}.\n\nRetraining would give a near-identical model: only the random seed differs."
        )
        use = box.addButton("Use Saved Model", QMessageBox.AcceptRole)
        retrain = box.addButton("Retrain", QMessageBox.DestructiveRole)
        box.addButton(QMessageBox.Cancel)
        box.setDefaultButton(use)
        box.exec_()
        if box.clickedButton() is use:
            self._open_saved_run(match, on_failed=self._refresh_train_button)
        elif box.clickedButton() is retrain:
            self._start_training(iterations, passes, number_of_topics)
        else:
            self._refresh_train_button()

    def _open_saved_run(self, run: dict, on_failed=None) -> None:
        """Shows a saved run, downloading its files from S3 first if they aren't here."""

        def shown(local_run):
            self._refresh_train_button()
            try:
                self.present_results(local_run)
            except Exception as e:
                logger.exception(f"Showing run {local_run['run_id']} failed: {e}")
                self.notify_user_of_error(
                    (False, f"Could not show run {local_run['run_id']}: {e}")
                )

        def failed(message):
            self.notify_user_of_error(
                (False, f"Could not open run {run['run_id']}: {message}")
            )
            if on_failed:
                on_failed()

        if topic_store.has_local_files(run):
            # Fast and local: points the chart paths at the run's own folder.
            try:
                shown(topic_store.ensure_local(run))
            except (OSError, ValueError) as e:
                failed(str(e))
            return
        self._start_run_job(
            lambda: topic_store.ensure_local(run),
            f"Downloading run {run['run_id']}",
            shown,
            on_error=failed,
        )

    def _start_run_job(self, task, description: str, on_done, on_error=None) -> None:
        """Runs task on run_thread, then calls on_done(result) or on_error(message) on the GUI thread.

        The worker's signals go to this window's own slots (bound methods of
        a QObject living on the GUI thread), so Qt queues them there; the
        slots then stop the thread, which is no longer running the task.
        """
        self._run_job_callbacks = (
            on_done,
            on_error or (lambda m: self.notify_user_of_error((False, m))),
        )
        self.run_thread = worker_thread()
        self.run_worker = ThreadWorker(task, description)
        self.run_worker.moveToThread(self.run_thread)
        self.run_worker.finished.connect(self._run_job_finished)
        self.run_worker.error.connect(self._run_job_failed)
        self.run_thread.started.connect(self.run_worker.run)
        self.run_thread.start()

    @pyqtSlot(object)
    def _run_job_finished(self, result) -> None:
        self._end_run_job(0, result)

    @pyqtSlot(str)
    def _run_job_failed(self, message: str) -> None:
        self._end_run_job(1, message)

    def _end_run_job(self, which: int, value) -> None:
        callback = self._run_job_callbacks[which]
        self._run_job_callbacks = None
        self._stop_run_thread()
        if self._close_pending:
            self._finish_pending_close()
            return
        callback(value)

    def _start_training(
        self, iterations: int, passes: int, number_of_topics: int
    ) -> None:
        """Preprocesses the corpus and trains on the training thread."""
        self.train_model_button.setEnabled(False)
        self.show_progress_bar("Training LDA Model...")

        # Preprocessing fills the first half of the bar and training the second.
        self.allocator.preprocess_progress.connect(self.update_preprocess_progress)
        self.allocator.train_progress.connect(self.update_train_progress)
        self.allocator.worker_status.connect(self.update_training_status)
        self._allocator_error = ""
        self.allocator.error.connect(self._remember_allocator_error)

        # The allocator emits from the worker thread; these queued connections
        # deliver to this window, so the UI is only touched on the GUI thread.
        # process_corpus and model_trained report their own failures through
        # allocator.error and return False, so the worker finishes with None;
        # the worker's error signal covers anything else.
        self.thread = worker_thread()
        self.training_worker = ThreadWorker(
            lambda: self._train_and_save(iterations, passes, number_of_topics),
            "Training the LDA model",
        )
        self.training_worker.moveToThread(self.thread)
        self.training_worker.finished.connect(self.on_worker_finished)
        self.training_worker.error.connect(self.on_worker_error)
        self.thread.started.connect(self.training_worker.run)
        self.thread.start()

    def _train_and_save(self, iterations: int, passes: int, num_of_topics: int):
        """Trains, then saves the run and links it to Mongo. Runs on the worker thread.

        The model files are saved locally first; recording the run and
        assigning topics in Mongo is skipped (with a warning on the run) when
        MONGO_URI is unset or Mongo fails, so a trained model is never lost.

        Returns:
            The run document, or None if training failed.
        """
        trained = self.allocator.run(
            nlp=nlp,
            stopwords=stop_words,
            wranglerInstance=self.wrangler,
            iterations=iterations,
            passes=passes,
            num_of_topics=num_of_topics,
        )
        if not trained:
            return None
        self.allocator.worker_status.emit("Saving the model and its charts...")
        run = topic_store.save_run(self.allocator, nlp, stop_words)
        self._upload_run(run)
        if not os.getenv("MONGO_URI"):
            run["warnings"].append(
                "MONGO_URI is not set: the run was saved locally only and no ticket topics were written."
            )
            return run
        self.allocator.worker_status.emit(
            "Recording the run and ticket topics in MongoDB..."
        )
        try:
            database = get_database()
            topic_store.record_run(database[topic_store.runs_collection()], run)
            run["assignment"] = topic_store.assign_trained_topics(
                database[os.environ.get("MONGO_COLLECTION", "tickets")],
                run,
                self.allocator,
            )
        except Exception as e:
            logger.exception(f"Could not record run {run['run_id']} in MongoDB: {e}")
            run["warnings"].append(
                f"Saved locally, but MongoDB could not be updated: {e}"
            )
        return run

    def _upload_run(self, run: dict) -> None:
        """Copies the saved run to S3 when MODEL_BUCKET is set. Runs on the worker thread.

        A failure becomes a warning on the run: the model is already saved
        locally and can be uploaded later with `python -m model_storage upload`.
        """
        try:
            cfg = model_storage.config()
        except model_storage.StorageNotConfigured as e:
            run["warnings"].append(f"Not uploaded to S3: {e}")
            return
        if cfg is None:
            return
        self.allocator.worker_status.emit(
            f"Uploading the model to s3://{cfg.bucket}/..."
        )
        try:
            model_storage.upload_and_record(run, cfg)
        except Exception as e:
            logger.exception(f"Could not upload run {run['run_id']} to S3: {e}")
            run["warnings"].append(
                f"Saved locally, but the upload to s3://{cfg.bucket} failed: {e}. "
                "Retry with: python -m model_storage upload " + run["run_id"]
            )

    def _remember_allocator_error(self, message: str):
        self._allocator_error = message

    def _disconnect_allocator(self):
        for signal in [
            self.allocator.preprocess_progress,
            self.allocator.train_progress,
            self.allocator.worker_status,
            self.allocator.error,
        ]:
            signal.disconnect()

    def _stop_worker_thread(self):
        self._disconnect_allocator()
        if self.pdialog is not None:
            self.pdialog.close()
        self.thread.quit()
        self.thread.wait()
        self.training_worker = None
        self.train_model_button.setEnabled(True)

    def on_worker_finished(self, run):
        """
        Handles the completion of the worker process, cleaning up the thread and
        presenting the results.

        Args:
            run: The saved run document, or None if training failed (the
                allocator's error message is shown instead).
        """
        self._stop_worker_thread()
        if self._close_pending:
            if run is None:
                logger.error(f"Training failed while closing: {self._allocator_error}")
            else:
                logger.info(
                    f"Training finished while closing; saved as run {run['run_id']}."
                )
            self._finish_pending_close()
            return
        if run is None:
            self.notify_user_of_error(
                (False, self._allocator_error or "Training failed; see the log.")
            )
            return
        message = f"LDA Model training has finished successfully.\n\nSaved as run {run['run_id']}."
        if run.get("storage"):
            message += f"\nUploaded to {run['storage']['url']}."
        if run.get("assignment"):
            message += f"\nTopics written to {run['assignment']['assigned']} ticket(s) in MongoDB."
        if run.get("warnings"):
            message += "\n\nWarning:\n" + "\n".join(run["warnings"])
        QMessageBox.information(self, "Process Complete", message)
        # This runs as a Qt slot, where an unhandled exception aborts the app.
        try:
            self.present_results(run)
        except Exception as e:
            logger.exception(f"Presenting results failed: {e}")
            self.notify_user_of_error(
                (False, f"Training finished, but the results could not be shown: {e}")
            )

    def on_worker_error(self, message: str):
        """Handles a failure in the worker thread."""
        self._stop_worker_thread()
        if self._close_pending:
            logger.error(f"Training failed while closing: {message}")
            self._finish_pending_close()
            return
        self.notify_user_of_error((False, message))

    def start_mongo_worker(self):
        """
        Writes every ticket in the selected ticket file to MongoDB in a separate
        thread, updating the progress dialog as it goes.

        Inserts new tickets and appends only unseen comments to existing ones;
        see DataWrangler.write_ticket_to_mongo. Independent of Process Data.
        """
        if not os.getenv("MONGO_URI"):
            message = "MONGO_URI is not set. Set it in Settings (Preferences on macOS)."
            logger.error(message)
            QMessageBox.warning(self, "MongoDB Not Configured", message)
            return
        if not pin_dialog.allowed(self, spend_guard.WRITE_DATA, "write tickets to MongoDB"):
            return
        try:
            with open(self.wrangler.ticket_file, "r") as tickets_file:
                tickets = json.load(tickets_file)
        except (OSError, json.JSONDecodeError) as e:
            self.notify_user_of_error(
                (False, f"Error: Could not read the ticket file: {e}")
            )
            return

        self.mongo_ticket_count = len(tickets)
        self.mongo_button.setEnabled(False)
        self.show_progress_bar("Writing Tickets to MongoDB...")

        self.mongo_thread = worker_thread()
        self.mongo_worker = ThreadWorker(
            lambda: self.wrangler.write_tickets_to_mongo(tickets),
            "Writing tickets to MongoDB",
        )
        self.mongo_worker.moveToThread(self.mongo_thread)

        # The wrangler emits from the worker thread; these queued connections
        # deliver to this window on the GUI thread.
        self.wrangler.mongo_write_progress.connect(self.update_mongo_progress)
        self.wrangler.worker_status.connect(self.update_training_status)
        self.mongo_worker.finished.connect(self.on_mongo_finished)
        self.mongo_worker.error.connect(self.on_mongo_error)
        self.mongo_thread.started.connect(self.mongo_worker.run)
        self.mongo_thread.start()

    def update_mongo_progress(self, written: int):
        self.update_training_progress_bar(
            written * 100 // max(self.mongo_ticket_count, 1)
        )

    def _stop_mongo_thread(self):
        self.wrangler.mongo_write_progress.disconnect(self.update_mongo_progress)
        self.wrangler.worker_status.disconnect(self.update_training_status)
        if self.pdialog is not None:
            self.pdialog.close()
        self.mongo_thread.quit()
        self.mongo_thread.wait()
        self.mongo_thread = None
        self.mongo_worker = None
        self.mongo_button.setEnabled(True)

    def on_mongo_finished(self, summaries: list):
        """Cleans up the thread and summarises what was written."""
        self._stop_mongo_thread()
        if self._close_pending:
            logger.info(
                f"MongoDB write finished while closing ({len(summaries)} tickets)."
            )
            self._finish_pending_close()
            return
        inserted = sum(summary["inserted"] for summary in summaries)
        updated = len(summaries) - inserted
        new_comments = sum(summary["new_comments"] for summary in summaries)
        message = (
            f"Wrote {len(summaries)} tickets to MongoDB: "
            f"{inserted} new, {updated} updated, {new_comments} new comments."
        )
        logger.success(message)
        QMessageBox.information(self, "MongoDB Write Complete", message)

    def on_mongo_error(self, message: str):
        """Handles a failure in the MongoDB write thread."""
        self._stop_mongo_thread()
        if self._close_pending:
            logger.error(f"MongoDB write failed while closing: {message}")
            self._finish_pending_close()
            return
        self.notify_user_of_error((False, message))

    def init_logging(self) -> None:
        """
        Initializes the logging configuration for the application.
        This method sets up the logger to output messages to the standard output with a specified format and log level.

        The `init_logging` method configures the logger to display messages in a colorized format that includes the timestamp,
        log level, and message content. The logging level is set to INFO, allowing informational messages and above to be logged.

        Args:
            None

        Returns:
            None
        """
        logger.level(
            "USER INPUT REQUIRED",
            no=26,
            color="<bold><blue>",
            icon="🤦🏾‍♂️",
        )
        logger.add(
            sys.stdout,
            colorize=True,
            format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
            level="INFO",
            diagnose=False,
        )
        logger.add(
            sink="./wrangle_log.log",
            colorize=True,
            serialize=True,
            format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
            level="DEBUG",
            diagnose=False,
        )

    def update_progressBar(self, value):
        self.progressBar.setValue(value)

    def select_ticket_file(self):
        """Opens a file dialog to select a JSON ticket file.

        This function displays a file dialog that allows the user to select a
        JSON file. Upon selection, it logs the file path and updates the
        ticket file in the wrangler.

        Args:
            None

        Returns:
            None
        """
        file_dialog = QFileDialog()
        file_dialog.setNameFilters(["JSON files (*.json)"])
        if file_dialog.exec_():
            file_path = file_dialog.selectedFiles()[0]
            logger.info(f"Ticket File Selected: {file_path}")
            self.wrangler.ticket_file = file_path

    def select_comments_dir(self):
        """
        Opens a file dialog for the user to select a directory containing comment files.
        This method updates the comments directory in the data wrangler with the selected path.

        The `select_comments_dir` function utilizes a file dialog to allow the user to choose a directory.
        Upon selection, it logs the chosen directory path and assigns it to the `comments_dir` attribute of the wrangler.

        Args:
            None

        Returns:
            None
        """
        if dir_path := QFileDialog.getExistingDirectory():
            logger.info(f"Comments Dir Selected: {dir_path}")
            self.wrangler.comments_dir = dir_path

    def present_results(self, run: dict) -> None:
        """
        Displays the results of a saved training run.

        Shows the run's topics and coherence plot in a results window, and
        opens the run's interactive pyLDAvis chart in the browser. Works for
        the run just trained and for previous runs.

        Args:
            run: A run document from topic_store.

        Returns:
            None
        """
        charts = run.get("charts") or {}
        # Paths from an older run may point at files since deleted.
        chart_path, plot_path = (
            path if path and pathlib.Path(path).exists() else None
            for path in (charts.get("lda_html"), charts.get("coherence_png"))
        )
        # Kept on self: a dialog with no reference is garbage-collected and closes.
        self.results_dialog = ResultsDialog(
            best_num_topics=run.get("best_num_topics"),
            topics=[
                (t["topic_id"], t["share"], t["words"]) for t in run.get("topics") or []
            ],
            chart_path=chart_path,
            plot_path=plot_path,
            parent=self,
        )
        self.results_dialog.setWindowTitle(
            f"LDA Results · {topic_store.describe_run(run)}"
        )
        self.results_dialog.show()
        self.results_dialog.open_chart()

    def _runs_collection(self, name: Optional[str] = None):
        """A runs collection (LDA by default), or None to read runs from disk."""
        if not os.getenv("MONGO_URI"):
            return None
        return get_database()[name or topic_store.runs_collection()]

    def _choose_run(self, title: str):
        """Asks the user to pick a previous run; returns it, or None."""
        runs = self._known_runs()
        if not runs:
            QMessageBox.information(
                self, title, "No saved topic-model runs were found."
            )
            return None
        labels = [topic_store.describe_run(run) for run in runs]
        label, ok = QInputDialog.getItem(self, title, "Run:", labels, 0, False)
        return runs[labels.index(label)] if ok else None

    def show_previous_run(self) -> None:
        """Shows a previous run's topics and coherence."""
        if self.run_thread is not None:
            QMessageBox.information(
                self, "Busy", "Wait for the current saved-run job to finish."
            )
            return
        run = self._choose_run("Previous Topic-Model Runs")
        if run is None:
            return
        self._open_saved_run(run)

    def start_assign_topics(self) -> None:
        """Loads a saved run and writes its topics to every ticket in MongoDB."""
        if not os.getenv("MONGO_URI"):
            QMessageBox.warning(
                self,
                "MongoDB Not Configured",
                "MONGO_URI is not set. Set it in Settings (Preferences on macOS).",
            )
            return
        if self.run_thread is not None:
            QMessageBox.information(
                self, "Busy", "Wait for the current saved-run job to finish."
            )
            return
        run = self._choose_run("Assign Topics from Saved Run")
        if run is None:
            return
        self.assign_topics_button.setEnabled(False)
        self.run_thread = worker_thread()
        self.run_worker = ThreadWorker(
            lambda: topic_store.assign_from_saved_run(
                get_database()[os.environ.get("MONGO_COLLECTION", "tickets")],
                run,
                nlp,
                stop_words,
            ),
            f"Assigning topics from run {run['run_id']}",
        )
        self.run_worker.moveToThread(self.run_thread)
        self.run_worker.finished.connect(
            lambda summary: self._assign_topics_done(run, summary)
        )
        self.run_worker.error.connect(self._assign_topics_failed)
        self.run_thread.started.connect(self.run_worker.run)
        self.run_thread.start()

    def _stop_run_thread(self) -> None:
        self.run_thread.quit()
        self.run_thread.wait()
        self.run_thread = None
        self.run_worker = None
        self.assign_topics_button.setEnabled(True)

    def _assign_topics_done(self, run: dict, summary: dict) -> None:
        self._stop_run_thread()
        if self._close_pending:
            logger.info(
                f"Topic assignment for run {run['run_id']} finished while closing: {summary}"
            )
            self._finish_pending_close()
            return
        message = (
            f"Run {run['run_id']}: topics written to {summary['assigned']} ticket(s); "
            f"{summary['empty']} had no words the model knows."
        )
        logger.success(message)
        QMessageBox.information(self, "Topics Assigned", message)

    def _assign_topics_failed(self, message: str) -> None:
        self._stop_run_thread()
        if self._close_pending:
            logger.error(f"Topic assignment failed while closing: {message}")
            self._finish_pending_close()
            return
        self.notify_user_of_error((False, message))

    # --- BERTopic --------------------------------------------------------------

    def start_bertopic_training(self) -> None:
        """Reads the tickets' outcomes, asks for BERTopic options, then trains and
        saves a run on run_thread."""
        if not os.getenv("MONGO_URI"):
            QMessageBox.warning(
                self,
                "MongoDB Not Configured",
                "BERTopic reads tickets and their embeddings from MongoDB. "
                "Set MONGO_URI in Settings (Preferences on macOS).",
            )
            return
        if self._training_running():
            QMessageBox.information(
                self, "Busy", "Wait for the LDA model training to finish."
            )
            return
        if self.run_thread is not None:
            QMessageBox.information(
                self, "Busy", "Wait for the current saved-run job to finish."
            )
            return
        # The outcome counts come from Mongo, so read them off the GUI thread.
        self._start_run_job(
            lambda: bertopic_model.outcome_counts(self._tickets_collection()),
            "Reading ticket outcomes",
            self._ask_bertopic_options,
        )

    def _tickets_collection(self):
        return get_database()[os.environ.get("MONGO_COLLECTION", "tickets")]

    def _ask_bertopic_options(self, outcome_counts) -> None:
        dialog = BertopicOptionsDialog(
            bertopic_runs.lda_runs(), outcome_counts, parent=self
        )
        if dialog.exec_() != QDialog.Accepted:
            return
        self._train_bertopic(dialog.options())

    def _train_bertopic(self, options: bertopic_runs.TrainOptions) -> None:
        self.bertopic_button.setEnabled(False)
        # Its own progress window and status target, apart from the LDA job's
        # self.pdialog. No cancel: a fit in progress can't be interrupted.
        self.bertopic_pdialog = QProgressDialog(
            "Training BERTopic model...", None, 0, 0, self
        )
        self.bertopic_pdialog.setWindowTitle("Progress")
        self.bertopic_pdialog.setWindowModality(QtCore.Qt.WindowModal)
        self.bertopic_pdialog.setMinimumDuration(0)
        self.bertopic_pdialog.show()
        self.bertopic_status = StatusRelay()
        self.bertopic_status.message.connect(self._update_bertopic_status)

        def done(run):
            self._end_bertopic_job()
            message = f"BERTopic training has finished.\n\nSaved as run {run['run_id']} in {run['model_dir']}."
            if run.get("storage"):
                message += f"\nUploaded to {run['storage']['url']}."
            if run.get("assignment") and run["assignment"].get("status") == "complete":
                # Anything short of complete is already among the warnings.
                message += "\n" + bertopic_model.describe_assignment(
                    run["assignment"], run["run_id"]
                )
            if run.get("warnings"):
                message += "\n\nWarning:\n" + "\n".join(run["warnings"])
            QMessageBox.information(self, "BERTopic Complete", message)
            self.present_bertopic_results(run)

        def failed(message):
            self._end_bertopic_job()
            self.notify_user_of_error((False, message))

        # As LDA runs: saved here, uploaded when MODEL_BUCKET is set, recorded in bertopic_run_log.
        self._start_run_job(
            lambda: bertopic_runs.train_and_save(
                self._tickets_collection(),
                nlp,
                stop_words,
                options,
                status=self.bertopic_status.message.emit,
                runs=self._runs_collection(bertopic_runs.runs_collection()),
                storage=model_storage.config,
            ),
            "Training the BERTopic model",
            done,
            on_error=failed,
        )

    def _update_bertopic_status(self, message: str) -> None:
        if self.bertopic_pdialog is not None:
            self.bertopic_pdialog.setLabelText(message)

    def _end_bertopic_job(self) -> None:
        if self.bertopic_pdialog is not None:
            self.bertopic_pdialog.close()
            self.bertopic_pdialog = None
        self.bertopic_status.message.disconnect()
        self.bertopic_button.setEnabled(True)

    def present_bertopic_results(self, run: dict) -> None:
        """Shows a BERTopic run's topics, scores and chart buttons."""
        # This runs as a Qt slot, where an unhandled exception aborts the app.
        try:
            # Kept on self: a dialog with no reference is garbage-collected and closes.
            subjects = None
            if os.getenv("MONGO_URI"):
                subjects = lambda r: bertopic_runs.example_subjects(
                    self._tickets_collection(), r
                )
            previous = getattr(self, "bertopic_results_dialog", None)
            if previous is not None:
                # Replacing it frees it; a subject lookup in flight must finish first.
                previous.wait_for_export()
            self.bertopic_results_dialog = BertopicResultsDialog(
                run, bertopic_runs.chart_paths(run), self, subjects=subjects
            )
            self.bertopic_results_dialog.show()
        except Exception as e:
            logger.exception(f"Showing BERTopic run {run.get('run_id')} failed: {e}")
            self.notify_user_of_error(
                (False, f"Could not show BERTopic run {run.get('run_id')}: {e}")
            )

    def _known_bertopic_runs(self) -> List[dict]:
        """BERTopic runs from Mongo and disk; disk only if Mongo can't be reached."""
        try:
            return bertopic_runs.known_runs(
                self._runs_collection(bertopic_runs.runs_collection())
            )
        except Exception as e:
            logger.warning(
                f"Could not list BERTopic runs from MongoDB ({e}); showing this computer's runs only."
            )
            return bertopic_runs.known_runs()

    def show_previous_bertopic_run(self) -> None:
        """Lets the user pick a BERTopic run (from MongoDB or this computer) and shows it,
        downloading its files from S3 first if they aren't here."""
        title = "Previous BERTopic Runs"
        if self.run_thread is not None:
            QMessageBox.information(
                self, "Busy", "Wait for the current saved-run job to finish."
            )
            return
        runs = self._known_bertopic_runs()
        if not runs:
            QMessageBox.information(
                self,
                title,
                f"No BERTopic runs were found in MongoDB or "
                f"{bertopic_model.model_dir()}.",
            )
            return
        labels = [
            f"{bertopic_runs.describe_run(run)} · {bertopic_runs.where(run)}"
            for run in runs
        ]
        label, ok = QInputDialog.getItem(self, title, "Run:", labels, 0, False)
        if not ok:
            return
        run = runs[labels.index(label)]
        if bertopic_runs.has_local_files(run) or not run.get("storage"):
            # Its topics and scores are in the record; the charts need its files.
            self.present_bertopic_results(run)
            return

        def failed(message):
            self.notify_user_of_error(
                (
                    False,
                    f"Could not download BERTopic run {run['run_id']}: {message}\n\n"
                    "Showing its topics and scores without the charts.",
                )
            )
            self.present_bertopic_results(run)

        self._start_run_job(
            lambda: bertopic_runs.ensure_local(run),
            f"Downloading BERTopic run {run['run_id']}",
            self.present_bertopic_results,
            on_error=failed,
        )

    # --- shutdown --------------------------------------------------------------

    def _app_threads(self) -> List[Tuple[str, QThread]]:
        """This window's own worker threads, with a label for each."""
        # self.thread shadows QObject.thread() until training first runs.
        candidates = [
            ("model training", getattr(self, "thread", None)),
            ("MongoDB write", self.mongo_thread),
            ("data processing", self.process_thread),
            ("saved-run job", self.run_thread),
        ]
        return [(label, t) for label, t in candidates if isinstance(t, QThread)]

    def running_jobs(self) -> List[str]:
        """Labels of the background jobs still running, in this window and its tabs."""
        jobs = [label for label, t in self._app_threads() if t.isRunning()]
        if self.triage_tab.is_busy():
            jobs.append("Decision Engine job")
        if self.tag_tab.is_busy():
            jobs.append("tag embedding")
        return jobs

    def closeEvent(self, event) -> None:
        """Defers closing while a background job runs.

        Destroying a running QThread aborts the app, and a Mongo or model call
        in flight can't be interrupted safely. So the close is ignored, the
        Decision Engine tab cancels the rest of any review batch, and the
        window closes itself once every job has finished and its thread has
        been joined (see _finish_pending_close).
        """
        jobs = self.running_jobs()
        if not jobs:
            event.accept()
            return
        event.ignore()
        self.triage_tab.request_shutdown()
        self.tag_tab.request_shutdown()
        self.setWindowTitle(
            f"{self._base_title} — closing after {', '.join(jobs)} finishes…"
        )
        if not self._close_pending:
            self._close_pending = True
            logger.warning(
                f"Close requested; waiting for {', '.join(jobs)} to finish before closing."
            )

    def _finish_pending_close(self) -> None:
        """Closes the window if a close was deferred and every job is now done."""
        if not self._close_pending:
            return
        jobs = self.running_jobs()
        if jobs:
            self.setWindowTitle(
                f"{self._base_title} — closing after {', '.join(jobs)} finishes…"
            )
            return
        # Let the finishing job's slot return before the window goes away.
        QTimer.singleShot(0, self.close)

    def join_threads_on_quit(self) -> None:
        """Last resort on application quit: waits for any job still running.

        Normally closeEvent has already waited. This covers quitting by
        another path; it blocks, but never lets a running QThread be destroyed.
        """
        self.triage_tab.wait_for_job()
        self.tag_tab.wait_for_job()
        if getattr(self, "bertopic_results_dialog", None) is not None:
            self.bertopic_results_dialog.wait_for_export()
        for label, thread in self._app_threads():
            if thread.isRunning():
                logger.warning(f"Waiting for {label} to finish before quitting…")
                thread.quit()
                thread.wait()

    def notify_user_of_error(self, error: Tuple[bool, str]) -> QMessageBox:
        """
        Displays a critical error message to the user.
        This method creates a message box that informs the user of an error and provides an option to abort the operation.

        The `notify_user_of_error` function takes an error tuple, where the second element contains the error message.
        It then presents this message in a critical message box, allowing the user to acknowledge the error.

        Args:
            error (Tuple[bool, str]): A tuple containing a boolean status and an error message string.

        Returns:
            QMessageBox: The message box displayed to the user.
        """
        return QMessageBox.critical(self, "Critical Error", error[1])

    def init_start_process(self) -> None:
        """Processes the selected ticket file and comments on a worker thread, with a progress bar.

        Reshapes the tickets, binds each one's comments, builds the corpus and
        writes the processed JSON files (see _process_data). Then sets up the
        allocator and enables the training inputs.
        """
        if self.process_thread is not None:
            return
        self.process_button.setEnabled(False)
        self.show_progress_bar("Processing Data...")
        self.pdialog.setCancelButton(None)  # the steps can't be interrupted
        self._process_total = 0

        # The wrangler emits from the worker thread; these queued connections
        # deliver to this window on the GUI thread.
        self.wrangler.tickets_loaded.connect(self._process_tickets_loaded)
        self.wrangler.ticket_progress.connect(self._process_reshape_progress)
        self.wrangler.binding_progress.connect(self._process_bind_progress)
        self.wrangler.corpus_creation_progress.connect(self._process_corpus_progress)
        self.wrangler.worker_status.connect(self.update_training_status)

        self.process_thread = worker_thread()
        self.process_worker = ThreadWorker(self._process_data, "Processing data")
        self.process_worker.moveToThread(self.process_thread)
        self.process_worker.finished.connect(self.on_process_finished)
        self.process_worker.error.connect(self.on_process_error)
        self.process_thread.started.connect(self.process_worker.run)
        self.process_thread.start()

    def _process_data(self) -> Tuple[List[str], List[str]]:
        """Runs the wrangling steps on the worker thread.

        Returns:
            (one document per ticket, their Zendesk IDs).

        Raises:
            RuntimeError: a step failed (its details are in the log).
        """
        if not self.wrangler.tickets_reshaped():
            raise RuntimeError("Failed to reshape tickets. Check the ticket file.")
        if not self.wrangler.comments_bound():
            raise RuntimeError("Failed to bind comments. Check the comments directory.")
        if not self.wrangler.create_corpus():
            raise RuntimeError("Failed to create corpus. The corpus is empty.")
        self.wrangler.worker_status.emit(
            "Writing the processed tickets and corpus files..."
        )
        self.wrangler.generate_json()
        return self.wrangler.ticket_documents(), self.wrangler.ticket_ids()

    def _process_tickets_loaded(self, count: int) -> None:
        self._process_total = count

    def _process_reshape_progress(self, done: int) -> None:
        self.update_training_progress_bar(
            process_progress("reshape", done, self._process_total)
        )

    def _process_bind_progress(self, done: int) -> None:
        self.update_training_progress_bar(
            process_progress("bind", done, self._process_total)
        )

    def _process_corpus_progress(self, done: int) -> None:
        self.update_training_progress_bar(
            process_progress("corpus", done, self._process_total)
        )

    def _stop_process_thread(self) -> None:
        for signal, slot in (
            (self.wrangler.tickets_loaded, self._process_tickets_loaded),
            (self.wrangler.ticket_progress, self._process_reshape_progress),
            (self.wrangler.binding_progress, self._process_bind_progress),
            (self.wrangler.corpus_creation_progress, self._process_corpus_progress),
            (self.wrangler.worker_status, self.update_training_status),
        ):
            signal.disconnect(slot)
        if self.pdialog is not None:
            self.pdialog.close()
        self.process_thread.quit()
        self.process_thread.wait()
        self.process_thread = None
        self.process_worker = None

    def on_process_finished(self, result: Tuple[List[str], List[str]]) -> None:
        """Sets up the allocator from the processed tickets and enables training."""
        self._stop_process_thread()
        if self._close_pending:
            logger.info("Data processing finished while closing.")
            self._finish_pending_close()
            return
        documents, ids = result
        self.allocator = LatentDirichletAllocator(num_of_topics=30)
        # One document per ticket; lemmatization runs in the training thread.
        self.allocator.prelemma_corpus = documents
        self.allocator.document_ids = ids
        self.train_model_button.setEnabled(True)
        for input_field in [
            self.num_topics_input,
            self.iterations_input,
            self.passes_input,
        ]:
            input_field.setEnabled(True)

        success_message = "The Data Located in the Provided Paths Has been Wrangled 🐄 and Massaged 💆🏽‍♂️...Please select Number of Topics, Iterations and Passes. Then Click Train Model to continue."
        logger.success("Data successfully wrangled and saved.")
        logger.log("USER INPUT REQUIRED", success_message)
        QMessageBox.information(
            self,
            "Data Has Successfully Processed",
            f"Processed {len(ids)} tickets.\n\nPlease select Number of Topics, Iterations and Passes. \n"
            "Then Click Train Model to continue.",
        )

    def on_process_error(self, message: str) -> None:
        """Reports a failed step; Process Data can be run again."""
        self._stop_process_thread()
        if self._close_pending:
            logger.error(f"Data processing failed while closing: {message}")
            self._finish_pending_close()
            return
        # A partial run leaves tickets behind; start the next run clean.
        self.wrangler.wrangled_tickets = []
        self.process_button.setEnabled(True)
        self.notify_user_of_error((False, message))


# The packaged build bundles the icon under assets/ (see packaging/lasso.spec).
ICON_PATH = (
    pathlib.Path(sys._MEIPASS) / "assets"
    if settings.is_frozen()
    else pathlib.Path(__file__).resolve().parents[1] / "packaging"
) / "icon.png"

# Before the QApplication: the macOS menu bar reads the name when it starts.
macos.set_app_name("Lasso")
app = QApplication(sys.argv)
app.setApplicationName("Lasso")
app.setWindowIcon(QIcon(str(ICON_PATH)))
theme.apply(app)
window = MainWindow("Lasso — Support Ticket Triage")
app.aboutToQuit.connect(window.join_threads_on_quit)
app.aboutToQuit.connect(chart_viewer.shutdown)

if __name__ == "__main__":
    # Paid calls in the app need a PIN per job, never a terminal prompt.
    spend_guard.gui_mode()
    try:
        logger.info(
            f"Paid model and embedding calls ask for the spending PIN, which is in "
            f"{spend_guard.ensure_pin()} ({spend_guard.identity()}: "
            f"{spend_guard.role()})."
        )
    except (spend_guard.SecurityError, OSError) as e:
        logger.error(f"Spending PIN unavailable, so paid calls will be refused: {e}")
    window.show()
    QTimer.singleShot(0, window.prompt_for_missing_settings)
    sys.exit(app.exec_())
