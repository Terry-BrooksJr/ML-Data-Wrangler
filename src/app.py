import json
import os
import pathlib
import sys
import time
from typing import Any, Callable, List, Tuple, Union
from datetime import datetime
from loguru import logger
from PyQt5.QtCore import QRect, pyqtSignal, pyqtSlot, QThread, QCoreApplication, QObject, QMutex, QTimer
from PyQt5.QtGui import QFont
from PyQt5 import QtCore
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QFormLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QProgressDialog,
    QProgressBar,
    QDialog,
    QTabWidget,
)

import topic_store
from decision_engine.persistence import get_database
from LDA_logic import LatentDirichletAllocator, stopwords
from results_dialog import ResultsDialog
from triage_tab import TriageTab
from utility import LogHighlighter, QTextEditStream, ThreadWorker
from wrangler import DataWrangler
import en_core_web_lg


nlp = en_core_web_lg.load()
stop_words: List[str] = stopwords.words("english")


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
        data_entry_layout (QVBoxLayout): Layout for data entry components.
        ticket_file_button (QPushButton): Button for selecting the ticket file.
        comments_dir_button (QPushButton): Button for selecting the comments directory.
        process_button (QPushButton): Button for processing the data.
        train_model_button (QPushButton): Button for training the model.
        num_topics_input (QLineEdit): Input field for the number of topics.
        iterations_input (QLineEdit): Input field for the number of iterations.
        passes_input (QLineEdit): Input field for the number of passes.
        log_output (QTextEdit): Text area for displaying log output.
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
        self._allocator_error: str = ""
        self.run_thread: QThread = None
        self.mongo_thread: QThread = None
        # Set when the user closes the window while a job is running; the
        # window closes itself once every job has finished.
        self._close_pending = False
        self._base_title = windowName

        # Setting up the main application window
        self.setWindowTitle(windowName)
        self.setGeometry(100, 100, 900, 750)

        # Main layout
        self.main_layout = QVBoxLayout()
        self.instruction_layout = QVBoxLayout()
        self.data_entry_layout = QVBoxLayout()

        # GUI Elements
        self.init_ui_components()
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
        instruction_label = QLabel(
            "Welcome to the Data Wrangler. \n\n"
            "Identify two file locations.\n"
            "1. The Ticket File Path from the ZenDesk Tickets API.\n"
            "2. The path to the comments directory with JSON comments for each ticket.\n\n"
            "Write Tickets to MongoDB loads them into the decision engine's tickets collection (needs MONGO_URI)."
        )
        warning_label = QLabel("NOTE: Ticket and comment files must be in JSON format.")
        warning_label.setStyleSheet("color: red;")

        self.instruction_layout.addWidget(instruction_label)
        self.instruction_layout.addWidget(warning_label)

        # Buttons
        self.wrangling_input_form = QFormLayout()
        self.ticket_file_button = QPushButton("Select Path to Ticket File")
        self.ticket_file = QLineEdit()

        self.comments_dir_button = QPushButton("Select Path to Comments Directory")
        self.comments_dir = QLineEdit()

        self.process_button = QPushButton("Process Data")
        self.mongo_button = QPushButton("Write Tickets to MongoDB")
        self.train_model_button = QPushButton("Train Model")
        # Saved runs survive a restart, so these work without processing data.
        self.previous_runs_button = QPushButton("Show Previous Run…")
        self.assign_topics_button = QPushButton("Assign Topics from Saved Run…")

        self.comments_dir.setReadOnly(True)
        self.ticket_file.setReadOnly(True)

        self.ticket_file_button.clicked.connect(self.select_ticket_file)
        self.comments_dir_button.clicked.connect(self.select_comments_dir)
        self.process_button.clicked.connect(self.init_start_process)
        self.mongo_button.clicked.connect(self.start_mongo_worker)
        self.train_model_button.clicked.connect(self.start_allocator_worker)
        self.previous_runs_button.clicked.connect(self.show_previous_run)
        self.assign_topics_button.clicked.connect(self.start_assign_topics)
        self.train_model_button.setEnabled(False)

        # Form layout for inputs
        self.form_layout = QFormLayout()
        self.num_topics_input = QLineEdit()
        self.iterations_input = QLineEdit()
        self.passes_input = QLineEdit()

        for input_field in [
            self.num_topics_input,
            self.iterations_input,
            self.passes_input,
        ]:
            input_field.setClearButtonEnabled(True)
            input_field.setEnabled(False)

        disabled_notice = QLabel(
            "The next three inputs are disabled until data is processed."
        )
        self.form_layout.addRow(disabled_notice)
        self.form_layout.addRow("Number of Topics:", self.num_topics_input)
        self.form_layout.addRow("Iterations:", self.iterations_input)
        self.form_layout.addRow("Passes:", self.passes_input)

        # Log output display
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.highlighter = LogHighlighter(self.log_output)
        sys.stdout = QTextEditStream(self.log_output)
        sys.stderr = QTextEditStream(self.log_output)
        self.log_output.setPlainText(
            f"Welcome, {os.getenv('USER', 'Learnosity Support Engineer')}!! Starting Learnosity Data Wrangler on {datetime.now().strftime('%A, %B %m, %Y')} at {datetime.now().strftime('%I:%M %p')}. \n Errors, logs and standard output will be show here... \n Well What Are You Waiting For?!?! Get to Work!"
        )
        # Arrange layout
        self.data_entry_layout.addWidget(self.ticket_file_button)
        self.data_entry_layout.addWidget(self.comments_dir_button)
        self.data_entry_layout.addWidget(self.process_button)
        self.data_entry_layout.addWidget(self.mongo_button)
        self.data_entry_layout.addLayout(self.form_layout)
        self.data_entry_layout.addWidget(self.train_model_button)
        self.data_entry_layout.addWidget(self.previous_runs_button)
        self.data_entry_layout.addWidget(self.assign_topics_button)
        self.data_entry_layout.addStretch()

        # Topic modeling and the decision engine each get a tab; the log pane
        # sits below both so either tab's output stays visible.
        topic_tab = QWidget()
        topic_layout = QVBoxLayout(topic_tab)
        topic_layout.addLayout(self.instruction_layout)
        topic_layout.addLayout(self.data_entry_layout)

        self.triage_tab = TriageTab()
        self.triage_tab.idle.connect(self._finish_pending_close)

        self.tabs = QTabWidget()
        self.tabs.addTab(topic_tab, "Topic Modeling")
        self.tabs.addTab(self.triage_tab, "Decision Engine")

        # Main layout arrangement
        self.main_layout.addWidget(self.tabs, stretch=3)
        self.main_layout.addWidget(self.log_output, stretch=1)

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
        self.pbar = QProgressBar(self)
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
        self.thread = QThread()
        self.training_worker = ThreadWorker(
            lambda: self._train_and_save(int(iterations), int(passes), int(number_of_topics)),
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
        if not os.getenv("MONGO_URI"):
            run["warnings"].append("MONGO_URI is not set: the run was saved locally only and no ticket topics were written.")
            return run
        self.allocator.worker_status.emit("Recording the run and ticket topics in MongoDB...")
        try:
            database = get_database()
            topic_store.record_run(database[topic_store.RUNS_COLLECTION], run)
            run["assignment"] = topic_store.assign_trained_topics(
                database[os.environ.get("MONGO_COLLECTION", "tickets")], run, self.allocator
            )
        except Exception as e:
            logger.exception(f"Could not record run {run['run_id']} in MongoDB: {e}")
            run["warnings"].append(f"Saved locally, but MongoDB could not be updated: {e}")
        return run

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
                logger.info(f"Training finished while closing; saved as run {run['run_id']}.")
            self._finish_pending_close()
            return
        if run is None:
            self.notify_user_of_error((False, self._allocator_error or "Training failed; see the log."))
            return
        message = f"LDA Model training has finished successfully.\n\nSaved as run {run['run_id']}."
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
            message = "MONGO_URI is not set. Set it (see .env.example) and restart the app."
            logger.error(message)
            QMessageBox.warning(self, "MongoDB Not Configured", message)
            return
        try:
            with open(self.wrangler.ticket_file, "r") as tickets_file:
                tickets = json.load(tickets_file)
        except (OSError, json.JSONDecodeError) as e:
            self.notify_user_of_error((False, f"Error: Could not read the ticket file: {e}"))
            return

        self.mongo_ticket_count = len(tickets)
        self.mongo_button.setEnabled(False)
        self.show_progress_bar("Writing Tickets to MongoDB...")

        self.mongo_thread = QThread()
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
        self.update_training_progress_bar(written * 100 // max(self.mongo_ticket_count, 1))

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
            logger.info(f"MongoDB write finished while closing ({len(summaries)} tickets).")
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
        )
        logger.add(
            sink="./wrangle_log.log",
            colorize=True,
            serialize=True,
            format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
            level="DEBUG",
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
            topics=[(t["topic_id"], t["share"], t["words"]) for t in run.get("topics") or []],
            chart_path=chart_path,
            plot_path=plot_path,
            parent=self,
        )
        self.results_dialog.setWindowTitle(f"LDA Results · {topic_store.describe_run(run)}")
        self.results_dialog.show()
        self.results_dialog.open_chart()

    def _runs_collection(self):
        """The topic_model_runs collection, or None to read runs from disk."""
        if not os.getenv("MONGO_URI"):
            return None
        return get_database()[topic_store.RUNS_COLLECTION]

    def _choose_run(self, title: str):
        """Asks the user to pick a previous run; returns it, or None."""
        try:
            runs = topic_store.list_runs(self._runs_collection())
        except Exception as e:
            logger.warning(f"Could not list runs from MongoDB ({e}); reading them from disk.")
            runs = topic_store.list_runs()
        if not runs:
            QMessageBox.information(self, title, "No saved topic-model runs were found.")
            return None
        labels = [topic_store.describe_run(run) for run in runs]
        label, ok = QInputDialog.getItem(self, title, "Run:", labels, 0, False)
        return runs[labels.index(label)] if ok else None

    def show_previous_run(self) -> None:
        """Shows a previous run's topics and coherence."""
        run = self._choose_run("Previous Topic-Model Runs")
        if run is None:
            return
        try:
            self.present_results(run)
        except Exception as e:
            logger.exception(f"Showing run {run['run_id']} failed: {e}")
            self.notify_user_of_error((False, f"Could not show run {run['run_id']}: {e}"))

    def start_assign_topics(self) -> None:
        """Loads a saved run and writes its topics to every ticket in MongoDB."""
        if not os.getenv("MONGO_URI"):
            QMessageBox.warning(self, "MongoDB Not Configured", "MONGO_URI is not set. Set it (see .env.example) and restart the app.")
            return
        run = self._choose_run("Assign Topics from Saved Run")
        if run is None:
            return
        self.assign_topics_button.setEnabled(False)
        self.run_thread = QThread()
        self.run_worker = ThreadWorker(
            lambda: topic_store.assign_from_saved_run(
                get_database()[os.environ.get("MONGO_COLLECTION", "tickets")], run, nlp, stop_words
            ),
            f"Assigning topics from run {run['run_id']}",
        )
        self.run_worker.moveToThread(self.run_thread)
        self.run_worker.finished.connect(lambda summary: self._assign_topics_done(run, summary))
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
            logger.info(f"Topic assignment for run {run['run_id']} finished while closing: {summary}")
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

    # --- shutdown --------------------------------------------------------------

    def _app_threads(self) -> List[Tuple[str, QThread]]:
        """This window's own worker threads, with a label for each."""
        # self.thread shadows QObject.thread() until training first runs.
        candidates = [("model training", getattr(self, "thread", None)),
                      ("MongoDB write", self.mongo_thread),
                      ("topic assignment", self.run_thread)]
        return [(label, t) for label, t in candidates if isinstance(t, QThread)]

    def running_jobs(self) -> List[str]:
        """Labels of the background jobs still running, in this window and the Decision Engine tab."""
        jobs = [label for label, t in self._app_threads() if t.isRunning()]
        if self.triage_tab.is_busy():
            jobs.append("Decision Engine job")
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
        self.setWindowTitle(f"{self._base_title} — closing after {', '.join(jobs)} finishes…")
        if not self._close_pending:
            self._close_pending = True
            logger.warning(f"Close requested; waiting for {', '.join(jobs)} to finish before closing.")

    def _finish_pending_close(self) -> None:
        """Closes the window if a close was deferred and every job is now done."""
        if not self._close_pending:
            return
        jobs = self.running_jobs()
        if jobs:
            self.setWindowTitle(f"{self._base_title} — closing after {', '.join(jobs)} finishes…")
            return
        # Let the finishing job's slot return before the window goes away.
        QTimer.singleShot(0, self.close)

    def join_threads_on_quit(self) -> None:
        """Last resort on application quit: waits for any job still running.

        Normally closeEvent has already waited. This covers quitting by
        another path; it blocks, but never lets a running QThread be destroyed.
        """
        self.triage_tab.wait_for_job()
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
        """
        Initializes the data processing workflow by validating and executing the wrangling and allocation steps.
        This function enables the necessary UI elements and logs the success or failure of the process.

        The `init_start_process` method checks if the wrangler has completed the necessary preprocessing steps
        and if the allocator has preprocessed the data. Upon successful validation, it generates a JSON output,
        enables the training model button, and activates relevant input fields. If any step fails, it logs the error
        and raises a RuntimeError.

        Args:
            self.wrangler: An object responsible for data wrangling operations.
            self.allocator: An object responsible for data allocation and preprocessing.

        Raises:
            RuntimeError: If the data processing fails at any step.
        """

        try:
            # Collect error messages
            error_messages = [
                (
                    not self.wrangler.tickets_reshaped(),
                    "Error: Failed to reshape tickets.",
                ),
                (not self.wrangler.comments_bound(), "Error: Failed to bind comments."),
            ]

            # Check for errors and notify user
            for condition, message in error_messages:
                if condition:
                    self.notify_user_of_error((False, message))
                    return

            # If all checks pass, proceed with JSON generation and UI adjustments
            corpus_task = self.wrangler.create_corpus()
            if not corpus_task:
                self.notify_user_of_error(
                    (False, "Error: Failed to create corpus. The corpus is empty.")
                )
                return
            self.wrangler.generate_json()

            self.allocator = LatentDirichletAllocator(num_of_topics=30)
            # One document per ticket; lemmatization runs in the training thread.
            self.allocator.prelemma_corpus = self.wrangler.ticket_documents()
            self.allocator.document_ids = self.wrangler.ticket_ids()
            self.process_button.setEnabled(False)
            self.train_model_button.setEnabled(True)

            # Enable input fields for training parameters using a loop
            for input_field in [
                self.num_topics_input,
                self.iterations_input,
                self.passes_input,
            ]:
                input_field.setEnabled(True)

            success_message = "The Data Located in the Provided Paths Has been Wrangled 🐄 and Massaged 💆🏽‍♂️...Please select Number of Topics, Iterations and Passes. Then Click Train Model to continue."
            logger.success("Data successfully wrangled and saved.")
            logger.log("USER INPUT REQUIRED", success_message)
            QMessageBox.warning(
                self,
                "Data Has Successfully Processed",
                "The Data Located in the Provided Paths Has been Wrangled and Massaged... \n\nPlease select Number of Topics, Iterations and Passes. \n Then Click Train Model to continue.",
            )
        except Exception as e:
            # Notify user and log unexpected errors
            self.notify_user_of_error((False, f"Unexpected error: {str(e)}"))
            logger.exception(f"Processing failed: {e}")
            raise RuntimeError("Data processing failed") from e


app = QApplication(sys.argv)
window = MainWindow("LRN Support Data Wrangler and LDA Trainer")
app.aboutToQuit.connect(window.join_threads_on_quit)

if __name__ == "__main__":
    window.show()
    sys.exit(app.exec_())
