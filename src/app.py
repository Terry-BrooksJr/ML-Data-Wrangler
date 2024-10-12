import os
import pathlib
import sys
import warnings
from typing import List

import gradio as gr
from gradio import HTML, Interface, LinePlot, Row
from loguru import logger
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from LDA_logic import LatentDirichletAllocator
from utility import QTextEditStream
from wrangler import DataWrangler

# SECTION - Instantiating Classes & Initializing Variables

# SECTION - Start Program Classes & Variables
wrangler: DataWrangler = DataWrangler()
app: QApplication = QApplication(sys.argv)
allocator: LatentDirichletAllocator = LatentDirichletAllocator(wrangler.corpus, 30)
# !SECTION - End Program Classes & Variables

# SECTION - Start GUI Classes & Variables
# Layout
window: QWidget = QWidget()
instruction_layout: QVBoxLayout = QVBoxLayout()
data_entry_layout: QVBoxLayout = QVBoxLayout()
# GUI Elements and Components
ticket_file_button = QPushButton("Select Path to Ticket File")
comments_dir_button = QPushButton("Select Path to Comments Directory")
process_button = QPushButton("Process Data")
train_model_button = QPushButton("Train Model")

# GUI Live Logging
log_output = QTextEdit()
sys.stdout = QTextEditStream(log_output)
sys.stderr = QTextEditStream(log_output)
# qtextedit_logger = QTextEditLogger(log_output)
# !SECTION - End GUI Classes & Variables
# !SECTION - Instantiating Classes & Initializing Variables


# SECTION - Start Functions
def select_ticket_file() -> None:
    """Opens a file dialog to select a JSON ticket file.

    This function displays a file dialog that allows the user to select a
    JSON file. Upon selection, it logs the file path and updates the
    ticket file in the wrangler.

    Args:
        None

    Returns:
        None
    """
    file_dialog: QFileDialog = QFileDialog()
    file_dialog.setNameFilters(["JSON files (*.json)"])
    if file_dialog.exec_():
        file_path: str = file_dialog.selectedFiles()[0]
        logger.info(f"Ticket File Selected: {file_path}")
        wrangler.ticket_file = file_path


def select_comments_dir() -> None:
    """Opens a dialog to select a directory for comments.

    This function displays a directory selection dialog that allows the user
    to choose a directory. Upon selection, it logs the directory path and
    updates the comments directory in the wrangler.

    Args:
        None

    Returns:
        None
    """
    if dir_path := QFileDialog.getExistingDirectory():
        logger.info(f"Comments Dir Selected: {dir_path}")
        wrangler.comments_dir = dir_path


def present_results(allocatorInstance: LatentDirichletAllocator) -> None:
    """Displays the results of the Latent Dirichlet Allocation (LDA) model.

    This function creates a user interface to present the top five topics
    identified by the LDA model and visualizes the coherence plot. It handles
    potential errors in loading the topics and the LDA graph, providing
    appropriate feedback in the gradio UI.

    Args:
        allocatorInstance: An instance of LatentDirichletAllocator that contains
            the results and methods for visualizing the LDA model.

    Returns:
        None
    """
    results_UI: Interface = Interface(
        fn=allocatorInstance.visualize_results(), inputs=None, outputs=["text"]
    )

    with results_UI:
        with Row():
            try:
                top_topics = HTML("""<h1> Top 5 Topics</h1> <br/><ul>""")
                top_five_topics = allocator.get_top_5_topic()
                topics1 = HTML(f"<li>{top_five_topics[0]}</li>")
                topics2 = HTML(f"<li>{top_five_topics[1]}</li>")
                topics3 = HTML(f"<li>{top_five_topics[2]}</li>")
                topics4 = HTML(f"<li>{top_five_topics[3]}</li>")
                topics5 = HTML(f"<li>{top_five_topics[4]}</li>")
                end_list = HTML("</ul>")
            except IndexError:
                top_topics = HTML(
                    """<h1> Top 5 Topics</h1>
                                    <p>Error Loading Top Five</p>
                                    """
                )

        with Row():
            coherencePlot = LinePlot(x_title="Number of Topics", y_title="Coherence")

        with Row():
            try:
                chart_html = os.path.join(pathlib.Path.cwd(), "lda_model.html")
                with open(chart_html, "r") as LDA_Chart_data:
                    LDA_Chart = HTML(f"{LDA_Chart_data.read()}")
            except FileNotFoundError:
                LDA_Chart = HTML("""<h1>Error: Loading LDA Graph </h1>""")

        results_UI.launch()


def toggle_GUI_button_state(
    train_button_instance: QPushButton, process_button_instance: QPushButton
) -> None:
    """Toggles the state of GUI buttons for data processing.

    This function disables the process button and enables the train button,
    indicating that the preprocessing steps have been completed. It also logs
    a warning message to prompt the user to continue with the training process.

    Args:
        train_button_instance: The QPushButton instance for the training button.
        process_button_instance: The QPushButton instance for the processing button.

    Returns:
        None
    """
    process_button_instance.setEnabled(False)
    train_button_instance.setEnabled(True)
    logger.opt(colors=True).warning(
        "<bold>All Required Steps Needed to Preprocess the Data Have Successfully Completed.</bold>"
    )
    logger.opt(capture=False).info(
        "<bold> <red>Please Click the 'Train Model' button to continue...</red></bold>"
    )


def init_start_process(
    wranglerInstance: DataWrangler, allocatorInstance: LatentDirichletAllocator
) -> None:
    """Initializes the data wrangling and processing workflow.

    This function orchestrates the data wrangling and processing by checking
    the necessary conditions on the provided wrangler and allocator instances.
    If all conditions are met, it processes the data and presents the results;
    otherwise, it logs an error and raises an exception.

    Args:
        wranglerInstance: An instance of DataWrangler responsible for data
            manipulation and preparation.
        allocatorInstance: An instance of LatentDirichletAllocator that handles
            data allocation and model training.

    Returns:
        None

    Raises:
        RuntimeError: If data processing fails due to any exception.
    """
    try:
        if (
            wranglerInstance.tickets_reshaped()
            and wranglerInstance.comments_bound()
            and wranglerInstance.create_corpus()
        ):
            if allocatorInstance.data_preprocessed():
                wrangler.generate_json()
                toggle_GUI_button_state(train_model_button, process_button)
            logger.success("Data successfully wrangled and saved")
    except Exception as e:
        logger.exception(f"Processing failed: {e}")
        raise RuntimeError("Data processing failed") from e


def init_wrapper():
    """Initializes the start process with the specified instances.

    This function serves as a wrapper to initiate the start process using
    the provided wrangler and allocator instances. It simplifies the
    initialization by directly passing the instances to the underlying
    initialization function.


    Args:
        None

    Returns:
        The result of the initialization process.
    """
    return init_start_process(wranglerInstance=wrangler, allocatorInstance=allocator)
 

def train_model(
    wranglerInstance: DataWrangler, allocatorInstance: LatentDirichletAllocator
) -> None:
    """Trains the LDA model using the provided wrangler and allocator instances.

    This function initiates the model training process and logs the progress.
    Upon successful training, it prepares the data for reporting and visualization,
    and presents the results if available.

    Args:
        wranglerInstance: An instance of DataWrangler used for data manipulation.
        allocatorInstance: An instance of LatentDirichletAllocator responsible for
            model training and visualization.

    Returns:
        None
    """
    logger.opt(colors=True).info("Initiating Model Training...")
    if allocatorInstance.model_trained(iterations=10, workers=4, passes=10, num_of_topics=30):
        logger.success(
            "Model successfully trained! Preparing Data for Reporting and Visualization..."
        )
        if allocatorInstance.visualize_results is not None:
            present_results(allocatorInstance)

def train_model_wrapper():
    return train_model(allocatorInstance=allocator, wranglerInstance=wrangler)
# !SECTION - End Functions

# SECTION - Start Program GUI Configuration

# Create main program GUI window
window.setWindowTitle("Data Wrangler")
window.setGeometry(100, 100, 600, 400)

# Instruction frame (equivalent) - Program GUI
instruction_label: QLabel = QLabel(
    "Welcome to the Data Wrangler. \n\n"
    "You will need to identify two file locations.\n"
    "The Tickets File Path represents the path to the ticket payload from the ZenDesk Tickets API.\n"
    "The second is the path to the comments directory, which is the location of the directory that contains the comments for each ticket."
)
warning_label: QLabel = QLabel(
    "NOTE: The Ticket and the individual comment files must be in JSON Format."
)
warning_label.setStyleSheet("color: red;")
instruction_layout.addWidget(instruction_label)
instruction_layout.addWidget(warning_label)

# Program GUI Interactive Components Configuration
ticket_file_button.clicked.connect(select_ticket_file)
comments_dir_button.clicked.connect(select_comments_dir)
process_button.clicked.connect(init_wrapper)
process_button.setEnabled(True)
train_model_button.clicked.connect(train_model_wrapper)
train_model_button.setEnabled(False)
data_entry_layout.addWidget(ticket_file_button)
data_entry_layout.addWidget(comments_dir_button)
data_entry_layout.addWidget(process_button)
data_entry_layout.addWidget(train_model_button)

# Log Output
log_output.setReadOnly(True)
data_entry_layout.addWidget(log_output)

# Main layout
main_layout = QVBoxLayout()
main_layout.addLayout(instruction_layout)
main_layout.addLayout(data_entry_layout)

# Set layout and show window
window.setLayout(main_layout)
window.show()
# !SECTION - End Program GUI Configuration

# SECTION -  Start Logging

logger.add(
    sys.stdout,
    colorize=True,
    format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
    level="INFO",
)
# !SECTION - End Logging

if __name__ == "__main__":
    logger.info("Initializing Wrangler...")
    sys.exit(app.exec_())
