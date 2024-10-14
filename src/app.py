import os
import pathlib
import sys
from typing import Tuple, Union

import gradio as gr
from gradio import HTML, Interface, LinePlot, Row
from loguru import logger
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QFormLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from LDA_logic import LatentDirichletAllocator
from utility import LogHighlighter, QTextEditStream
from wrangler import DataWrangler

class LDAApplication(QApplication):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.wrangler = DataWrangler()
        self.allocator = LatentDirichletAllocator(self.wrangler.corpus, 30)

        self.init_logging()
        
        # Log output display
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.highlighter = LogHighlighter(self.log_output)
        sys.stdout = QTextEditStream(self.log_output)
        sys.stderr = QTextEditStream(self.log_output)

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
    
    def init_start_process(self) -> None:
        """
        Initializes the data processing workflow by validating and executing the wrangling and allocation steps.
        This function enables the necessary UI elements and logs the success or failure of the process.

        The `init_start_process` method checks if the wrangler has completed the necessary preprocessing steps
        and if the allocator has preprocessed the data. Upon successful validation, it generates a JSON output,
        enables the training model button, and activates relevant input fields. If any step fails, it logs the error
        and raises a RuntimeError.

        Args:
   
        Raises:
            RuntimeError: If the data processing fails at any step.
        """
        try:
            # Collect error messages
            error_messages = [
                (not self.wrangler.tickets_reshaped(), "Error: Failed to reshape tickets."),
                (not self.wrangler.comments_bound(), "Error: Failed to bind comments."),
                (not self.wrangler.create_corpus(), "Error: Failed to create corpus."),
                (not self.allocator.data_preprocessed(), "Error: Data preprocessing in allocator failed.")
            ]

            # Check for errors and notify user
            for condition, message in error_messages:
                if condition:
                    self.notify_user_of_error((False, message))
                    return

            # If all checks pass, proceed with JSON generation and UI adjustments
            self.wrangler.generate_json()
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
                success_message,
            )
        except Exception as e:
            # Notify user and log unexpected errors
            self.notify_user_of_error((False, f"Unexpected error: {str(e)}"))
            logger.exception(f"Processing failed: {e}")
            raise RuntimeError("Data processing failed") from e


    def validate_inputs(
            self, number_of_topics, iterations, passes
        ) -> Tuple[bool, Union[str, None]]:
        """
                Validates the user inputs for model training parameters.
                This method checks that the inputs are integers and within acceptable ranges.

                The `validate_inputs` function ensures that the provided values for the number of topics, iterations,
                and passes meet the specified criteria. It returns a boolean indicating the validity of the inputs
                along with an error message if the inputs are invalid.

                Args:
                    number_of_topics (str): The number of topics to be used in the model.
                    iterations (str): The number of iterations for the training process.
                    passes (str): The number of passes for the training process.

                Returns:
                    tuple: A tuple containing a boolean indicating validity and an error message.
        """
        if not all(map(str.isdigit, [number_of_topics, iterations, passes])):
            return False, "All inputs must be integers."
        if int(passes) >= 20 or int(iterations) >= 200:
            return False, "Passes should be < 20 and iterations < 200."
        return True, ""

    def train_model(self) -> None:
        """
        Trains the model using the specified parameters for topics, iterations, and passes.
        This method validates the input values and initiates the training process if the inputs are valid.

        The `train_model` function retrieves user input for the number of topics, iterations, and passes,
        validates these inputs, and then calls the model training method on the allocator instance.
        If the training is successful, it logs a success message and presents the results.

        Args:

            Returns:
                None

        Raises:
            QMessageBox: Displays a warning if the input validation fails.
        """
        number_of_topics = self.num_topics_input.text()
        iterations = self.iterations_input.text()
        passes = self.passes_input.text()

        valid, error_message = self.validate_inputs(
            number_of_topics, iterations, passes
        )
        if not valid:
            QMessageBox.warning(self, "Input Validation Error", error_message)
            return

        if self.allocator.model_trained(
            iterations=int(iterations),
            workers=4,
            passes=int(passes),
            num_of_topics=int(number_of_topics),
        ):
            logger.success("Model successfully trained!")
            self.present_results()
            
    def present_results(self) -> None:
        """
        Displays the results of the model training in a user interface.
        This method visualizes the top topics and coherence plot, and attempts to load an LDA graph.

        The `present_results` function creates an interface to show the results of the topic modeling.
        It retrieves the top five topics from the allocator instance and displays them, along with a coherence plot.
        If any errors occur during the retrieval of topics or loading of the graph, appropriate error messages are shown.

        Args:

        Returns:
            None

        Raises:
            IndexError: If there are not enough topics to display.
            FileNotFoundError: If the LDA graph file cannot be found.
        """
        results_UI = Interface(
            fn=self.allocator.visualize_results(), inputs=None, outputs=["text"]
        )
        with results_UI:
            with Row():
                try:
                    top_topics = HTML("""<h1> Top 5 Topics</h1> <br/><ul>""")
                    top_five_topics = self.QApp.allocator.get_top_5_topic()
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
                coherencePlot = LinePlot(
                    x_title="Number of Topics", y_title="Coherence"
                )

            with Row():
                try:
                    chart_html = os.path.join(pathlib.Path.cwd(), "lda_model.html")
                    with open(chart_html, "r") as LDA_Chart_data:
                        LDA_Chart = HTML(f"{LDA_Chart_data.read()}")
                except FileNotFoundError:
                    LDA_Chart = HTML("""<h1>Error: Loading LDA Graph </h1>""")

            results_UI.launch()

        
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
        return QMessageBox.critical(
            self, title="Critical Error", text=error[1])
            

class MainWindow(QMainWindow):
    """
    Represents the main application window for the Data Wrangler tool.
    This class initializes the user interface and manages interactions for data processing and model training.

    The `MainWindow` class sets up the main application window, including layouts, buttons, and logging.
    It provides methods for selecting files, processing data, and training models, while ensuring user inputs are validated.

    Args:
        windowName (str): The title of the main application window.

    Attributes:
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

    def __init__(self, QApp: LDAApplication, windowName: str) -> None:
        """
        Initializes the main application window for the Data Wrangler tool.
        This constructor sets up the user interface, including layouts and components, and initializes necessary data handling objects.

        The `__init__` method configures the main window's title and size, 
        and initializes the user interface components and logging. It establishes the layout for the application, ensuring
        that all elements are properly arranged for user interaction.

        Args:
            windowName (str): The title to be displayed in the main application window.

        Returns:
            None
        """
        super().__init__()

        self.QApp = QApp
   
        # Setting up the main application window
        self.setWindowTitle(windowName)
        self.setGeometry(100, 100, 600, 400)

        # Main layout
        self.main_layout = QVBoxLayout()
        self.instruction_layout = QVBoxLayout()
        self.data_entry_layout = QVBoxLayout()

        # GUI Elements
        self.init_ui_components()

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
            "2. The path to the comments directory with JSON comments for each ticket."
        )
        warning_label = QLabel("NOTE: Ticket and comment files must be in JSON format.")
        warning_label.setStyleSheet("color: red;")

        self.instruction_layout.addWidget(instruction_label)
        self.instruction_layout.addWidget(warning_label)

        # Buttons
        self.ticket_file_button = QPushButton("Select Path to Ticket File")
        self.comments_dir_button = QPushButton("Select Path to Comments Directory")
        self.process_button = QPushButton("Process Data")
        self.train_model_button = QPushButton("Train Model")

        self.ticket_file_button.clicked.connect(self.QApp.select_ticket_file)
        self.comments_dir_button.clicked.connect(self.QApp.select_comments_dir)
        self.process_button.clicked.connect(self.QApp.init_start_process)
        self.train_model_button.clicked.connect(self.QApp.train_model)
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


        # Arrange layout
        self.data_entry_layout.addWidget(self.ticket_file_button)
        self.data_entry_layout.addWidget(self.comments_dir_button)
        self.data_entry_layout.addWidget(self.process_button)
        self.data_entry_layout.addLayout(self.form_layout)
        self.data_entry_layout.addWidget(self.train_model_button)
        self.data_entry_layout.addWidget(self.log_output)

        # Main layout arrangement
        self.main_layout.addLayout(self.instruction_layout)
        self.main_layout.addLayout(self.data_entry_layout)




app = LDAApplication(sys.argv)

window = MainWindow("LRN Support Data Wrangler and LDA Trainer")


if __name__ == "__main__":
    window.show()
    sys.exit(app.exec_())
