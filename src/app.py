import sys

from loguru import logger
from PyQt5.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QLabel,
                             QPushButton, QVBoxLayout, QWidget)

from classes import DataWrangler
from utility import DelayedKeyboardInterrupt

# Initialize DataWrangler
DATA = DataWrangler()


def select_ticket_file():
    file_dialog = QFileDialog()
    file_dialog.setNameFilters(["JSON files (*.json)"])
    if file_dialog.exec_():
        file_path = file_dialog.selectedFiles()[0]
        logger.info(f"Ticket File Selected: {file_path}")
        DATA.ticket_file = file_path


def select_comments_dir():
    if dir_path := QFileDialog.getExistingDirectory():
        logger.info(f"Comments Dir Selected: {dir_path}")
        DATA.comments_dir = dir_path


def process_data():
    DATA.process()


app = QApplication(sys.argv)

# Create main window
window = QWidget()
window.setWindowTitle("Data Wrangler")
window.setGeometry(100, 100, 600, 400)

# Instruction frame (equivalent)
instruction_layout = QVBoxLayout()
instruction_label = QLabel(
    "Welcome to the Data Wrangler. \n\n"
    "You will need to identify two file locations.\n"
    "The Tickets File Path represents the path to the ticket payload from the ZenDesk Tickets API.\n"
    "The second is the path to the comments directory, which is the location of the directory that contains the comments for each ticket."
)
warning_label = QLabel(
    "NOTE: The Ticket and the individual comment files must be in JSON Format."
)
warning_label.setStyleSheet("color: red;")

instruction_layout.addWidget(instruction_label)
instruction_layout.addWidget(warning_label)

# Data entry frame (equivalent)
data_entry_layout = QVBoxLayout()

# Buttons
ticket_file_button = QPushButton("Select Path to Ticket File")
ticket_file_button.clicked.connect(select_ticket_file)

comments_dir_button = QPushButton("Select Path to Comments Directory")
comments_dir_button.clicked.connect(select_comments_dir)

process_button = QPushButton("Process Data")
process_button.clicked.connect(process_data)

data_entry_layout.addWidget(ticket_file_button)
data_entry_layout.addWidget(comments_dir_button)
data_entry_layout.addWidget(process_button)

# Main layout
main_layout = QVBoxLayout()
main_layout.addLayout(instruction_layout)
main_layout.addLayout(data_entry_layout)

# Set layout and show window
window.setLayout(main_layout)
window.show()

if __name__ == "__main__":
    logger.info("Initializing Wrangler...")
    sys.exit(app.exec_())
