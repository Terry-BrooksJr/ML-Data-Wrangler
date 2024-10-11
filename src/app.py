import sys
import gradio as gr
import os
import pathlib
import warnings

from loguru import logger
from PyQt5.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QLabel,
                             QPushButton, QVBoxLayout, QWidget)
from wrangler import DataWrangler


from LDA_logic import LatentDirichletAllocator

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


# Initialize DataWrangler
wrangler = DataWrangler()
app = QApplication(sys.argv)
allocator = LatentDirichletAllocator(wrangler.corpus, 30)
def select_ticket_file():
    file_dialog = QFileDialog()
    file_dialog.setNameFilters(["JSON files (*.json)"])
    if file_dialog.exec_():
        file_path = file_dialog.selectedFiles()[0]
        logger.info(f'Ticket File Selected: {file_path}')
        wrangler.ticket_file = file_path

def select_comments_dir():
    if dir_path := QFileDialog.getExistingDirectory():
        logger.info(f'Comments Dir Selected: {dir_path}')
        wrangler.comments_dir = dir_path


 
# Create main window
window = QWidget()
window.setWindowTitle("Data Wrangler")
window.setGeometry(100, 100, 600, 400)

# Instruction frame (equivalent)
instruction_layout = QVBoxLayout()
instruction_label = QLabel("Welcome to the Data Wrangler. \n\n"
                           "You will need to identify two file locations.\n"
                           "The Tickets File Path represents the path to the ticket payload from the ZenDesk Tickets API.\n"
                           "The second is the path to the comments directory, which is the location of the directory that contains the comments for each ticket.")
warning_label = QLabel("NOTE: The Ticket and the individual comment files must be in JSON Format.")
warning_label.setStyleSheet("color: red;")

instruction_layout.addWidget(instruction_label)
instruction_layout.addWidget(warning_label)
 
# Data entry frame (equivalent)
data_entry_layout = QVBoxLayout()


def present_results(allocatorInstance: LatentDirichletAllocator) -> None:
    results_UI = gr.Interface(
                fn=allocatorInstance.visualize_results(),
                inputs= None,
                outputs=["text"])

    with results_UI:
        with gr.Row():
            try:
                top_topics = gr.HTML("""<h1> Top 5 Topics</h1> <br/><ul>""")
                top_five_topics = allocator.get_top_5_topic()
                topics1 = gr.HTML(f"<li>{top_five_topics[0]}</li>")
                topics2 = gr.HTML(f"<li>{top_five_topics[1]}</li>")
                topics3 = gr.HTML(f"<li>{top_five_topics[2]}</li>")
                topics4 = gr.HTML(f"<li>{top_five_topics[3]}</li>")
                topics5 = gr.HTML(f"<li>{top_five_topics[4]}</li>")
                end_list = gr.HTML("</ul>")
            except IndexError:
                top_topics = gr.HTML("""<h1> Top 5 Topics</h1>
                                    <p>Error Loading Top Five</p>
                                    """)

        with gr.Row():
            coherencePlot = gr.LinePlot(x_title='Number of Topics', y_title="Coherence")
        
        with gr.Row():
            try:
                chart_html = os.path.join(pathlib.Path.cwd(),"lda_model.html")
                with open(chart_html, "r") as LDA_Chart_data:
                    LDA_Chart = gr.HTML(f'{LDA_Chart_data.read()}')
            except FileNotFoundError:
                LDA_Chart = gr.HTML("""<h1>Error:Loading LDA Graph</h1>""")

        results_UI.launch()

def init_start_process(wranglerInstance: DataWrangler, allocatorInstance: LatentDirichletAllocator) -> None:
        
        try:
            if (
                wranglerInstance.tickets_reshaped()
                and wranglerInstance.comments_bound()
                and wranglerInstance.create_corpus()
            ):
                if (
                     allocatorInstance.data_preprocessed()
                     and allocatorInstance.model_trained()
                     and allocatorInstance.visualize_results is not None
                ):
                    present_results(allocatorInstance)
                    
                logger.success("Data successfully wrangled and saved")
        except Exception as e:
            logger.exception(f"Processing failed: {e}")
            raise RuntimeError("Data processing failed") from e 
        

#  Buttons
ticket_file_button = QPushButton("Select Path to Ticket File")
ticket_file_button.clicked.connect(select_ticket_file)

comments_dir_button = QPushButton("Select Path to Comments Directory")
comments_dir_button.clicked.connect(select_comments_dir)

process_button = QPushButton("Process Data")
process_button.clicked.connect(init_start_process)


#
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
