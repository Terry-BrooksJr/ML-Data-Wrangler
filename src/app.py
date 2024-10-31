import os
import pathlib
import sys
import warnings
from typing import List, Union, Callable
import en_core_web_lg
import gradio as gr
from utility import WORKER_STATUS
from LDA_logic import LatentDirichletAllocator, stopwords
from wrangler import DataWrangler
from loguru import logger
import sys
from gradio_log import Log

nlp = en_core_web_lg.load()
stop_words: List[str] = stopwords.words("english")

log_file = (os.path.join(pathlib.Path.cwd(), "logs", "output.log")) # <-

fmt = "<green>{time}</> |<bold> {level: <8}</bold> |<white> {message}</white>"
logger.add(log_file, level="DEBUG", colorize=True, diagnose=True, backtrace=True)
logger.add(sys.stdout, level="DEBUG", format=fmt, colorize=True)
logger.level('APPLICATION MESSAGE', no=26)

def log_warning(message, category, filename, lineno, file=None, line=None):
    logger.warning(f" {message}")


warnings.showwarning = log_warning
# Wrangling and Allocator initialization
wrangler = DataWrangler()
wrangler_worker = wrangler.WranglerWorker(on_status=WORKER_STATUS.CREATED)
allocator = LatentDirichletAllocator(wrangler.corpus, 30)
allocate_worker = allocator.LDAModelWorker(on_status=WORKER_STATUS.CREATED)
def select_ticket_file(ticket_file:str) -> str:
    wrangler.ticket_file = ticket_file
    logger.info(f"Ticket File Selected: {ticket_file}")
    return ticket_file

def select_comments_dir(comments_dir:List[str]) -> List[str]:
    wrangler.comments_dir = comments_dir
    logger.info(f"Comments Directory Selected: {comments_dir}")
    return comments_dir

def wrangler_readiness_check() -> List[Union[str|Callable]]:
    if wrangler.is_ready():
        logger.info("Wrangler is ready")
        return process_data(wranglerInstance=wrangler)
    return ValueError('Data Wrangler Instance Is NOT READY to process data. Please ensure that a Ticket JSON File, and a directory of comment JSON files has been selected')
    
def process_data(wranglerInstance: DataWrangler = wrangler) -> List[Union[str|Callable]]:
    """
    Processes the data using the provided DataWrangler instance. 
    This function attempts to run the data wrangling asynchronously and generates a JSON output upon success.

    Args:
        wranglerInstance (DataWrangler, optional): An instance of DataWrangler to process the data. Defaults to the global `wrangler`.

    Returns:
        List[Union[str, Callable]]: A list containing a success message and updates for the interactive components if processing is successful, 
        or an error message and updates indicating failure if an exception occurs.

    Raises:
        Exception: Logs an exception if the data processing fails.
    """
    try:
        if wrangler_worker.run_async(wranglerInstance=wranglerInstance):
            wrangler.generate_json()
            logger.info("Data successfully wrangled and saved.")
            return ["Data successfully processed. You can now train the model.", gr.update(interactive=True), gr.update(interactive=True)]
    except Exception as e:
        logger.exception(f"Processing failed: {e}")
        return f"Data processing failed: {e}", gr.update(interactive=False), gr.update(interactive=False)

def certify_corpus():
    try:
        wrangler.generate_corpus_json()
        logger.success("Corpus certified successfully.")
        return gr.update(visible=True)
    except Exception as e:
        logger.error(f"Corpus certification failed: {e}")
        return gr.update(visible=False)

def train_model(num_topics, iterations, passes):
    if not (num_topics.isdigit() and iterations.isdigit() and passes.isdigit()):
        return "All inputs must be integers."

    if int(passes) >= 20 or int(iterations) >= 200:
        return "Passes should be < 20 and iterations < 200."

    if allocator.data_preprocessed():
        if not allocator.model_trained(
            iterations=int(iterations),
            workers=4,
            passes=int(passes),
            num_of_topics=int(num_topics),
        ):
            return "Model training failed."
        logger.success("Model successfully trained!")
        return present_results()

def present_results():
    if top_topics := allocator.get_top_5_topic():
        return f"Top 5 Topics: {top_topics}"
    else:
        return "Error loading top five topics."


with gr.Blocks() as demo:
    with gr.Row():
        gr.Markdown("## Zendesk Ticket Data Wrangler and LDA Processor")
        gr.Markdown("""
            ### Instructions:
            1. Select the ticket file from the ZenDesk Tickets API.
            2. Select the comments directory with JSON comments for each ticket.
            3. Configure the number of topics, iterations, and passes.
            4. Train the LDA model.
        """)

    with gr.Tab("Data Preparation 🤼‍♀️"):
        with gr.Row():
            ticket_file = gr.FileExplorer(glob="*/**.json", file_count="single", label="Select Path to Ticket File")
            comments_dir = gr.FileExplorer(glob="*/**.json", label="Select Path to Comments Directory", interactive=True)

        comments_dir.change(fn=select_comments_dir, inputs=comments_dir)
        ticket_file.change(fn=select_ticket_file, inputs=ticket_file)

        with gr.Row():  
            process_output = gr.Textbox(interactive=False, lines=15)
            Log(log_file, dark=True, label="Training Output", show_label=True)
        
        with gr.Row():
            process_button = gr.Button("Prepare Data For Training", interactive=True)
            certify_corpus_button = gr.Button("Certify Corpus 🗂️", interactive=False)
        
        with gr.Row():
            download_button = gr.File(label="Download your file here", visible=False)

            process_button.click(fn=wrangler_readiness_check, inputs=None, outputs=[process_output, certify_corpus_button, download_button])

        certify_corpus_button.click(fn=certify_corpus, inputs=None, outputs=download_button)

    with gr.Tab("Model Training 🏋️‍♀️"):
        with gr.Row():
            num_topics_input = gr.Number(label="Number of Topics")
            iterations_input = gr.Number(label="📊Iterations")
            workers_input = gr.Number(label="CPU Workers", interactive=False, value=2, info="This is not configurable")
            passes_input = gr.Number(label="Passes")
        
        with gr.Row():
            train_button = gr.Button("Train Model", interactive=False)

        with gr.Row():
            train_output = gr.Textbox(interactive=False, lines=15)

        train_button.click(fn=train_model, inputs=[num_topics_input, iterations_input, passes_input], outputs=train_output)

    demo.load(fn=lambda: "Loading logs...", inputs=None, outputs=process_output)

if __name__ == "__main__":
    demo.queue().launch()