import os
import pathlib
import sys
import time
from datetime import datetime
from typing import List, Tuple, Union
from uuid import uuid4
import en_core_web_lg
import gradio as gr
from gradio import HTML, Interface, LinePlot, Row
from loguru import logger
from utility import WORKER_STATUS

from LDA_logic import LatentDirichletAllocator, stopwords
from utility import LogHighlighter, QTextEditStream
from wrangler import DataWrangler

nlp = en_core_web_lg.load()
stop_words: List[str] = stopwords.words("english")


import os
import pathlib
import sys
import warnings
from typing import List, Tuple

import gradio as gr
from utility import QTextEditStream


# Wrangling and Allocator initialization
wrangler = DataWrangler()
wrangle_worker = wrangler.WranglerWorker(on_status=WORKER_STATUS.CREATED)

allocator = LatentDirichletAllocator(wrangler.corpus, 30)
allocate_worker = allocator.LDAModelWorker(on_status=WORKER_STATUS.CREATED)

# Function to select ticket file
def select_ticket_file(ticket_file:str) -> str:
    wrangler.ticket_file = ticket_file
    logger.info(f"Ticket File Selected: {ticket_file}")
    return ticket_file


# Function to select comments directory
def select_comments_dir(comments_dir:list[str]) -> List[str]:
    wrangler.comments_dir = comments_dir
    logger.info(f"Comments Directory Selected: {comments_dir}")
    return comments_dir


# Function to process the data
def process_data():
    try:
        if wrangler.tickets_reshaped() and wrangler.comments_bound() and wrangler.create_corpus() :
            wrangler.generate_json()
            logger.info("Data successfully wrangled and saved.")
            return "Data successfully processed. You can now train the model.", gr.update(interactive=True)
    except Exception as e:
        logger.exception(f"Processing failed: {e}")
        return f"Data processing failed: {e}", gr.update(interactive=False)


# Function to train the model
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


# Function to present the results
def present_results():
    top_topics = allocator.get_top_5_topic()
    if top_topics:
        return f"Top 5 Topics: {top_topics}"
    else:
        return "Error loading top five topics."



theme = gr.themes.Ocean(
    text_size="sm",
    spacing_size="sm",
)

with gr.Blocks(theme=theme) as demo:
    with gr.Row():
        gr.Markdown("## Zendesk Ticket Data Wrangler and LDA Processor")

        gr.Markdown("""
            ### Instructions:
            1. Select the ticket file from the ZenDesk Tickets API.
            2. Select the comments directory with JSON comments for each ticket.
            3. Configure the number of topics, iterations, and passes.
            4. Train the LDA model.
        """)
    with gr.Tab("Data Perpetration"):
        with gr.Row():
            ticket_file = gr.FileExplorer(glob="*/**.json", file_count="single", label="Select Path to Ticket File")
            comments_dir = gr.FileExplorer(glob="*/**.json", label="Select Path to Comments Directory",  interactive=True, )
    
        comments_dir.change(fn=select_comments_dir, inputs=comments_dir)
        
        ticket_file.change(fn=select_ticket_file, inputs=ticket_file, api_name="ticket_file")
       
        with gr.Row():  
            with gr.Column():
                gr.Markdown("### Data Cleaning and Wrangling Output")
                process_output = gr.Textbox( interactive=False, placeholder=f'Hello, {os.getenv("USER", "Learnosity Support Engineer")}! Status updates, overall progress and errors  for the Data preparation stage will show here.', lines=15, container=True)
        with gr.Row():
            process_button = gr.Button("Prepare Data For Training", interactive=True)
            
            process_button.click(fn=lambda:wrangle_worker.run_async(wrangler), inputs=None, outputs=[process_output])
    

    
    with gr.Tab("Model Training"):
        # Form inputs for model training
        with gr.Row():
            num_topics_input = gr.Number(label="Number of Topics")
            iterations_input = gr.Number(label="Iterations")
            workers_input = gr.Number(label="CPU Workers", interactive=False, value=2, info="This is not configurable")
            passes_input = gr.Number(label="Passes")
        with gr.Row():
            train_button = gr.Button("Train Model",  interactive=False, )
        with gr.Row():  
            with gr.Column():
                gr.Markdown("### Model Training Output")
                train_output = gr.Textbox(interactive=False, placeholder=f'Hello, {os.getenv("USER", "Learnosity Support Engineer")}! Status updates, overall progress and errors  related to the training of the model and results.', lines=15, container=True)
    


    train_button.click(
            fn=train_model,
            inputs=[num_topics_input, iterations_input, passes_input],
            outputs=train_output
        )


if __name__ == "__main__":
    demo.launch()

