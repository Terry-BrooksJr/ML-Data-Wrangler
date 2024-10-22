import json
import os
import re
import pathlib
import threading
from concurrent.futures import ThreadPoolExecutor
import unicodedata
from datetime import datetime
from enum import Enum
from html import unescape
from typing import List, TextIO, Tuple, Union, Callable
from utility import WORKER_STATUS
from loguru import logger
import validators
from PyQt5.QtCore import pyqtSignal, QMutex, QObject
from tqdm_loggable.auto import tqdm
import gradio as gr


class MyEncoder(json.JSONEncoder):
    """Custom JSON encoder for serializing specific object types.

    This encoder extends the default JSONEncoder to handle serialization of
    custom objects such as TicketStatus, datetime, and Comment. It provides
    a way to convert these objects into a JSON-compatible format.
    """

    def default(self, obj):
        """
        Converts custom objects to a JSON-compatible format.

        This method checks the ticket_type of the object and returns a corresponding
        JSON representation. If the object ticket_type is not recognized, it falls
        back to the default serialization method.

        Args:
            obj: The object to be serialized.

        Returns:
            A JSON-compatible representation of the object.

        Raises:
            TypeError: If the object ticket_type is not serializable.
        """
        if isinstance(obj, TicketStatus):
            return {"status": obj.name}
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, Comment):
            return obj.to_dict_format()
        return json.JSONEncoder.default(self, obj)


class TicketStatus(Enum):
    """
    Enumeration representing the various statuses of a ticket.

    This class defines the possible states a ticket can be in during its
    lifecycle, including OPEN, HOLD, PENDING, SOLVED, and CLOSED. Each
    status is associated with a unique integer value for easy identification.
    """

    OPEN = 1
    HOLD = 2
    PENDING = 3
    SOLVED = 4
    CLOSED = 5


class Comment:
    """
    Represents a comment with an ID, creation timestamp, and body text.

    This class is used to encapsulate the details of a comment, including
    its unique identifier, the date and time it was created, and the content
    of the comment. It provides a method to convert the comment into a
    dictionary format for easier serialization or data manipulation.
    """

    def __init__(self, id: int, created_at: datetime, body: str) -> None:
        """
        Initializes a Comment instance with an ID, creation time, and body.

        Args:
            id: A unique identifier for the comment.
            created_at: The date and time when the comment was created.
            body: The text content of the comment.
        """
        self.id = id
        self.created_at = created_at
        self.body = body

    def __getitem__(self, key):
        return key

    def __setitem__(self, key, value):
        self.detail[key] = value

    def to_dict_format(self) -> dict:
        """
        Converts the Comment instance to a dictionary format.

        This method returns a dictionary representation of the comment,
        including its ID, creation timestamp, and body text.

        Returns:
            A dictionary containing the comment's details.
        """

        return {
            "created_at": self.created_at,
            "id": self.id,
            "body": self.body,
        }


class Ticket:
    """
    Represents a support ticket with various attributes and associated comments.

    This class encapsulates the details of a support ticket, including its
    unique identifier, creation and last updated timestamps, status, subject,
    tags, outcome, and ticket_type. It also maintains a list of comments related to
    the ticket, allowing for comprehensive tracking of the ticket's progress.
    """

    def __init__(
        self,
        id: int,
        created_at: datetime,
        status: TicketStatus,
        last_updated: datetime,
        subject: str,
        tags: List[str] = None,
        outcome: str = None,
        ticket_type: str = None,
    ) -> None:
        """
        Initializes a Ticket instance with the specified attributes.

        Args:
            id: A unique identifier for the ticket.
            created_at: The date and time when the ticket was created.
            status: The current status of the ticket, represented by a TicketStatus.
            last_updated: The date and time when the ticket was last updated.
            subject: The subject or title of the ticket.
            tags: Optional list of tags associated with the ticket.
            outcome: Optional outcome of the ticket resolution.
            ticket_type: Optional ticket_type of the ticket.

        """
        self.id = id
        self.created_at = created_at
        self.last_updated = last_updated
        self.status = status
        self.subject = subject
        self.tags = tags or []
        self.outcome: str = outcome
        self.ticket_type: str = ticket_type
        self.comments: List[Comment] = []

    def __str__(self):
        return f"Ticket {self.id} ({self.status.name})"

    def __getitem__(self, key):
        return key


class DataWrangler:
    """A class for processing and managing ticket data and associated comments.

    This class provides methods to reshape ticket data from JSON files, bind
    comments to tickets, create a text corpus from the tickets and comments,
    clean text data, and generate JSON files containing processed ticket data.
    It facilitates the organization and analysis of ticket-related information.
    """

    def __init__(
            self
        ) -> None:
            """Initializes the DataWrangler with specified directories for comments and tickets.

            Args:
                comments_dir: The directory where comment files are stored.
                ticket_file: The path to the JSON file containing ticket data.
            """
            self.ticket_file = None
            self.comments_dir = None
            self.wrangled_tickets: List[Ticket] = []
            self.corpus: str = ""

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, name: str, value) -> None:
        return setattr(self, name, value)

    @staticmethod
    def reshaped_comment(comment) -> Comment:
        """Reshapes a comment dictionary into a Comment object.

        This static method takes a dictionary representation of a comment and
        converts it into a Comment object, extracting the relevant fields.
        It raises an exception if the reshaping process encounters any issues.

        Args:
            comment: A dictionary containing the comment data, including
                    'id', 'created_at', and 'plain_body'.

        Returns:
            A Comment object populated with the provided data.

        Raises:
            RuntimeError: If there is an error during the reshaping process.
        """
        
        try:
            return Comment(
                id=comment["id"],
                created_at=comment["created_at"],
                body=comment["plain_body"],
            )
        except Exception as e:
            logger.exception(f"Failed to reshape comment: {e}")
            raise RuntimeError("Comment reshaping failed") from e

    def generate_processed_tickets_json(
        self,
        filename: str = f"processed_tickets{datetime.now().strftime('%Y-%m-%d')}.json",
    ) -> str:
        """
        Generates JSON files for processed tickets and the associated corpus.

        This function creates two JSON files: one containing the processed tickets and another containing the corpus data. The filenames are constructed based on the current date, and the function returns file handles for both output files.

        Args:
            filename (str, optional): The name of the output file for processed tickets. Defaults to "processed_tickets" followed by the current date.

        Returns:
            str: A tuple containing the paths to the processed tickets and the corpus JSON files.

        Raises:
            IOError: If there is an issue opening or writing to the output files.
        """

        with open(filename, "w+") as output1:
            json.dump(
                [ticket.__dict__ for ticket in self.wrangled_tickets],
                output1,
                indent=4,
                cls=MyEncoder,
            )
            logger.success(f"JSON output successfully generated for Processed Tickets located at {filename}")
            return output1
    
    def generate_corpus_json(
        self,ui_download_element, 
        filename: str = f"corpus_{datetime.now().strftime('%Y-%m-%d')}.json", 
    ) -> str:
        """
        Generates JSON files for corpus.

        This function creates one JSON files containing the corpus data. The filenames are constructed based on the current date, and the function returns path for both files.

        Args:
            filename (str, optional): The name of the output file for processed tickets. Defaults to "corpus" followed by the current date.

        Returns:
            Tuple[TextIO, TextIO]: A tuple containing the paths to the processed tickets and the corpus JSON files.

        Raises:
            IOError: If there is an issue opening or writing to the output files.
        """

        with open(filename, "w+") as output2:
            json.dump(self.corpus, output2, indent=4,cls=MyEncoder)
            logger.success(f"JSON output successfully generated for Corpus located at {filename}")
            ui_download_element.visible = True
            return  output2.read()
    class WranglerWorker:
        """
        Represents a worker class for async processing tickets and their associated comments.
        """
        
        def __init__(self,  on_status: Callable[[str], None]):
            self.lock = threading.Lock()
            self.on_status = on_status  # A callback to report status or progress
            self.executor = ThreadPoolExecutor(max_workers=2)
        def on_status(self, status:str, is_failure:bool=False, task_complete:bool=False) -> str:
            if not is_failure:
                logger.info(str)
                self.on_status = WORKER_STATUS.BUSY
                return f"{self.on_status} - {status}"

            if task_complete:
                logger.success(str)
                self.on_status = WORKER_STATUS.FINISHED
                return f"{self.on_status} - {status}"
            
            logger.error(str)
            self.on_status = WORKER_STATUS.FAILED
            return f"{self.on_status} - {status}"
        def _cleanse(self, body_of_text: str) -> List[str]:
            """Cleanses the provided text by normalizing and unescaping each line."""
            # check_body = [check_line for check_line in body_of_text.splitlines() if not re.search(r"^\"\"$",check_line) ]
            # cleaned_lines = [
            check_body = unicodedata.normalize("NFKC", unescape(body_of_text)).replace("\n", " ", -1).replace("\r", " ", -1).replace("  ", " ").split()
            #     for line  in check_body if line != "\'\'"
            # ]
            scrubbed = [
                word for word in check_body
                if not any(
                    [
                        validators.email(word),
                        validators.url(word),
                        validators.uuid(word),
                        validators.hashes.md5(word),
                        validators.ip_address.ipv4(word),
                    ]
                )
            ]
            logger.debug("Successfully Cleansed Comment Data For Corpus")

            return " ".join(scrubbed)

        def _bind_comments(self, ticket: Ticket,wranglerInstance, progress=gr.Progress(track_tqdm=True)):

            """Helper method to bind comments to a single ticket."""
            comments_found = False
            no_comments_processed = 0 
            for filename in tqdm(wranglerInstance.comments_dir):
                file_ticket_no = filename.split("/")[-1]
                file_ticket_no = int(file_ticket_no.split(".")[0])
                if file_ticket_no == ticket.id:       

                    with open(filename, "r") as comments_file:
                        comments_data = json.load(comments_file)
                        for key, value in comments_data.items():
                            for comment in tqdm(value):
                                logger.debug(f"Binding {len(value)} comments for ticket {ticket.id}")
                                reshaped_comment = self._reshaped_comment(comment)
                                ticket.comments.append(reshaped_comment)
                                comments_found = True
                                no_comments_processed += 1
            if not comments_found:
                logger.warning(f"No comments found for ticket {ticket.id}")
            self.on_status = WORKER_STATUS.FINISHED
            return (ticket, no_comments_processed)

        def comments_bound(self, wranglerInstance, progress=gr.Progress(track_tqdm=True)) -> bool:
            """Binds comments to the wrangled tickets asynchronously."""
            try:
                total_comments_bound = 0
                for ticket in tqdm(wranglerInstance.wrangled_tickets):
                    future = self.executor.submit(lambda: self._bind_comments(ticket=ticket, wranglerInstance=wranglerInstance))
                    total_comments_bound += future.result()[1]
                    logger.debug(f"Finished binding comments for ticket {future.result()[0]['id']}")
                logger.success(f"All {total_comments_bound} comments bound successfully.")
                return True
            except Exception as e:
                logger.exception(f"Error while binding comments: {e}")
                return False

        def tickets_reshaped(self, wranglerInstance, progress=gr.Progress(track_tqdm=True)) -> bool:
            """Reshapes ticket data from a JSON file into Ticket objects."""
            try:
                with open(wranglerInstance.ticket_file, "r") as tickets_file:
                    tickets_data = json.load(tickets_file)
                    for ticket_data in tqdm(tickets_data):
                        reshaped_ticket = self._reshape_ticket(ticket_data)
                        wranglerInstance.wrangled_tickets.append(reshaped_ticket)
                        logger.debug(f"Appended ticket {reshaped_ticket.id} to wrangled tickets.")
                logger.success(f"All {len(wranglerInstance.wrangled_tickets)} tickets reshaped successfully - {len(wranglerInstance.wrangled_tickets)/len(tickets_data) * 100} Success Rate")
                return True
            except Exception as e:
                logger.exception(f"Failed to reshape tickets: {e}")
                return False

        def create_corpus(self, wranglerInstance,ui_download_element,  progress=gr.Progress(track_tqdm=True)) -> List[str]:
            """Creates a text corpus from the wrangled tickets and their comments."""
            corpus = []
            try:
                for filename in tqdm(wranglerInstance.comments_dir):
                    with open(filename, 'r') as file:
                        file = json.loads(file.read())
                        ticket_no = filename.split("/")[-1]
                        ticket_no = ticket_no.split(".")[0]
                        for comment in tqdm(file[ticket_no]):
                            cleansed_comment = self._cleanse(comment['plain_body'])
                            corpus.append(cleansed_comment)
                logger.debug("Corpus created successfully.")
                final_corpus = " ".join(corpus)
                wranglerInstance.corpus = final_corpus
                wranglerInstance.generate_corpus_json(ui_download_element=ui_download_element)
                return final_corpus
            except Exception as e:
                logger.exception(f"Failed to create corpus: {e}")
                return " "

        def run_async(self, wranglerInstance, ui_download_element):
            """Run the complete wrangling process asynchronously."""
            try:
                logger.log("APPLICATION MESSAGE", "Starting ticket reshaping...")
                future_tickets = self.executor.submit(lambda: self.tickets_reshaped(wranglerInstance))


                if task := self.create_corpus(
                    wranglerInstance=wranglerInstance, ui_download_element=ui_download_element
                ):
                    wranglerInstance.corpus = task
                    logger.log("APPLICATION MESSAGE", f"Corpus created with {len(wranglerInstance.corpus)} entries. {str(task)}")
                if future_tickets.result():
                    logger.log("APPLICATION MESSAGE", "Tickets reshaped successfully.")
                    logger.log("APPLICATION MESSAGE", "Starting comment binding...")
                    future_comments = self.executor.submit(lambda: self.comments_bound(wranglerInstance=wranglerInstance))
                    if future_comments.result():
                        logger.log("APPLICATION MESSAGE", "Comments bound successfully.")

                    else:
                        logger.log("APPLICATION MESSAGE", "Failed to bind comments.")
                else:
                    logger.log("APPLICATION MESSAGE", "Failed to reshape tickets.")
            except Exception as e:
                logger.exception(f"Error in async processing: {e}")
                logger.log("APPLICATION MESSAGE", f"Error: {e}")

        def _reshape_ticket(self, ticket_data: dict):
            """Helper method to reshape ticket data into a Ticket object."""
            reshaped_ticket = Ticket(
                id=ticket_data["id"],
                created_at=datetime.strptime(ticket_data["created_at"], "%Y-%m-%dT%H:%M:%SZ"),
                last_updated=datetime.strptime(ticket_data["updated_at"], "%Y-%m-%dT%H:%M:%SZ"),
                subject=ticket_data["subject"],
                tags=ticket_data.get("tags", []),
                outcome=ticket_data["fields"][2]["value"],
                ticket_type=ticket_data["fields"][0]["value"],
                status=TicketStatus[ticket_data["status"].upper()],
            )
            first_comment = Comment(
                id=ticket_data["id"],  # Generate a comment ID based on the ticket ID
                created_at=datetime.strptime(ticket_data["created_at"], "%Y-%m-%dT%H:%M:%SZ"),
                body=ticket_data["description"],
            )
            reshaped_ticket.comments.append(first_comment)
            return reshaped_ticket

        def _reshaped_comment(self, comment_data: dict):
            """Helper method to reshape comment data."""
            return Comment(
                id=comment_data.get("id", 0),
                created_at=datetime.strptime(
                    comment_data["created_at"], "%Y-%m-%dT%H:%M:%SZ"
                ),
                body=self._cleanse(comment_data["plain_body"]),
            )