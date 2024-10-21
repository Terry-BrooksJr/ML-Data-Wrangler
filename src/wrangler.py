import json
import os
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
from tqdm import tqdm


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

    def generate_json(
        self,
        filename: str = f"processed_tickets{datetime.now().strftime('%Y-%m-%d')}.json",
    ) -> Tuple[TextIO, TextIO]:
        """
        Generates JSON files for processed tickets and the associated corpus.

        This function creates two JSON files: one containing the processed tickets and another containing the corpus data. The filenames are constructed based on the current date, and the function returns file handles for both output files.

        Args:
            filename (str, optional): The name of the output file for processed tickets. Defaults to "processed_tickets" followed by the current date.

        Returns:
            Tuple[TextIO, TextIO]: A tuple containing the file handles for the processed tickets and the corpus JSON files.

        Raises:
            IOError: If there is an issue opening or writing to the output files.
        """

        def construct_path(filename):
            """
            Constructs a file path for a given filename in the 'completed' directory.

            This function takes a filename as input and returns the full path by joining the current working directory with the 'completed' directory and the provided filename. This ensures that the file is correctly located within the specified directory structure.

            Args:
                filename (str): The name of the file for which the path is to be constructed.

            Returns:
                str: The full path to the specified file in the 'completed' directory.
            """
            return os.path.join(pathlib.Path.cwd(), "completed", filename)

        filename = construct_path(filename)
        corpus_filename = construct_path(
            f"corpus_{datetime.now().strftime('%Y-%m-%d')}.json"
        )

        with open(filename, "w+") as output1:
            json.dump(
                [ticket.__dict__ for ticket in self.wrangled_tickets],
                output1,
                indent=4,
                cls=MyEncoder,
            )

            with open(corpus_filename, "w+") as output2:
                json.dump(
                    self.corpus,
                    output2,
                    indent=4,
                    cls=MyEncoder,
                )
            return (output1, output2)

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
            check_body = body_of_text.splitlines()
            cleaned_lines = [
                unicodedata.normalize("NFKC", unescape(line))
                .replace("\n", " ")
                .replace("\r", " ")
                for line in check_body
            ]
            scrubbed = [
                word for word in cleaned_lines
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
            logger.success("Successfully Cleansed Corpus")
            return scrubbed

        def _bind_comments(self, ticket: Ticket,wranglerInstance):

            """Helper method to bind comments to a single ticket."""
            self.on_status = WORKER_STATUS.BUSY
            comments_found = False
            for filename in wranglerInstance.comments_dir:
                if filename.split()[-2] == ticket.id:       
                    logger.info(f"Binding comments for ticket {ticket.id}")
                    with open(filename, "r") as comments_file:
                        comments_data = json.load(comments_file)
                        for key, value in comments_data.items():
                            for comment in value:
                                reshaped_comment = self._reshaped_comment(comment)
                                ticket.comments.append(reshaped_comment.to_dict_format())
                                comments_found = True
            if not comments_found:
                logger.warning(f"No comments found for ticket {ticket.id}")
            self.on_status = WORKER_STATUS.FINISHED
            return ticket

        def comments_bound(self, wranglerInstance) -> bool:
            """Binds comments to the wrangled tickets asynchronously."""
            try:
                futures = []
                for ticket in wranglerInstance.wrangled_tickets:
                    future = self.executor.submit(self._bind_comments, ticket)
                    futures.append(future)
                for future in futures:
                    result = future.result()  # Wait for the result
                    logger.info(f"Finished binding comments for ticket {result.id}")
                logger.success("All comments bound successfully.")
                return True
            except Exception as e:
                logger.exception(f"Error while binding comments: {e}")
                return False

        def tickets_reshaped(self, wranglerInstance) -> bool:
            """Reshapes ticket data from a JSON file into Ticket objects."""
            try:
                with open(wranglerInstance.ticket_file, "r") as tickets_file:
                    tickets_data = json.load(tickets_file)
                    for ticket_data in tickets_data:
                        reshaped_ticket = self._reshape_ticket(ticket_data)
                        wranglerInstance.wrangled_tickets.append(reshaped_ticket)
                        logger.info(f"Appended ticket {reshaped_ticket.id} to wrangled tickets.")
                logger.success(f"All {len(wranglerInstance.wrangled_tickets)} tickets reshaped successfully - {len(wranglerInstance.wrangled_tickets)/len(tickets_data) * 100} Success Rate")
                return True
            except Exception as e:
                logger.exception(f"Failed to reshape tickets: {e}")
                return False

        def create_corpus(self, wranglerInstance) -> List[str]:
            """Creates a text corpus from the wrangled tickets and their comments."""
            corpus = []
            try:
                for filename in wranglerInstance.comments_dir:
                    with open(filename, 'r') as file:
                        file = json.loads(file.read())
                        ticket_no = filename.split("/")[-1]
                        ticket_no = ticket_no.split(".")[0]
                        for comment in file[ticket_no]:
                            cleansed_comment = self._cleanse(comment['plain_body'])
                            corpus.extend(cleansed_comment)
                logger.success("Corpus created successfully.")
                wranglerInstance.generate_json()
                return corpus
            except Exception as e:
                logger.exception(f"Failed to create corpus: {e}")
                return []

        def run_async(self, wranglerInstance):
            """Run the complete wrangling process asynchronously."""
            try:
                self.on_status("Starting ticket reshaping...")
                future_tickets = self.executor.submit(lambda: self.tickets_reshaped(wranglerInstance))
                corpus_task = self.create_corpus(wranglerInstance=wranglerInstance)
                if corpus_task:
                    wranglerInstance.corpus = corpus_task
                    self.on_status(f"Corpus created with {len(wranglerInstance.corpus)} entries.", task_complete=True)
                if future_tickets.result():
                    self.on_status("Tickets reshaped successfully.", task_complete=True)
                    self.on_status("Starting comment binding...")
                    future_comments = self.executor.submit(lambda: self.comments_bound(wranglerInstance=wranglerInstance))
                    if future_comments.result():
                        self.on_status("Comments bound successfully.", task_complete=True)
                        self.on_status("Creating corpus...")
                 
                    else:
                        self.on_status("Failed to bind comments.", is_failure=True)
                else:
                    self.on_status("Failed to reshape tickets.", is_failure=True)
            except Exception as e:
                logger.exception(f"Error in async processing: {e}")
                self.on_status(f"Error: {e}", is_failure=True)

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
            reshaped_comment = Comment(
                id=comment_data.get("id", 0),
                created_at=datetime.strptime(comment_data["created_at"], "%Y-%m-%dT%H:%M:%SZ"),
                body=comment_data["body"]
            )
            return reshaped_comment