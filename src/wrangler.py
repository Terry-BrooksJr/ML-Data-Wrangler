import json
import os
import pathlib
import random
import re
import unicodedata
from datetime import datetime
from enum import Enum
from html import unescape
from typing import List, TextIO, Tuple

from loguru import logger

from utility import remove_urls


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


class DataWrangler:
    """A class for processing and managing ticket data and associated comments.

    This class provides methods to reshape ticket data from JSON files, bind
    comments to tickets, create a text corpus from the tickets and comments,
    clean text data, and generate JSON files containing processed ticket data.
    It facilitates the organization and analysis of ticket-related information.
    """

    def __init__(
        self,
        comments_dir: pathlib.Path = pathlib.Path.cwd(),
        ticket_file: pathlib.Path = pathlib.Path.cwd() / "tickets.json",
    ) -> None:
        """Initializes the DataWrangler with specified directories for comments and tickets.

        Args:
            comments_dir: The directory where comment files are stored.
            ticket_file: The path to the JSON file containing ticket data.
        """
        self.ticket_file = ticket_file
        self.comments_dir = comments_dir
        self.wrangled_tickets: List[Ticket] = []
        self.corpus: str = ""

    def tickets_reshaped(self) -> bool:
        """Reshapes ticket data from a JSON file into Ticket objects.

        This method reads ticket data from a specified JSON file and converts
        each ticket into a Ticket object, including associated comments. It
        logs the success of each reshaping operation and returns a boolean
        indicating the overall success of the process.

        Returns:
            True if all tickets are successfully reshaped and added to the
            wrangled tickets list, False otherwise.

        Raises:
            Exception: If there is an error during the reading or reshaping
            process.
        """
        try:
            with open(self.ticket_file, "r") as tickets_file:
                tickets_data = json.load(tickets_file)
                for ticket in tickets_data:
                    reshaped_ticket = Ticket(
                        id=ticket["id"],
                        created_at=datetime.strptime(
                            ticket["created_at"], "%Y-%m-%dT%H:%M:%SZ"
                        ),
                        last_updated=datetime.strptime(
                            ticket["updated_at"], "%Y-%m-%dT%H:%M:%SZ"
                        ),
                        subject=ticket["subject"],
                        tags=ticket.get("tags", []),
                        outcome=ticket["fields"][2]["value"],
                        ticket_type=ticket["fields"][0]["value"],
                        status=TicketStatus[ticket["status"].upper()],
                    )
                    first_comment = Comment(
                        id=random.randint(9999, 9999999999999),
                        created_at=datetime.strptime(
                            ticket["created_at"], "%Y-%m-%dT%H:%M:%SZ"
                        ),
                        body=ticket["description"],
                    )
                    reshaped_ticket.comments.append(first_comment)

                    self.wrangled_tickets.append(reshaped_ticket)
                    logger.success(f"Successfully reshaped ticket {ticket['id']}")
            return True
        except Exception as e:
            logger.exception(f"Failed to reshape tickets: {e}")
            return False

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

    def comments_bound(self) -> bool:
        """Binds comments from files to their corresponding tickets.

        This method iterates through the wrangled tickets and attempts to match
        comments stored in files within a specified directory. It loads the comments
        associated with each ticket, reshapes them, and appends them to the ticket's
        comments list, logging the process and any issues encountered.

        Returns:
            True if comments are successfully bound to the tickets, False otherwise.

        Raises:
            Exception: If there is an error during the binding process.
        """
        try:
            for ticket in self.wrangled_tickets:
                for filename in os.listdir(self.comments_dir):
                    if filename.startswith(str(ticket.id)):
                        comments_file_path = os.path.join(self.comments_dir, filename)
                        logger.info(f"Binding comments for ticket {ticket.id}")
                        with open(comments_file_path, "r") as comments_file:
                            comments_data = json.load(comments_file)
                            # Check if the ticket ID exists in comments_data
                            for key, value in comments_data.items():
                                for comment in value:
                                    reshaped_comment = self.reshaped_comment(comment)
                                    ticket.comments.append(
                                        reshaped_comment.to_dict_format()
                                    )
                            logger.warning(f"No comments found for ticket {ticket.id}")
                            for key, value in comments_data.items():
                                print(f"key:{key}, value:{value}")
                        logger.success(f"Comments bound to ticket {ticket.id}")
            return True
        except Exception as e:
            logger.exception(f"Error while binding comments: {e}")
            return False

    def create_corpus(self):
        """Creates a text corpus from the wrangled tickets and their comments.

        This method iterates through the wrangled tickets, extracting and merging
        the comments associated with each ticket into a single corpus string.
        It logs the progress and success of the corpus creation process, providing
        feedback on the comments being processed.

        Returns:
            True if the corpus is successfully created, False otherwise.

        Raises:
            Exception: If there is an error during the corpus creation process.
        """
        grouped_comments = []

        try:
            for ticket in self.wrangled_tickets:
                logger.info(f"Selecting {ticket.id} for Corpus Merge")
                for comment in ticket.comments:
                    if not isinstance(comment, dict):
                        comment = comment.__dict__
                    logger.info(f"Selecting comment {comment['id']}")
                    clean_comment = self.cleanse(comment["body"])
                    grouped_comments.append(clean_comment)
                    logger.success(f"Merged comment  {comment['id']} into corpus")
            self.corpus = " ".join(grouped_comments)
            logger.success("Corpus Successfully Created")
            return True
        except Exception:
            logger.exception("Failed to Create Corpus")
            return False

    def cleanse(self, body_of_text: str) -> str:
        """
        Cleanses the provided text by normalizing and unescaping each line.
        This function processes the input text to ensure it is in a consistent format.

        The `cleanse` method iterates through each line of the input text, applying normalization
        and unescaping to ensure that the text is properly formatted. The cleaned lines are then
        concatenated and returned as a single string.

        Args:
            body_of_text (str): The text to be cleansed, consisting of multiple lines.

        Returns:
            str: The cleansed text, with all lines normalized and unescaped.
        """

        try:
            cleaned_lines: List[str] = []
            for line in body_of_text:
                line = unicodedata.normalize("NFKC", unescape(line))
                line = line.replace("\n", " ")
                line = line.replace("\r", " ")
                line = remove_urls(line)
                cleaned_lines.append(line)
            logger.success("Successfully Cleansed Corpus")
            return "".join(cleaned_lines)
        except Exception:
            logger.exception("Failed To Cleanse Corpus")

    def generate_json(
        self,
        filename: str = f"processed_tickets{datetime.now().strftime('%Y-%m-%d')}.json",
    ) -> Tuple[TextIO, TextIO]:
        """Generates a JSON file containing processed ticket data and returns the file object.

        This method creates a JSON file that stores the details of the processed
        tickets in a structured format. If no filename is provided, it defaults
        to a generated filename based on the current date, and the method returns
        the file object for further operations if needed.

        Args:
            filename: The name of the file to which the JSON data will be written.
                    If None, a default filename will be generated.

        Returns:
            The file object of the created JSON file.

        Raises:
            IOError: If there is an error writing to the file.
        """

        filename = os.path.join(pathlib.Path.cwd(), "completed", filename)
        corpus_filename = os.path.join(
            pathlib.Path.cwd(),
            "completed",
            f"corpus_{{datetime.now().strftime('%Y-%m-%d')}}",
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
