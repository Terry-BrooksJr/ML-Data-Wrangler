import json
import os
import pathlib
import random
import re
import unicodedata
from datetime import datetime, timezone
from enum import Enum
from html import unescape
from typing import Iterable, List, Optional, TextIO, Tuple, Union
from xml.etree import ElementTree as ET

import validators
from loguru import logger
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import DuplicateKeyError
from PyQt5.QtCore import pyqtSignal, QMutex, QObject
from PyQt5.QtWidgets import QMessageBox

from decision_engine.zendesk_status import REVIEW_ADDED_KEY

# Control characters that XML 1.0 cannot represent, even escaped.
# Zendesk custom field IDs (ticket["fields"] is a list of {"id", "value"}).
REQUEST_TYPE_FIELD_ID = 27124341
OUTCOME_FIELD_ID = 27219391


def custom_field(ticket: dict, field_id: int):
    """A ticket's custom field value by its Zendesk ID, or None if it's absent."""
    for field in ticket.get("fields") or []:
        if field.get("id") == field_id:
            return field.get("value")
    return None


XML_ILLEGAL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f￾￿]")


# Unicode spaces Zendesk text carries (mostly from &nbsp;), turned into plain spaces.
_SPECIAL_SPACES = re.compile(r"[\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000]")
# Invisible characters with no meaning in ticket text. U+200C/U+200D (joiners)
# are kept: they shape emoji sequences and some scripts.
_ZERO_WIDTH = re.compile(r"[\u200b\u2060\ufeff]")
# C0/C1 control characters other than tab and newline. ESC (from e.g. &#27;)
# would otherwise reach terminals as ANSI/OSC sequences.
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_TRAILING_SPACES = re.compile(r"[ \t]+$", re.MULTILINE)


def clean_text(text: Optional[str]) -> Optional[str]:
    """Decodes HTML entities in ticket text before it is stored in Mongo.

    Entities are decoded until the text stops changing, so double-escaped
    (or deeper) ones like "&amp;nbsp;" are handled too. Line endings become "\n",
    non-breaking and other special spaces become plain spaces, and control
    characters and invisible zero-width characters are removed. Trailing
    whitespace is stripped from each line, but spaces and tabs inside or at
    the start of a line are kept, so code and indentation survive. Words are
    not removed: unlike _cleanse, which prepares topic-modelling text, this
    keeps the real wording the decision engine needs. Idempotent.

    Returns:
        The cleaned text, or None when given None.
    """
    if text is None:
        return None
    # Decode to a fixed point, so cleaning is idempotent however deeply the
    # text was escaped. Terminates: every pass that changes the text shortens it.
    while (decoded := unescape(text)) != text:
        text = decoded
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_CHARS.sub("", _ZERO_WIDTH.sub("", _SPECIAL_SPACES.sub(" ", text)))
    return _TRAILING_SPACES.sub("", text)


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

    def __getitem__(self, key: str):
        if key not in self.to_dict_format():
            raise KeyError(key)
        return getattr(self, key)

    def __setitem__(self, key: str, value) -> None:
        if key not in self.to_dict_format():
            raise KeyError(key)
        setattr(self, key, value)

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

    def output_to_dict(
        self, output_path: Optional[Union[str, pathlib.Path]] = None
    ) -> dict:
        """
        Converts the ticket to a dictionary and writes it to disk as JSON.

        Args:
            output_path: Where to write the JSON file. Defaults to
                completed/ticket_<id>.json in the current working directory.

        Returns:
            The JSON-compatible dictionary that was written to disk.
        """
        ticket_dict = json.loads(
            json.dumps(
                {
                    "id": self.id,
                    "created_at": self.created_at,
                    "last_updated": self.last_updated,
                    "status": self.status.name,
                    "subject": self.subject,
                    "tags": self.tags,
                    "outcome": self.outcome,
                    "ticket_type": self.ticket_type,
                    "comments": self.comments,
                },
                cls=MyEncoder,
            )
        )

        output_path = pathlib.Path(
            output_path or pathlib.Path.cwd() / "completed" / f"ticket_{self.id}.json"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as output:
            json.dump(ticket_dict, output, indent=4, ensure_ascii=False)
        logger.success(f"Wrote ticket {self.id} to {output_path}")
        return ticket_dict


# Process Data's steps and the share of its progress bar each fills, as
# (start, end) percentages. Writing the JSON files fills the rest.
PROCESS_STAGES = {"reshape": (0, 25), "bind": (25, 70), "corpus": (70, 95)}


def process_progress(stage: str, done: int, total: int) -> int:
    """Overall Process Data progress, 0-100, after `done` of `total` tickets in a stage."""
    start, end = PROCESS_STAGES[stage]
    if total <= 0:
        return end
    return start + (end - start) * min(done, total) // total


class DataWrangler(QObject):
    """A class for processing and managing ticket data and associated comments.

    This class provides methods to reshape ticket data from JSON files, bind
    comments to tickets, create a text corpus from the tickets and comments,
    clean text data, and generate JSON files containing processed ticket data.
    It facilitates the organization and analysis of ticket-related information.

    Signals:
        tickets_loaded: Emitted with the number of tickets read from the ticket file.
        ticket_finished: Emitted when a ticket processing operation is completed.
        ticket_progress: Emitted with the progress of ticket processing.
        binding_finished: Emitted when the binding of comments to tickets is completed.
        binding_progress: Emitted with the progress of comment binding.
        corpus_creation_finished: Emitted when the corpus creation is completed.
        corpus_creation_progress: Emitted with the progress of corpus creation.
        error: Emitted with an error message if an error occurs during processing.
        worker_status: Emitted with the current status of the worker.
    """

    tickets_loaded = pyqtSignal(int)
    ticket_finished = pyqtSignal()
    ticket_progress = pyqtSignal(int)
    binding_finished = pyqtSignal()
    binding_progress = pyqtSignal(int)
    corpus_creation_finished = pyqtSignal()
    corpus_creation_progress = pyqtSignal(int)
    _mutex = QMutex()
    error = pyqtSignal(str)
    worker_status = pyqtSignal(str)
    mongo_write_progress = pyqtSignal(int)
    mongo_write_finished = pyqtSignal()

    def __init__(
        self,
        comments_dir: pathlib.Path = pathlib.Path.cwd() / "comments",
        ticket_file: pathlib.Path = pathlib.Path.cwd() / "tickets-only.json",
    ) -> None:
        """Initializes the DataWrangler with specified directories for comments and tickets.

        Args:
            comments_dir: The directory where comment files are stored.
            ticket_file: The path to the JSON file containing ticket data.
        """
        super().__init__()
        self.ticket_file = ticket_file
        self.comments_dir = comments_dir
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
            os.makedirs(os.path.join(pathlib.Path.cwd(), "completed"), exist_ok=True)
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

    @staticmethod
    def _cleanse(body_of_text: str) -> str:
        """Cleanses the provided text by normalizing and unescaping it, then
        dropping words that are emails, URLs, UUIDs, MD5 hashes, IPv4 addresses
        or contain no letters or digits."""
        if not body_of_text:
            return ""
        text = unicodedata.normalize("NFKC", unescape(body_of_text))

        scrubbed = []
        for word in text.split():
            # Check the word without surrounding punctuation, e.g. "(https://x.com)."
            candidate = word.strip("()[]{}<>\"'.,;:!?")
            if not any(char.isalnum() for char in candidate):
                continue
            if any(
                [
                    validators.email(candidate),
                    validators.url(candidate),
                    validators.uuid(candidate),
                    validators.md5(candidate),
                    validators.ipv4(candidate),
                ]
            ):
                continue
            scrubbed.append(word)

        logger.success("Successfully Cleansed Corpus")
        return " ".join(scrubbed)

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
            self.worker_status.emit("Collecting Ticket Objects for Comment Binding...")
            for i, ticket in enumerate(self.wrangled_tickets):
                self.worker_status.emit(f"Binding Comments for Ticket {ticket.id}")
                logger.info(f"Binding comments for ticket {ticket.id}")
                raw_comments = self._raw_comments_for(ticket.id)
                if raw_comments:
                    # Zendesk's first comment is the description, so the file
                    # replaces the description comment tickets_reshaped added.
                    ticket.comments = []
                for comment in raw_comments:
                    reshaped_comment = self.reshaped_comment(comment)
                    ticket.comments.append(reshaped_comment.to_dict_format())
                if raw_comments:
                    logger.success(f"Comments bound to ticket {ticket.id}")
                else:
                    logger.warning(f"No comments found for ticket {ticket.id}")
                self.binding_progress.emit(i + 1)
            self.binding_finished.emit()
            return True
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.exception(f"Error while binding comments: {e}")
            return False

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
                self.tickets_loaded.emit(len(tickets_data))
                self.worker_status.emit(f"Reshaping {len(tickets_data)} tickets...")
                for i, ticket in enumerate(tickets_data):
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
                        outcome=custom_field(ticket, OUTCOME_FIELD_ID),
                        ticket_type=custom_field(ticket, REQUEST_TYPE_FIELD_ID),
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
                    logger.success(f"Successfully reshaped ticket {ticket['id']}")
                    self.wrangled_tickets.append(reshaped_ticket)
                    logger.info(
                        f"Appended ticket {ticket['id']} to wrangled_tickets property on the caller object"
                    )
                    logger.debug(
                        f"Length of Wrangled Tickets: {len(self.wrangled_tickets)}"
                    )
                    self.ticket_progress.emit(i + 1)
            self.ticket_finished.emit()
            return True
        except Exception as e:
            logger.exception(f"Failed to reshape tickets: {e}")
            return False

    def _raw_comments_for(self, ticket_id: int) -> List[dict]:
        """Loads the raw comment dicts for a ticket from its file in comments_dir.

        Matches files named "<id>.json" or "<id>_<anything>.json", and accepts
        either a JSON list of comments or a dict of comment lists.
        """
        raw_comments = []
        for filename in os.listdir(self.comments_dir):
            if pathlib.Path(filename).stem.split("_")[0] != str(ticket_id):
                continue
            with open(os.path.join(self.comments_dir, filename), "r") as comments_file:
                comments_data = json.load(comments_file)
            if isinstance(comments_data, dict):
                for value in comments_data.values():
                    raw_comments.extend(value)
            else:
                raw_comments.extend(comments_data)
        return raw_comments

    def ticket_to_xml(
        self,
        ticket: Union[int, dict],
        output_path: Optional[Union[str, pathlib.Path]] = None,
        xsd_path: Optional[Union[str, pathlib.Path]] = None,
    ) -> str:
        """Converts a ticket to XML verbatim, without cleansing any text.

        The XML follows the element order of ticket_schema.xsd. Text is copied
        exactly as Zendesk stored it; only control characters that XML cannot
        represent at all are removed. Zendesk exports people and groups as
        numeric IDs, so those IDs are used as the Name/Group/Assignee values.

        Args:
            ticket: A raw Zendesk ticket dict, or a ticket ID to look up in
                ticket_file.
            output_path: Optional path to write the XML to.
            xsd_path: Optional path to ticket_schema.xsd. Sets
                xsi:noNamespaceSchemaLocation and, if lxml is installed,
                validates the XML against the schema.

        Returns:
            The XML document as a string.

        Raises:
            KeyError: If a ticket ID is given and it is not in ticket_file.
        """
        if not isinstance(ticket, dict):
            with open(self.ticket_file, "r") as tickets_file:
                matches = [t for t in json.load(tickets_file) if t["id"] == int(ticket)]
            if not matches:
                raise KeyError(f"Ticket {ticket} not found in {self.ticket_file}")
            ticket = matches[0]

        def text(value) -> str:
            if value is None:
                return ""
            return XML_ILLEGAL_CHARS_RE.sub("", str(value))

        def add(parent, tag, value) -> ET.Element:
            element = ET.SubElement(parent, tag)
            element.text = text(value)
            return element

        root = ET.Element("Ticket")
        if xsd_path:
            root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
            root.set("xsi:noNamespaceSchemaLocation", pathlib.Path(xsd_path).name)

        add(root, "Id", ticket["id"])
        add(root, "Subject", ticket.get("subject"))
        add(root, "SubmittedDate", ticket.get("created_at"))
        add(root, "ReceivedVia", (ticket.get("via") or {}).get("channel"))
        requester = ET.SubElement(root, "Requester")
        add(requester, "Name", ticket.get("requester_id"))
        ccs = ET.SubElement(root, "CCs")
        for cc_id in ticket.get("email_cc_ids") or []:
            add(ET.SubElement(ccs, "CC"), "Name", cc_id)
        add(root, "StatusCategory", ticket.get("status"))
        add(root, "TicketStatus", ticket.get("status"))
        add(root, "Type", ticket.get("type"))
        add(root, "Priority", ticket.get("priority"))
        add(root, "Group", ticket.get("group_id"))
        add(root, "Assignee", ticket.get("assignee_id"))
        add(root, "RequestType", custom_field(ticket, REQUEST_TYPE_FIELD_ID))
        add(root, "TicketOutcome", custom_field(ticket, OUTCOME_FIELD_ID))

        comments = ET.SubElement(root, "Comments")
        raw_comments = self._raw_comments_for(ticket["id"])
        if not raw_comments and ticket.get("description"):
            # The first Zendesk comment is the description, so only fall back
            # to it when the ticket has no comment file.
            raw_comments = [
                {
                    "author_id": ticket.get("requester_id"),
                    "created_at": ticket.get("created_at"),
                    "public": True,
                    "plain_body": ticket["description"],
                }
            ]
        for raw_comment in raw_comments:
            comment = ET.SubElement(comments, "Comment")
            add(comment, "Author", raw_comment.get("author_id"))
            add(comment, "Date", raw_comment.get("created_at"))
            add(
                comment,
                "IsInternalNote",
                "false" if raw_comment.get("public", True) else "true",
            )
            add(comment, "Text", raw_comment.get("plain_body"))

        # ET.indent only adds whitespace between elements, so comment text stays verbatim.
        ET.indent(root, space="  ")
        xml_text = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            + ET.tostring(root, encoding="unicode")
            + "\n"
        )

        if xsd_path:
            try:
                from lxml import etree
            except ImportError:
                logger.warning("lxml not installed; skipping schema validation.")
            else:
                schema = etree.XMLSchema(etree.parse(str(xsd_path)))
                schema.assertValid(etree.fromstring(xml_text.encode("utf-8")))
                logger.success(f"Ticket {ticket['id']} XML is valid against {xsd_path}")

        if output_path:
            with open(output_path, "w", encoding="utf-8") as output:
                output.write(xml_text)
            logger.success(f"Wrote ticket {ticket['id']} XML to {output_path}")

        return xml_text

    @staticmethod
    def _parse_zendesk_time(value: Optional[str]) -> Optional[datetime]:
        """Parses Zendesk's "2026-08-18T18:53:13Z" timestamps as aware UTC datetimes."""
        if not value:
            return None
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )

    @staticmethod
    def _transcript_key(entry: dict) -> tuple:
        """Identifies a transcript entry for de-duplication across ingests.

        Uses the Zendesk comment ID when there is one, otherwise falls back to
        Author + Date + Text, as the ingestion requirements allow. The text is
        compared after clean_text, so bodies stored before cleaning was added
        still match their cleaned re-ingested copies.
        """
        if entry.get("comment_id") is not None:
            return ("id", entry["comment_id"])
        timestamp = entry.get("timestamp")
        if isinstance(timestamp, datetime) and timestamp.tzinfo is not None:
            # pymongo returns naive UTC datetimes, so compare in that form.
            timestamp = timestamp.astimezone(timezone.utc).replace(tzinfo=None)
        return (
            "content",
            entry.get("author"),
            timestamp,
            clean_text(entry.get("body")),
        )

    def ticket_to_document(self, ticket: Union[int, dict]) -> dict:
        """Maps a raw Zendesk ticket and its comments to a Mongo `tickets` document.

        The shape is the `tickets` collection the decision engine reads
        (decision_engine.payload builds the model's input from it). The subject and comment text go through clean_text (HTML
        entities decoded, control characters removed) but are not cleansed:
        no words are dropped, because the decision engine needs the real
        wording. raw_xml keeps the original text.

        Args:
            ticket: A raw Zendesk ticket dict, or a ticket ID to look up in
                ticket_file.

        Returns:
            The document, without the review-state fields (`awaiting_review`,
            `agent_reviews`, ...) that write_ticket_to_mongo manages.
        """
        if not isinstance(ticket, dict):
            with open(self.ticket_file, "r") as tickets_file:
                matches = [t for t in json.load(tickets_file) if t["id"] == int(ticket)]
            if not matches:
                raise KeyError(f"Ticket {ticket} not found in {self.ticket_file}")
            ticket = matches[0]

        def as_id(value) -> Optional[str]:
            return None if value is None else str(value)

        requester_id = as_id(ticket.get("requester_id"))
        raw_comments = self._raw_comments_for(ticket["id"])
        if not raw_comments and ticket.get("description"):
            # Mirrors ticket_to_xml: the description is the first comment.
            raw_comments = [
                {
                    "author_id": ticket.get("requester_id"),
                    "created_at": ticket.get("created_at"),
                    "public": True,
                    "plain_body": ticket["description"],
                }
            ]

        transcript = []
        for raw_comment in raw_comments:
            author = as_id(raw_comment.get("author_id"))
            transcript.append(
                {
                    "comment_id": as_id(raw_comment.get("id")),
                    "author": author,
                    # Only the requester can be identified without a Zendesk
                    # users lookup; every other author's role stays unknown.
                    "role": (
                        "requester"
                        if author is not None and author == requester_id
                        else None
                    ),
                    # Entities decoded here; raw_xml keeps the original text.
                    "body": clean_text(raw_comment.get("plain_body")),
                    "timestamp": self._parse_zendesk_time(
                        raw_comment.get("created_at")
                    ),
                    "is_internal_note": not raw_comment.get("public", True),
                }
            )

        return {
            "zendesk_id": str(ticket["id"]),
            "subject": clean_text(ticket.get("subject")),
            "received_via": (ticket.get("via") or {}).get("channel"),
            "status": ticket.get("status"),
            "status_category": ticket.get("status"),
            "ticket_type": ticket.get("type"),
            "priority": ticket.get("priority"),
            "group": as_id(ticket.get("group_id")),
            "assignee": as_id(ticket.get("assignee_id")),
            "request_type": custom_field(ticket, REQUEST_TYPE_FIELD_ID),
            "historical_outcome": custom_field(ticket, OUTCOME_FIELD_ID),
            "tags": ticket.get("tags") or [],
            "requester_id": requester_id,
            "organization_id": as_id(ticket.get("organization_id")),
            "cc_list": [str(cc_id) for cc_id in ticket.get("email_cc_ids") or []],
            # TODO(open item #8): no stable customer identifier is agreed yet,
            # and the export has no requester email for the interim key.
            "customer_id": None,
            # Jira links aren't in the ticket export: they are imported
            # separately into jira_links (decision_engine.jira_links), which
            # this document leaves out so a re-ingest doesn't clear them.
            "raw_xml": self.ticket_to_xml(ticket),
            "transcript": transcript,
            "created_at": self._parse_zendesk_time(ticket.get("created_at")),
            "updated_at": self._parse_zendesk_time(ticket.get("updated_at")),
        }

    @staticmethod
    def get_tickets_collection() -> Collection:
        """Connects to the `tickets` collection named by MONGO_URI, MONGO_DB and MONGO_COLLECTION."""
        mongo = MongoClient(os.environ["MONGO_URI"])
        database = mongo[os.environ.get("MONGO_DB", "support_db")]
        return database[os.environ.get("MONGO_COLLECTION", "tickets")]

    def write_ticket_to_mongo(
        self, ticket: Union[int, dict], collection: Optional[Collection] = None
    ) -> dict:
        """Inserts or updates one ticket in Mongo, keyed by zendesk_id.

        New tickets are inserted with `awaiting_review: true`. For tickets
        already stored, only comments not seen before are appended and the
        mutable fields are refreshed; `awaiting_review` is set back to true
        only when there are new comments. Re-running on the same data changes
        nothing, so the write is idempotent. This never calls the decision
        engine and never writes to Zendesk.

        Args:
            ticket: A raw Zendesk ticket dict, or a ticket ID to look up in
                ticket_file.
            collection: The `tickets` collection. Defaults to
                get_tickets_collection().

        Returns:
            A summary: zendesk_id, whether it was inserted, and how many new
            comments were appended.
        """
        collection = (
            collection if collection is not None else self.get_tickets_collection()
        )
        document = self.ticket_to_document(ticket)
        zendesk_id = document["zendesk_id"]
        now = datetime.now(timezone.utc)

        existing = collection.find_one({"zendesk_id": zendesk_id}, {"transcript": 1})
        if existing is None:
            document.update(
                {
                    "awaiting_review": True,
                    "reengage_attempts": [],
                    "agent_reviews": [],
                    "outcomes": [],
                    REVIEW_ADDED_KEY: False,
                    "last_ingest": {
                        "at": now,
                        "new_comments": len(document["transcript"]),
                    },
                }
            )
            try:
                collection.insert_one(document)
                logger.success(f"Inserted ticket {zendesk_id} into Mongo")
                return {
                    "zendesk_id": zendesk_id,
                    "inserted": True,
                    "new_comments": len(document["transcript"]),
                }
            except DuplicateKeyError:
                # Another writer inserted it first (needs a unique index on
                # zendesk_id); merge into theirs instead.
                document = self.ticket_to_document(ticket)
                existing = collection.find_one(
                    {"zendesk_id": zendesk_id}, {"transcript": 1}
                )

        stored = existing.get("transcript", [])
        seen = {self._transcript_key(entry) for entry in stored}
        candidates = [
            entry
            for entry in document.pop("transcript")
            if self._transcript_key(entry) not in seen
        ]

        # Each push is conditional on the comment not being in the stored
        # transcript, checked atomically by the server, so two writers
        # ingesting the same ticket can't both append it.
        appended = 0
        for entry in candidates:
            appended += collection.update_one(
                {"zendesk_id": zendesk_id, **self._absent_from_transcript(entry)},
                {"$push": {"transcript": entry}, "$set": {"awaiting_review": True}},
            ).modified_count

        # Bodies stored before clean_text existed (or by an older app version)
        # are cleaned as the ticket is re-ingested. Guarded on the old value,
        # so a newer write is never overwritten.
        for index, entry in enumerate(stored):
            body = entry.get("body")
            cleaned = clean_text(body)
            if cleaned != body:
                collection.update_one(
                    {"zendesk_id": zendesk_id, f"transcript.{index}.body": body},
                    {"$set": {f"transcript.{index}.body": cleaned}},
                )

        # raw_xml is refreshed too, so it always matches the latest payload.
        collection.update_one(
            {"zendesk_id": zendesk_id},
            {
                "$set": {
                    **document,
                    "last_ingest": {"at": now, "new_comments": appended},
                }
            },
        )
        logger.success(
            f"Updated ticket {zendesk_id} in Mongo ({appended} new comments)"
        )
        return {"zendesk_id": zendesk_id, "inserted": False, "new_comments": appended}

    @staticmethod
    def _absent_from_transcript(entry: dict) -> dict:
        """A query matching tickets whose transcript doesn't contain `entry` yet."""
        if entry.get("comment_id") is not None:
            return {"transcript.comment_id": {"$ne": entry["comment_id"]}}
        match = {
            "author": entry.get("author"),
            "timestamp": entry.get("timestamp"),
            "body": entry.get("body"),
        }
        return {"transcript": {"$not": {"$elemMatch": match}}}

    @staticmethod
    def clean_stored_tickets(
        collection: Optional[Collection] = None, dry_run: bool = False
    ) -> dict:
        """Applies clean_text to tickets already in Mongo (subject and comment bodies).

        For tickets stored before cleaning was added; new writes are cleaned
        by ticket_to_document, and re-ingesting a ticket also cleans its older
        bodies. Each update only applies if the fields still hold the values
        that were read, so text written in the meantime is never reverted;
        such tickets are counted as skipped_changed and picked up by a re-run.
        Run it again once no older app version is writing to the collection.
        Review state is left alone. Safe to re-run.

        Args:
            dry_run: Count what would change without writing anything.

        Returns:
            Counts: scanned, tickets_changed, comments_changed, skipped_changed.
        """
        collection = (
            collection
            if collection is not None
            else DataWrangler.get_tickets_collection()
        )
        summary = {
            "scanned": 0,
            "tickets_changed": 0,
            "comments_changed": 0,
            "skipped_changed": 0,
        }
        for doc in collection.find(
            {}, {"zendesk_id": 1, "subject": 1, "transcript": 1}
        ):
            summary["scanned"] += 1
            update, guard = {}, {"_id": doc["_id"]}
            subject = clean_text(doc.get("subject"))
            if subject != doc.get("subject"):
                update["subject"] = subject
                guard["subject"] = doc.get("subject")
            for index, entry in enumerate(doc.get("transcript") or []):
                body = clean_text(entry.get("body"))
                if body != entry.get("body"):
                    update[f"transcript.{index}.body"] = body
                    guard[f"transcript.{index}.body"] = entry.get("body")
            if not update:
                continue
            if (
                not dry_run
                and collection.update_one(guard, {"$set": update}).matched_count == 0
            ):
                summary["skipped_changed"] += 1
                continue
            summary["tickets_changed"] += 1
            summary["comments_changed"] += sum(
                key.startswith("transcript.") for key in update
            )
        logger.info(
            f"clean_stored_tickets ({'dry run' if dry_run else 'applied'}): {summary}"
        )
        return summary

    @staticmethod
    def ensure_ticket_indexes(collection: Collection) -> bool:
        """Creates the unique index on zendesk_id if the collection has none.

        Uniqueness is what makes concurrent inserts of the same new ticket
        safe: the losing insert raises DuplicateKeyError and
        write_ticket_to_mongo merges into the winner's document. An existing
        non-unique index on zendesk_id is left alone with a warning, since
        converting it needs collMod or a drop and rebuild. Idempotent.

        Returns:
            True if a unique index on zendesk_id is in place.
        """
        for name, info in collection.index_information().items():
            if list(info.get("key", [])) == [("zendesk_id", 1)]:
                if info.get("unique"):
                    return True
                logger.warning(
                    f"Index {name} on zendesk_id is not unique; concurrent inserts of a new ticket can create "
                    "duplicates. Recreate it with unique=True."
                )
                return False
        collection.create_index("zendesk_id", unique=True, name="zendesk_id_1")
        logger.info("Created unique index zendesk_id_1")
        return True

    def write_tickets_to_mongo(
        self,
        tickets: Optional[Iterable[Union[int, dict]]] = None,
        collection: Optional[Collection] = None,
    ) -> List[dict]:
        """Writes every ticket in ticket_file (or the given tickets) to Mongo.

        Returns:
            One write_ticket_to_mongo summary per ticket.
        """
        collection = (
            collection if collection is not None else self.get_tickets_collection()
        )
        self.ensure_ticket_indexes(collection)
        if tickets is None:
            with open(self.ticket_file, "r") as tickets_file:
                tickets = json.load(tickets_file)

        summaries = []
        for i, ticket in enumerate(tickets):
            ticket_id = ticket["id"] if isinstance(ticket, dict) else ticket
            self.worker_status.emit(f"Writing Ticket {ticket_id} to MongoDB")
            summaries.append(self.write_ticket_to_mongo(ticket, collection))
            self.mongo_write_progress.emit(i + 1)
        self.mongo_write_finished.emit()
        return summaries

    def create_corpus(self) -> str:
        """Creates a text corpus from the wrangled tickets and their comments."""
        grouped_comments = []
        self.worker_status.emit("Building the corpus...")

        for i, ticket in enumerate(self.wrangled_tickets):
            logger.info(f"Selecting {ticket.id} for Corpus Merge")
            for comment in ticket.comments:
                logger.info(f"Selecting comment {comment['id']}")
                grouped_comments.append(self._cleanse(comment["body"]))
                logger.success(f"Merged comment {comment['id']} into corpus")
            self.corpus_creation_progress.emit(i + 1)

        self.corpus = " ".join(grouped_comments)
        self.corpus_creation_finished.emit()
        logger.success("Corpus Successfully Created")
        return self.corpus

    @classmethod
    def mongo_ticket_document(cls, ticket_doc: dict) -> str:
        """Builds the topic-modelling document for a Mongo `tickets` document.

        Matches ticket_documents (cleansed subject, then each comment), so a
        saved topic model can be applied to tickets read back from Mongo.
        """
        return " ".join(
            [cls._cleanse(ticket_doc.get("subject"))]
            + [
                cls._cleanse(entry.get("body"))
                for entry in ticket_doc.get("transcript") or []
            ]
        )

    def ticket_ids(self) -> List[str]:
        """Zendesk IDs of the wrangled tickets, in ticket_documents order."""
        return [str(ticket.id) for ticket in self.wrangled_tickets]

    def ticket_documents(self) -> List[str]:
        """Returns one cleansed document per wrangled ticket, for topic modelling.

        Each document is the ticket's subject followed by its comments.
        """
        return [
            " ".join(
                [self._cleanse(ticket.subject)]
                + [self._cleanse(comment["body"]) for comment in ticket.comments]
            )
            for ticket in self.wrangled_tickets
        ]
