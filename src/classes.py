import json
import os
import pathlib
from datetime import datetime
from enum import Enum
from typing import List, TextIO

from loguru import logger

logger.add(sink="./wrangle_log.log", colorize=True, serialize=True)


class TicketStatus(Enum):
    OPEN = 1
    HOLD = 2
    PENDING = 3
    SOLVED = 4
    CLOSED = 5


class Comment:
    def __init__(self, id: int, created_at: datetime, body: str) -> None:
        self.id: int = id
        self.created_at: datetime = created_at
        self.body: str = body

    def to_dict_format(self) -> str:
        created_date = datetime.strptime(self.created_at, "%Y-%d-%m")
        comment_dict = {created_date: {id: self.id, body: self.body}}
        return json.dumps(comment_dict)


class Ticket:
    def __init__(
        self,
        id: int,
        created_at: datetime,
        status: TicketStatus,
        last_updated: datetime,
        subject: str,
        tags: List[str] = [],
        outcome: str = None,
        type: str = None,
    ) -> None:
        self.id: int = id
        self.created_at: datetime = created_at
        self.last_updated: datetime = last_updated
        self.status: TicketStatus = status
        self.tags: List[str] = tags
        self.outcome: str = outcome
        self.type: str = type
        self.comments: List[Comment] = []


class DataWrangler:
    def __init__(
        self,
        comments_dir: pathlib.Path = pathlib.Path.cwd(),
        ticket_file: TextIO = os.path.join(pathlib.Path.cwd(), "tickets.json"),
    ) -> None:
        self.ticket_file: pathlib.Path = ticket_file
        self.comments_dir = comments_dir
        self.wrangled_tickets = []

    @classmethod
    def tickets_reshaped(cls) -> bool:
        try:
            with open(cls.ticket_file, "r") as tickets:
                for ticket in tickets:
                    ticket = json.loads(ticket)
                    reshaped_ticket = Ticket(
                        id=ticket.id,
                        created_at=datetime.strptime(
                            ticket.created_at, "%Y-%d-%mT%H:%M:%S"
                        ),
                        last_updated=datetime.strptime(
                            ticket.last_updated, "YYYY-MM-DDTHH:mm:ss"
                        ),
                        subject=ticket.subject,
                        tags=ticket.tags,
                        outcome=ticket.fields[2].value,
                        type=ticket.fields[0].value,
                    )
                    cls.wrangled_tickets.append(reshaped_ticket)
                    logger.success(f"Successfully Reshaped {ticket.id}")
            return True
        except Exception as e:
            logger.exception(f"Failed to Reshape Ticket")
            return False
            # raise RuntimeError from e

    @staticmethod
    def reshaped_comment(comment:str) -> Comment:
        comment = json.loads(comment)
        try:
            reshaped_comment = Comment(
                id=comment.id, created_at=comment.created_at, body=comment.plain_body
            )
            return reshaped_comment
        except Exception as e:
            logger.exception(f"Failed to Reshape comment - {e}")
            raise RuntimeError from e

    @classmethod
    def comments_bound(cls) -> bool:
        try:
            for ticket in cls.wrangled_tickets:
                if len(ticket.comments) > 1:
                    logger.warning(
                        f"Prebound Ticket {ticket.id} Found...Clearing and Re-binding"
                    )
                    del ticket.comments[1:]
                for _, _, files in os.walk(cls.comments_dir):
                    for filename in files:
                        if ticket.id == filename.split()[0]:
                            logger.info(
                                f"Corrosponding Comments File Found for {ticket.id}...Binding Comments"
                            )
                            with open(filename, "r") as comments:
                                comments = json.loads(comments)
                                total_comments = len(comments)
                                for comment in comments:
                                    c_comment = DataWrangler.reshaped_comment(comment)                                     
                                    ticket.comments.append()
                                    total_comments -= 1
                                    logger.info(
                                        f"Bound Comment {c_comment.id} - {total_comments} left to bind."
                                    )
            return True
        except Exception as e:
            logger.exception(e)
            # raise RuntimeError from e

    def process(self) -> None:
        try:
            if self.tickets_reshaped():
                if self.comments_bound():
                    logger.success('Data Successfully Wrangled')
                    return True
        except Exception as e:
            logger.exception(e)
            raise RuntimeError('Unable to Process') from e
