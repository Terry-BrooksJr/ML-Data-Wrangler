import json
import os
import pathlib
from datetime import datetime
from enum import Enum
from typing import List, TextIO
import random
from loguru import logger

logger.add(sink="./wrangle_log.log", colorize=True, serialize=True)

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
       if isinstance(obj, TicketStatus): 
           return { "status" : obj.name}
       elif isinstance(obj,datetime):
           return obj.isoformat() 
       return json.JSONEncoder.default(self, obj)

  

class TicketStatus(Enum):
    OPEN = 1
    HOLD = 2
    PENDING = 3
    SOLVED = 4
    CLOSED = 5

class Comment:
    def __init__(self, id: int, created_at: datetime, body: str) -> None:
        self.id = id
        self.created_at = created_at
        self.body = body

    def to_dict_format(self) -> dict:
        return {
            "created_at": self.created_at,
            "id": self.id,
            "body": self.body,
        }


class Ticket:
    def __init__(
        self,
        id: int,
        created_at: datetime,
        status: TicketStatus,
        last_updated: datetime,
        subject: str,
        tags: List[str] = None,
        outcome: str = None,
        type_: str = None,
    ) -> None:
        self.id = id
        self.created_at = created_at
        self.last_updated = last_updated
        self.status = status
        self.subject = subject
        self.tags = tags or []
        self.outcome = outcome
        self.type = type_
        self.comments: List[Comment] = []


class DataWrangler:
    def __init__(
        self,
        comments_dir: pathlib.Path = pathlib.Path.cwd(),
        ticket_file: pathlib.Path = pathlib.Path.cwd() / "tickets.json",
    ) -> None:
        self.ticket_file = ticket_file
        self.comments_dir = comments_dir
        self.wrangled_tickets: List[Ticket] = []
        self.corpus: str = ""

    def tickets_reshaped(self) -> bool:
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
                        type_=ticket["fields"][0]["value"],
                        status=TicketStatus[ticket["status"].upper()],
                    )
                    first_comment = Comment(id=random.randint(9999,9999999999), created_at=datetime.strptime(ticket["created_at"], "%Y-%m-%dT%H:%M:%SZ"), body=ticket["description"])
                    reshaped_ticket.comments.append(first_comment)

                    self.wrangled_tickets.append(reshaped_ticket)
                    logger.success(f"Successfully reshaped ticket {ticket['id']}")
            return True
        except Exception as e:
            logger.exception(f"Failed to reshape tickets: {e}")
            return False

    @staticmethod
    def reshaped_comment(comment) -> Comment:
        try:
            return Comment(
                id=comment["id"],
                created_at= comment["created_at"],
                body=comment["plain_body"]
            )
        except Exception as e:
            logger.exception(f"Failed to reshape comment: {e}")
            raise RuntimeError("Comment reshaping failed") from e
            
    def comments_bound(self) -> bool:
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
                                    ticket.comments.append(reshaped_comment.to_dict_format())
                            logger.warning(f"No comments found for ticket {ticket.id}")
                            for key,value in comments_data.items():
                                print(f"key:{key}, value:{value}")
                        logger.success(f"Comments bound to ticket {ticket.id}")
            return True
        except Exception as e:
            logger.exception(f"Error while binding comments: {e}")
            return False
    def create_corpous(self):
        try:
            for ticket in self.wrangled_tickets:
                grouped_comments = []
                logger.info(f"Selecting {ticket .id} for Corpus Merge")
                for comment in ticket.comments:
                    print(type(comment))
                    if not isinstance(comment, dict):
                        comment = comment.__dict__
                    logger.info(f"Selecting {comment['id']}")
                    grouped_comments.append(comment["body"])
                    logger.success(f"Merged {comment['id']}")
            self.corpus = " ".join(grouped_comments)
            logger.success('Corpus SUccessfully Created')
            return True
        except Exception as e:
            logger.exception('Failed to Create Corpus')
            return False

    def process(self) -> None:
        try:
            if self.tickets_reshaped() and self.comments_bound() and self.create_corpous():

                output_file = pathlib.Path(f"wrangled_data_{datetime.now().strftime('%Y-%m-%d')}.json")
                with open(output_file, "w") as output:
                    json.dump([ticket.__dict__ for ticket in self.wrangled_tickets], output, indent=4, cls=MyEncoder)
                logger.success("Data successfully wrangled and saved")
        except Exception as e:
            logger.exception(f"Processing failed: {e}")
            raise RuntimeError("Data processing failed") from e

    @staticmethod
    def clean(body: str) -> str:
        return body.strip().replace("\n", "")

                                                