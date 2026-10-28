"""Who a leadership escalation @mentions, decided by the ticket's assignee.

The roster is the single source for both the prompt's "Leadership recipients"
table (prompts.py renders it from here) and the check that overrides a reply
naming the wrong people (guards.apply_guards), so the two can't disagree.

Rules (from the leadership escalation brief):
- Rich Shupe, VP of Support, is a recipient unless he owns the ticket.
- An engineer's Team Lead is a recipient too. Rich Shupe leads some
  engineers himself (Internal Projects); he is named once.
- The owner is never a recipient.
- An assignee not in the roster (or none) goes to Rich Shupe and needs a
  mapping review; a ticket Rich Shupe owns has no recipient and needs a
  routing review.
"""

from dataclasses import dataclass
from datetime import date, timedelta
from typing import List, Optional, Tuple

VP_ROLE = "VP of Support"
TEAM_LEAD_ROLE = "Team Lead"
ENGINEER_ROLE = "Engineer"
# What a review-needed internal note starts with (the prompt asks for it too).
REVIEW_PREFIX = "Routing review needed:"


@dataclass(frozen=True)
class Person:
    user_id: str  # Zendesk user ID, as the payload's `assignee` holds it
    name: str
    role: str
    team_lead: Optional[str] = None


ROSTER: Tuple[Person, ...] = (
    Person("1085778551", "Rich Shupe", VP_ROLE),
    Person("361165286529", "Allain Dollete", TEAM_LEAD_ROLE),
    Person("361165271809", "Horacio Balserio", TEAM_LEAD_ROLE),
    Person("384720527257", "Stephen Sayage", ENGINEER_ROLE, "Allain Dollete"),
    Person("385812489257", "Michael Fares", ENGINEER_ROLE, "Allain Dollete"),
    Person("372535774197", "Kit Amiel Tumulak", ENGINEER_ROLE, "Allain Dollete"),
    Person("33366497853597", "Terry Bu", ENGINEER_ROLE, "Allain Dollete"),
    Person("12289991012893", "Will Griffiths", ENGINEER_ROLE, "Horacio Balserio"),
    Person("384720516977", "Anthony Mai", ENGINEER_ROLE, "Horacio Balserio"),
    Person("387134954297", "Chris Short", ENGINEER_ROLE, "Horacio Balserio"),
    # Internal Projects Support Engineer, led by Rich Shupe directly.
    Person("12289975534365", "Terry Brooks", ENGINEER_ROLE, "Rich Shupe"),
)
# When someone last confirmed the roster against the real team. Terry
# Brooks reviews it quarterly (taxonomy worksheet, E5); update this date then.
ROSTER_REVIEWED = date(2026, 10, 28)
ROSTER_REVIEW_INTERVAL = timedelta(days=92)


def roster_review_due(today: Optional[date] = None) -> bool:
    """True once a quarter has passed since ROSTER_REVIEWED."""
    return (today or date.today()) - ROSTER_REVIEWED > ROSTER_REVIEW_INTERVAL


VP_SUPPORT = next(p.name for p in ROSTER if p.role == VP_ROLE)
_BY_ID = {p.user_id: p for p in ROSTER}


def is_support(user_id) -> bool:
    """True if user_id is someone on the roster (anyone in Support)."""
    return owner(user_id) is not None


def owner(assignee) -> Optional[Person]:
    """The roster entry whose user ID equals assignee, or None."""
    return _BY_ID.get(str(assignee).strip()) if assignee is not None else None


def recipients_for(assignee) -> Tuple[List[str], Optional[str]]:
    """(recipients, review reason) for a leadership escalation on this assignee.

    Recipients are in the order the prompt asks for: Team Lead first, then
    Rich Shupe. The reason is None unless a person must review the routing.
    """
    person = owner(assignee)
    if person is None:
        shown = "none" if assignee in (None, "") else assignee
        reason = f"the assignee's user ID ({shown}) isn't in the leadership roster."
        return [VP_SUPPORT], reason
    if person.name == VP_SUPPORT:
        return (
            [],
            f"{VP_SUPPORT} owns the ticket and there is no one else to escalate to.",
        )
    names = [person.team_lead] if person.team_lead else []
    names.append(VP_SUPPORT)
    # dict.fromkeys keeps the order and names a lead who is also the VP once.
    return [n for n in dict.fromkeys(names) if n != person.name], None


def format_recipients(names: List[str]) -> Optional[str]:
    """How leadership_recipient holds them: comma-separated, or None for nobody."""
    return ", ".join(names) or None


def parse_recipients(value) -> List[str]:
    """The names in a leadership_recipient string, in order, without blanks."""
    return [n.strip() for n in str(value or "").split(",") if n.strip()]


def roster_table() -> str:
    """The roster as the prompt shows it: one aligned row per person."""
    rows = [("User ID", "Person", "Role", "Team Lead")]
    rows += [(p.user_id, p.name, p.role, p.team_lead or "-") for p in ROSTER]
    # At least the widths the table was first written with, so it reads the same.
    widths = [
        max(minimum, *(len(row[i]) for row in rows))
        for i, minimum in enumerate((14, 18, 13))
    ]
    return "\n".join(
        "  "
        + "   ".join(cell.ljust(widths[i]) for i, cell in enumerate(row[:3]))
        + "   "
        + row[3]
        for row in rows
    )
