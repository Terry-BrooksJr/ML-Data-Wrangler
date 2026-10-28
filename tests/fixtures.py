"""Synthetic ticket documents, one per category plus a precedence conflict.

Each fixture is shaped like a mapped Mongo `tickets` document (see
wrangler.DataWrangler.ticket_to_document) and is written so that a correct
reading of the system prompt lands on `expected`. They are synthetic: open item #2 (a real Escalate-to-Support-Owner
edge case) is still outstanding.
"""

from datetime import datetime, timedelta, timezone

NOW = datetime(2026, 10, 5, 12, 0, tzinfo=timezone.utc)


def _ago(days: float) -> datetime:
    return NOW - timedelta(days=days)


def _comment(author, role, body, days_ago, internal=False):
    return {
        "author": author,
        "role": role,
        "body": body,
        "timestamp": _ago(days_ago),
        "is_internal_note": internal,
    }


def _ticket(zendesk_id, subject, transcript, **extra):
    doc = {
        "zendesk_id": zendesk_id,
        "subject": subject,
        "assignee": "Support Owner A",
        "priority": None,
        "status": "open",
        "tags": [],
        "raw_xml": "<Ticket><Id>%s</Id></Ticket>" % zendesk_id,
        "transcript": transcript,
        "reengage_attempts": [],
        "agent_reviews": [],
        "awaiting_review": True,
    }
    doc.update(extra)
    return doc


ESCALATE_LEADERSHIP = _ticket(
    "90001",
    "Student responses visible to another district",
    [
        _comment(
            "Customer",
            "customer",
            "Teachers in our district can see student responses from a different district's sessions. This is a data privacy incident for us.",
            2,
        ),
        _comment(
            "Support Owner A",
            "agent",
            "Confirmed: responses from another tenant are returned by the reports endpoint. Escalating internally.",
            1,
            internal=True,
        ),
    ],
    priority="urgent",
)

ESCALATION_INTERNAL = _ticket(
    "90002",
    "Need a credit for the outage last week",
    [
        _comment(
            "Customer",
            "customer",
            "We lost two days of testing during the outage. Can you issue a service credit on our account?",
            4,
        ),
    ],
)

FLAG_REENGAGE = _ticket(
    "90003",
    "Items fail to render in preview",
    [
        _comment("Customer", "customer", "Some items fail to render in preview.", 12),
        _comment(
            "Support Owner A",
            "agent",
            "Could you send the item reference and the browser console log?",
            10,
        ),
    ],
    status="pending",
)

FLAG_SOLVE = _ticket(
    "90004",
    "How do I reset 2FA for a teacher account?",
    [
        _comment(
            "Customer",
            "customer",
            "A teacher is locked out; please reset their 2FA.",
            3,
        ),
        _comment(
            "Support Owner A", "agent", "Done: 2FA has been reset for that account.", 2
        ),
        _comment(
            "Customer",
            "customer",
            "Confirmed, they're back in. Nothing else needed, thanks!",
            1,
        ),
    ],
)

ON_TRACK = _ticket(
    "90006",
    "Import job slow for large item banks",
    [
        _comment(
            "Customer", "customer", "Our nightly import of 40k items is slow.", 2
        ),
        _comment(
            "Support Owner A",
            "agent",
            "Thanks, could you send the job ID and the time it started?",
            1,
        ),
    ],
    status="pending",
)

MANUAL_TRIAGE = _ticket(
    "90005",
    "help",
    [
        _comment("Customer", "customer", "it doesnt work", 1),
    ],
)

# Satisfies Re-Engage (support's question unanswered for 10 days) AND
# Escalate to Leadership (customer says they are moving to a competitor at
# renewal). Precedence must resolve this to escalate_leadership.
PRECEDENCE_CONFLICT = _ticket(
    "90006",
    "Bulk import failing again",
    [
        _comment(
            "Customer",
            "customer",
            "The bulk import failed again. If this isn't fixed before our renewal next month we will move to another vendor; our contract is under review.",
            14,
        ),
        _comment(
            "Support Owner A",
            "agent",
            "Sorry about that. Can you share the import job ID so we can investigate?",
            10,
        ),
    ],
    priority="high",
)

CATEGORY_FIXTURES = {
    "escalate_leadership": ESCALATE_LEADERSHIP,
    "escalation_internal": ESCALATION_INTERNAL,
    "flag_reengage": FLAG_REENGAGE,
    "flag_solve": FLAG_SOLVE,
    "on_track": ON_TRACK,
    "manual_triage": MANUAL_TRIAGE,
}
