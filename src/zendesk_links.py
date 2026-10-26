"""Links from the app to a ticket in Zendesk's agent workspace.

The View on Zendesk buttons open the ticket in the default browser, in a new
window where the browser allows it (webbrowser.open_new; some browsers open a
new tab instead). Zendesk asks the user to sign in as usual.
"""

import webbrowser
from typing import Optional

TICKET_URL = "https://learnositysupport.zendesk.com/agent/tickets/{}"
BUTTON_TEXT = "View on Zendesk"
BUTTON_TIP = "Open the ticket in Zendesk, in a new browser window."


def ticket_url(zendesk_id) -> Optional[str]:
    """The agent-workspace URL for a ticket, or None if the ID isn't a ticket
    number (only digits go into the URL)."""
    text = str(zendesk_id or "").strip()
    return TICKET_URL.format(text) if text.isdigit() else None


def open_ticket(zendesk_id) -> bool:
    """Opens the ticket in the browser; False if the ID isn't a ticket number."""
    url = ticket_url(zendesk_id)
    if url is None:
        return False
    webbrowser.open_new(url)
    return True
