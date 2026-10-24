"""Asks for the spending PIN before a paid job (see spend_guard)."""

from typing import Iterable, Optional

import spend_guard
from loguru import logger
from PyQt5.QtWidgets import QInputDialog, QLineEdit, QMessageBox, QWidget


def ask_for_grant(
    parent: QWidget, operation: str, services: Iterable[str]
) -> Optional[spend_guard.Grant]:
    """A grant for operation's paid calls, or None if refused or cancelled.

    Checks the login's role first, then asks for the PIN until it is right,
    cancelled, or the wrong tries lock paid calls. A module function so tests
    can answer it without a dialog.
    """
    services = frozenset(services)
    try:
        spend_guard.require_permission(
            spend_guard.SPEND, f"start paid calls ({operation})"
        )
        path = spend_guard.ensure_pin()
    except spend_guard.SecurityError as e:
        QMessageBox.warning(parent, "Paid Calls Not Allowed", str(e))
        return None
    note = ""
    while True:
        pin, ok = QInputDialog.getText(
            parent,
            "Spending PIN",
            f"{operation} makes paid requests to {spend_guard.describe(services)}.\n\n"
            f"Enter the spending PIN. It's in this file, readable only by you:\n{path}"
            + (f"\n\n{note}" if note else ""),
            QLineEdit.Password,
        )
        if not ok:
            return None
        try:
            return spend_guard.authorize(pin, operation, services)
        except spend_guard.WrongPin as e:
            note = str(e)
        except spend_guard.SecurityError as e:
            logger.warning(f"Paid calls refused: {e}")
            QMessageBox.warning(parent, "Paid Calls Not Allowed", str(e))
            return None


def allowed(parent: QWidget, permission: str, action: str) -> bool:
    """True if this login's role allows action; otherwise says why and returns False."""
    try:
        spend_guard.require_permission(permission, action)
        return True
    except spend_guard.PermissionDenied as e:
        QMessageBox.warning(parent, "Not Allowed", str(e))
        return False
