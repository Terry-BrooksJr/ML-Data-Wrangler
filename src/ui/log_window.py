"""A separate window for the app's log, opened from the Decision Engine tab.

The log pane used to sit below the tabs and took height the review list and
its detail pane needed. It now lives in its own window that keeps collecting
output while hidden, so opening it shows everything logged so far.
"""

from typing import Optional

from PyQt5 import QtCore
from PyQt5.QtWidgets import QTextEdit, QVBoxLayout, QWidget


class LogWindow(QWidget):
    """A top-level window holding the log pane.

    Closing it only hides it: the pane keeps receiving output, and the next
    open_window() shows it again with nothing lost.

    Args:
        log_output: The log pane; this window takes ownership of it.
        parent: The main window, so this one closes when the app does.
    """

    def __init__(self, log_output: QTextEdit, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent, QtCore.Qt.Window)
        self.setWindowTitle("Lasso Logs")
        self.resize(900, 500)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.addWidget(log_output)

    def open_window(self) -> None:
        """Shows the window, or brings it to the front if already open."""
        self.show()
        self.raise_()
        self.activateWindow()
