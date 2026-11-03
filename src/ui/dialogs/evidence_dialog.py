"""Add Evidence dialog: a Jira export or Aha page for a manual-triage case.

The reviewer picks a file or pastes text; it is parsed here (evidence.extract)
so a file that can't be read is refused before anything is saved, and the
reviewer confirms what was found, since evidence can't be removed once added.
Only the extracted fields leave the dialog, never the file.
"""

import os
from typing import Optional

from decision_engine import evidence
from loguru import logger
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from ui.widgets import Button, hint, use_si_scrollbars

# The kind picker's labels, the evidence kind each picks, and its file filter.
KINDS = {
    "Jira issue export (XML)": (evidence.JIRA, "Jira XML export (*.xml);;All files (*)"),
    "Aha idea or feature (PDF or text)": (
        evidence.AHA,
        "Aha page (*.pdf *.txt);;All files (*)",
    ),
}


def confirm_plain(parent: QWidget, title: str, text: str) -> bool:
    """A Yes/Cancel question shown as plain text: it quotes the export, which
    could otherwise carry rich-text markup that restyles the question."""
    box = QMessageBox(QMessageBox.Question, title, text, QMessageBox.Yes | QMessageBox.Cancel, parent)
    box.setTextFormat(Qt.PlainText)
    box.setDefaultButton(QMessageBox.Cancel)
    return box.exec_() == QMessageBox.Yes


class EvidenceDialog(QDialog):
    """Asks for one piece of evidence and parses it.

    After exec_() returns Accepted, `evidence` holds the parsed
    evidence.Evidence.
    """

    def __init__(self, zendesk_id: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.evidence: Optional[evidence.Evidence] = None
        self._file_content: Optional[bytes] = None
        self._file_name: Optional[str] = None
        self.setWindowTitle(f"Add Evidence: Ticket {zendesk_id}")
        self.resize(640, 480)

        self.kind = QComboBox()
        self.kind.addItems(list(KINDS))
        self.choose_button = Button("Choose File…")
        self.choose_button.clicked.connect(self.choose_file)
        self.file_label = hint("No file chosen.")
        self.file_label.setTextFormat(Qt.PlainText)
        kind_row = QHBoxLayout()
        kind_row.addWidget(QLabel("Evidence"))
        kind_row.addWidget(self.kind, stretch=1)
        kind_row.addWidget(self.choose_button)

        self.paste_box = use_si_scrollbars(QTextEdit())
        self.paste_box.setAcceptRichText(False)
        self.paste_box.setPlaceholderText(
            "…or paste the Jira XML, or the text of the Aha page, here."
        )
        self.paste_box.textChanged.connect(self._pasted)
        # Parse errors can quote the file (an XML tag name, say).
        self.error_label = hint("", role="warning")
        self.error_label.setTextFormat(Qt.PlainText)

        self.add_button = Button("Add Evidence…", primary=True)
        self.add_button.clicked.connect(self.check_and_add)
        cancel = Button("Cancel")
        cancel.clicked.connect(self.reject)
        buttons = QHBoxLayout()
        buttons.addStretch()
        buttons.addWidget(cancel)
        buttons.addWidget(self.add_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(10)
        layout.addWidget(
            hint(
                "Add a Jira XML export, or an Aha idea's History page printed to PDF or "
                "copied as text. Only the fields read from it are saved, not the file. "
                "Evidence can't be edited or removed once added."
            )
        )
        layout.addLayout(kind_row)
        layout.addWidget(self.file_label)
        layout.addWidget(self.paste_box, stretch=1)
        layout.addWidget(self.error_label)
        layout.addLayout(buttons)

    def _kind(self) -> tuple:
        return KINDS[self.kind.currentText()]

    def choose_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose Evidence", "", self._kind()[1]
        )
        if not path:
            return
        if os.path.getsize(path) > evidence.MAX_INPUT_BYTES:
            self.error_label.setText(
                f"The file is over {evidence.MAX_INPUT_BYTES // (1024 * 1024)} MB."
            )
            return
        with open(path, "rb") as f:
            self.load_file(os.path.basename(path), f.read())

    def load_file(self, name: str, content: bytes) -> None:
        """Uses a file's content; it replaces anything pasted."""
        self._file_name, self._file_content = name, content
        self.paste_box.blockSignals(True)
        self.paste_box.clear()
        self.paste_box.blockSignals(False)
        self.file_label.setText(f"File: {name}")
        self.error_label.clear()

    def _pasted(self) -> None:
        if self._file_content is not None and self.paste_box.toPlainText():
            self._file_name = self._file_content = None
            self.file_label.setText("No file chosen.")
        self.error_label.clear()

    def parse(self) -> Optional[evidence.Evidence]:
        """The evidence read from the file or the pasted text, or None with
        the reason shown."""
        kind = self._kind()[0]
        if self._file_content is not None:
            content, source = self._file_content, self._file_name
        else:
            content, source = self.paste_box.toPlainText(), "pasted"
        try:
            return evidence.extract(kind, content, source)
        except evidence.EvidenceError as e:
            self.error_label.setText(f"Can't add this: {e}")
            return None
        except Exception as e:  # a parser bug must not take the dialog down
            logger.exception(f"Couldn't parse {source} as {kind} evidence")
            self.error_label.setText(f"Can't add this: it couldn't be read ({type(e).__name__}).")
            return None

    def check_and_add(self) -> None:
        found = self.parse()
        if found is None:
            return
        if not confirm_plain(
            self,
            "Add Evidence",
            f"Found: {evidence.describe({'kind': found.kind, 'data': found.data})}\n\n"
            "Add it to the case? It can't be removed later.",
        ):
            return
        self.evidence = found
        self.accept()
