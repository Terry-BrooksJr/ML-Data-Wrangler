import sys

from loguru import logger
from PyQt5.QtCore import QRegExp
from PyQt5.QtGui import QColor, QFont, QSyntaxHighlighter, QTextCharFormat


# Custom loguru handler to redirect logs to QTextEdit
class QTextEditLogger:
    """A logger that outputs messages to a QTextEdit widget using Loguru.

    This class integrates the Loguru logging library with a QTextEdit widget,
    allowing log messages to be displayed in a graphical user interface.
    """

    def __init__(self, text_edit_widget):
        """Initializes the QTextEditLogger with a QTextEdit widget.

        Args:
            text_edit_widget: The QTextEdit widget where log messages will be displayed.
        """
        self.text_edit_widget = text_edit_widget
        self._init_loguru()

    def _init_loguru(self):
        """Initializes the Loguru logger with a custom handler.

        This method removes the default Loguru logger configuration and adds
        a custom handler that directs log messages to the QTextEdit widget.
        """
        logger.remove()
        logger.add(
            self._write_to_text_edit,
            format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
            level="INFO",
        )

    def _write_to_text_edit(self, message):
        """Appends a log message to the QTextEdit widget.

        Args:
            message: The log message to be appended to the QTextEdit.
        """
        self.text_edit_widget.append(message)


# Custom class to redirect output to QTextEdit
class QTextEditStream:
    """A stream-like interface for appending text to a QTextEdit widget.

    This class provides a way to write messages directly to a QTextEdit widget,
    allowing for easy integration of text output in a graphical user interface.
    """

    def __init__(self, text_edit_widget):
        """Initializes the QTextEditStream with a QTextEdit widget.

        Args:
            text_edit_widget: The QTextEdit widget to which messages will be appended.
        """
        self.text_edit_widget = text_edit_widget

    def write(self, message):
        """Appends a message to the QTextEdit widget.

        Args:
            message: The message to be appended to the QTextEdit.
        """
        self.text_edit_widget.append(message)

    def flush(self):
        """Flushes the stream.

        This method is a placeholder and does not perform any action,
        as flushing is not necessary for this implementation.
        """
        pass


class LogHighlighter(QSyntaxHighlighter):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Define log level formats
        self.info_format = QTextCharFormat()
        self.info_format.setForeground(QColor("blue"))
        self.info_format.setFontWeight(QFont.Normal)

        self.warning_format = QTextCharFormat()
        self.warning_format.setForeground(QColor("orange"))
        self.warning_format.setFontWeight(QFont.Bold)

        self.error_format = QTextCharFormat()
        self.error_format.setForeground(QColor("red"))
        self.error_format.setFontWeight(QFont.Bold)

        self.debug_format = QTextCharFormat()
        self.debug_format.setForeground(QColor("green"))
        self.debug_format.setFontWeight(QFont.Normal)

        # Define highlighting rules for each log level
        self.highlightingRules = [
            (QRegExp(r"\bINFO\b"), self.info_format),
            (QRegExp(r"\bWARNING\b"), self.warning_format),
            (QRegExp(r"\bERROR\b"), self.error_format),
            (QRegExp(r"\bDEBUG\b"), self.debug_format),
        ]

    def highlightBlock(self, text):
        # Apply highlighting rules to each line of the log
        for pattern, format in self.highlightingRules:
            expression = QRegExp(pattern)
            index = expression.indexIn(text)
            while index >= 0:
                length = expression.matchedLength()
                self.setFormat(index, length, format)
                index = expression.indexIn(text, index + length)
