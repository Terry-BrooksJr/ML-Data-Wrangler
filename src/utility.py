import re
import sys

import validators
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
            format="{time:w} | {level} | {message}",
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
    """
    Highlights log messages in a text editor based on their severity levels.
    This class extends QSyntaxHighlighter to apply different formatting styles to log levels such as INFO, WARNING, ERROR, and DEBUG.

    The `LogHighlighter` class defines specific text formats for each log level and applies these formats to the text
    in a QTextEdit widget. It uses regular expressions to identify log levels and highlight them accordingly.

    Args:
        parent (QWidget, optional): The parent widget for the highlighter. Defaults to None.

    Attributes:
        info_format (QTextCharFormat): Format for INFO log messages.
        warning_format (QTextCharFormat): Format for WARNING log messages.
        error_format (QTextCharFormat): Format for ERROR log messages.
        debug_format (QTextCharFormat): Format for DEBUG log messages.
        highlightingRules (list): A list of tuples containing regex patterns and their corresponding formats.
    """

    def __init__(self, parent=None) -> None:
        """
        Initializes the LogHighlighter with specified formatting for different log levels.
        This constructor sets up the text formats for INFO, WARNING, ERROR, and DEBUG log messages.

        The `__init__` method defines the appearance of each log level by configuring the text color and weight.
        It also establishes the highlighting rules that will be used to format log messages in the text editor.

        Args:
            parent (QWidget, optional): The parent widget for the highlighter. Defaults to None.

        Returns:
            None
        """
        super().__init__(parent)

        # Define log level formats
        self.info_format = QTextCharFormat()
        self.info_format.setForeground(QColor("yellow"))
        self.info_format.setFontWeight(QFont.Weight.Normal)

        self.warning_format = QTextCharFormat()
        self.warning_format.setForeground(QColor("orange"))
        self.warning_format.setFontWeight(QFont.Weight.Bold)

        self.error_format = QTextCharFormat()
        self.error_format.setForeground(QColor("red"))
        self.error_format.setFontWeight(QFont.Weight.Bold)

        self.debug_format = QTextCharFormat()
        self.debug_format.setForeground(QColor("green"))
        self.debug_format.setFontWeight(QFont.Weight.Normal)

        self.time_format = QTextCharFormat()
        self.time_format.setForeground(QColor("green"))
        self.time_format.setFontWeight(QFont.Weight.Thin)

        self.success_format = QTextCharFormat()
        self.success_format.setForeground(QColor("cyan"))
        self.success_format.setFontWeight(QFont.Weight.ExtraBold)
        self.success_format.setFontPointSize(15)

        self.user_input_required_format = QTextCharFormat()
        self.user_input_required_format.setForeground(QColor("blue"))
        self.user_input_required_format.setFontCapitalization(
            QFont.Capitalization.AllUppercase
        )
        self.user_input_required_format.fontUnderline()
        # asserting rules for each log level
        self.highlightingRules = [
            (
                QRegExp(
                    r"[0-9]{4}-[0-9]{2}-[0-9]{2} ([A-Za-z0-9]+(:[A-Za-z0-9]+)+)\.[0-9]+ \| (TRACE|DEBUG|INFO|NOTICE|WARN|WARNING|ERROR|SEVERE|FATAL)\| ([A-Za-z0-9]+( [A-Za-z0-9]+)+)\.\.\.[A-Za-z0-9]+(\s+([A-Za-z]+\s+)+)[A-Za-z0-9]+"
                ),
                self.user_input_required_format,
            ),
            (
                QRegExp(
                    r"[0-9]{4}-[0-9]{2}-[0-9]{2} at [0-9]{2}:[0-9]{2}:+[0-9]{2}(\.[0-9]{1,3})?"
                ),
                self.time_format,
            ),
            (QRegExp(r"\bINFO\b"), self.info_format),
            (QRegExp(r"\bSUCCESS\b"), self.success_format),
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


def remove_useless_data(text: str) -> str:
    """
    Filters out unhelpful data from the input text by removing specific patterns.

    This function takes a string input, splits it into individual words, and removes any words that match certain validation criteria, such as emails, URLs, UUIDs, MD5 hashes, and IPv4 addresses. The remaining words are then combined back into a single string and returned.

    Args:
        text (str): The input string containing text to be processed.

    Returns:
        str: A string with unhelpful data removed.

    Raises:
        None
    """
    scrubbed = [
        word
        for word in text
        if not (
            validators.email(word)
            or validators.url(word)
            or validators.uuid(word)
            or validators.hashes.md5(word)
            or validators.ip_address.ipv4(word)
        )
    ]
    logger.success("Removed Unhelpful and Descriptive Text")
    return " ".join(scrubbed)


5
