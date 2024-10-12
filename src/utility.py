from loguru import logger


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
