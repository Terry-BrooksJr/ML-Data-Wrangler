"""The "Opening Lasso…" window shown while the app starts.

app.py imports spaCy, BERTopic and the en_core_web_lg model before it can
build the main window, which takes a while with nothing on screen. show()
puts this up first, before those imports, and message() updates its status
line between them.

The artwork is splash.png beside the icon (packaging/, or assets/ in the
packaged build): 1920x1200, shown at 480x300 with its corners rounded, and the
status line drawn over its bottom band. Without it, the splash is drawn here
from the icon.

Imports only PyQt5, so it appears straight away. That rules out ui.theme
(importing SiliconUI is itself slow): the colours below are copies of its
WINDOW, CARD, TEXT, TEXT_DIM and ACCENT_TEXT.
"""

import pathlib
from typing import Optional

from PyQt5.QtCore import QRectF, Qt
from PyQt5.QtGui import QColor, QFont, QGuiApplication, QPainter, QPainterPath, QPen, QPixmap
from PyQt5.QtWidgets import QApplication, QSplashScreen, QWidget

WINDOW = "#1C191F"
CARD = "#332E38"
TEXT = "#DFDFDF"
TEXT_DIM = "#AFAFAF"
ACCENT_TEXT = "#c58bc2"

FONT_FAMILIES = ["Segoe UI", ".AppleSystemUIFont", "Helvetica Neue"]
WIDTH, HEIGHT = 480, 300
ICON_SIZE = 128
# The status line's area, which message() repaints.
STATUS_RECT = QRectF(0, 240, WIDTH, 30)

_splash: Optional[QSplashScreen] = None


def _font(pixels: int, weight: int = QFont.Normal) -> QFont:
    font = QFont()
    font.setFamilies(FONT_FAMILIES)
    font.setPixelSize(pixels)
    font.setWeight(weight)
    return font


def _pixmap(icon_path: str, scale: float) -> QPixmap:
    """The splash on a rounded card, drawn at the screen's scale so it's sharp on Retina."""
    pixmap = QPixmap(int(WIDTH * scale), int(HEIGHT * scale))
    pixmap.setDevicePixelRatio(scale)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)

    card = QPainterPath()
    card.addRoundedRect(QRectF(0.5, 0.5, WIDTH - 1, HEIGHT - 1), 14, 14)
    painter.fillPath(card, QColor(WINDOW))
    artwork = QPixmap(str(pathlib.Path(icon_path).with_name("splash.png")))
    if not artwork.isNull():
        painter.setClipPath(card)
        painter.drawPixmap(QRectF(0, 0, WIDTH, HEIGHT), artwork, QRectF(artwork.rect()))
        painter.setClipping(False)
    painter.setPen(QPen(QColor(CARD), 1))
    painter.drawPath(card)
    if not artwork.isNull():
        painter.end()
        return pixmap

    icon = QPixmap(icon_path)
    if not icon.isNull():
        size = int(ICON_SIZE * scale)
        icon = icon.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        icon.setDevicePixelRatio(scale)
        painter.drawPixmap(int((WIDTH - ICON_SIZE) / 2), 40, icon)

    painter.setPen(QColor(TEXT))
    painter.setFont(_font(28, QFont.DemiBold))
    painter.drawText(QRectF(0, 178, WIDTH, 40), Qt.AlignCenter, "Lasso")
    painter.setPen(QColor(ACCENT_TEXT))
    painter.setFont(_font(13))
    painter.drawText(QRectF(0, 214, WIDTH, 22), Qt.AlignCenter, "Support Ticket Triage")
    painter.end()
    return pixmap


class _Splash(QSplashScreen):
    def drawContents(self, painter: QPainter) -> None:
        # Our own status line rather than QSplashScreen's, which draws in the
        # app font, not yet set here.
        painter.setRenderHint(QPainter.TextAntialiasing)
        painter.setPen(QColor(TEXT_DIM))
        painter.setFont(_font(13))
        painter.drawText(STATUS_RECT, Qt.AlignCenter, self.message())


def show(app: QApplication, icon_path: str) -> None:
    """Shows the splash with "Opening Lasso…". Call right after creating the QApplication."""
    global _splash
    screen = QGuiApplication.primaryScreen()
    scale = screen.devicePixelRatio() if screen else 1.0
    _splash = _Splash(_pixmap(icon_path, scale))
    # Rounded corners: let the transparent pixmap corners show through.
    _splash.setAttribute(Qt.WA_TranslucentBackground)
    _splash.show()
    message("Opening Lasso…")


def message(text: str) -> None:
    """Sets the status line, if the splash is up. Paints at once: the slow imports keep the event loop from running."""
    if _splash is None:
        return
    _splash.showMessage(text)
    QApplication.processEvents()


def finish(window: QWidget) -> None:
    """Closes the splash once window is showing."""
    global _splash
    if _splash is not None:
        _splash.finish(window)
        _splash = None
