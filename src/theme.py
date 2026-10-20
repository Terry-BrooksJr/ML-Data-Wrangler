"""The app's look: PyQt-SiliconUI's dark design.

The buttons, combo boxes, checkboxes, progress bars, scroll bars and cards are
SiliconUI widgets (see widgets.py); they paint themselves. The plain Qt widgets
around them (tabs, tables, trees, text boxes, spin boxes, dialogs, menus) get
the stylesheet below, built from SiliconUI's own colour tokens so the two
match. SiliconUI's widgets come only in dark, so the app has no light or
native theme. The log pane is a console in the same palette: LogHighlighter's
colours (yellow, cyan, green) are chosen for a dark background.
"""

import contextlib
import io
import os
import sys

with contextlib.redirect_stdout(io.StringIO()):
    # Importing siui prints a note (in Chinese) and sets QT_SCALE_FACTOR from
    # the Windows DPI. Keep stdout quiet, and keep the variable only where the
    # library means it to apply: on Windows, if the user hadn't set it.
    _user_scale = os.environ.get("QT_SCALE_FACTOR")
    import siui  # noqa: F401  (runs the library's set-up)
    from siui.core import SiColor, SiGlobal

    if _user_scale is not None:
        os.environ["QT_SCALE_FACTOR"] = _user_scale
    elif sys.platform != "win32":
        os.environ.pop("QT_SCALE_FACTOR", None)

from PyQt5.QtGui import QColor, QFont, QPalette


def _token(name: str) -> str:
    return SiGlobal.siui.colors.fromToken(getattr(SiColor, name))


# SiliconUI's dark palette (siui.gui.color_group.DarkColorGroup), by role.
WINDOW = _token("INTERFACE_BG_A")  # behind everything
FIELD = _token("INTERFACE_BG_B")  # text boxes, tables, the log
CARD = _token("INTERFACE_BG_C")  # cards and dialog sections
RAISED = _token("INTERFACE_BG_D")  # headers, hovered rows
BUTTON = _token("BUTTON_PANEL")
TEXT = _token("TEXT_B")
TEXT_DIM = _token("TEXT_D")
TEXT_FAINT = _token("TEXT_E")
ACCENT = _token("THEME")
ACCENT_TEXT = _token("TEXT_THEME")
SELECTION = "#493F4E"  # SiliconUI's line-edit selection colour
WARNING = "#E0A782"
ERROR = "#d36764"  # SiliconUI's line-edit error indicator

FONT_FAMILIES = ["Segoe UI", ".AppleSystemUIFont", "Helvetica Neue", "PingFang SC", "Microsoft YaHei"]
FONT_PIXELS = 14

# The QTextEdit that shows the log; app.py gives it this objectName.
LOG_PANE = "logPane"
LOG_PANE_STYLE = f"""
QTextEdit#{LOG_PANE} {{
    background-color: {FIELD};
    color: {TEXT};
    border: 1px solid {CARD};
    border-radius: 8px;
    padding: 6px;
    font-family: "Menlo", "Consolas", "DejaVu Sans Mono", monospace;
    font-size: 12px;
}}
"""

# A leading dot (.QPushButton) matches that class only, not the SiliconUI
# subclasses, which paint themselves.
STYLESHEET = f"""
QMainWindow, QDialog, QMessageBox, QProgressDialog, QInputDialog {{
    background-color: {WINDOW};
}}
QWidget {{
    color: {TEXT};
    selection-background-color: {SELECTION};
    selection-color: {TEXT};
}}
QLabel {{ background: transparent; }}
QLabel[role="heading"] {{ color: {TEXT}; font-size: 16px; font-weight: 600; }}
QLabel[role="hint"] {{ color: {TEXT_DIM}; }}
QLabel[role="warning"] {{ color: {WARNING}; }}
QToolTip {{
    background-color: {BUTTON};
    color: {TEXT};
    border: none;
    border-radius: 6px;
    padding: 4px 8px;
}}

QMenuBar {{ background-color: {WINDOW}; }}
QMenuBar::item:selected {{ background-color: {CARD}; border-radius: 4px; }}
QMenu {{
    background-color: {CARD};
    border: 1px solid {RAISED};
    border-radius: 8px;
    padding: 4px;
}}
QMenu::item {{ padding: 6px 20px; border-radius: 4px; }}
QMenu::item:selected {{ background-color: {RAISED}; }}

QTabWidget::pane {{ border: none; }}
QTabBar {{ qproperty-drawBase: 0; }}
QTabBar::tab {{
    background: transparent;
    color: {TEXT_DIM};
    padding: 8px 16px;
    margin-right: 4px;
    border-bottom: 3px solid transparent;
}}
QTabBar::tab:hover {{ color: {TEXT}; }}
QTabBar::tab:selected {{ color: {TEXT}; border-bottom-color: {ACCENT_TEXT}; }}

QGroupBox {{
    background-color: {CARD};
    border: none;
    border-radius: 8px;
    margin-top: 28px;
    padding: 12px 10px 10px 10px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 4px;
    padding: 0 0 6px 0;
    color: {TEXT};
    font-size: 15px;
    font-weight: 600;
}}

.QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QDateEdit {{
    background-color: {FIELD};
    color: {TEXT};
    border: 1px solid transparent;
    border-radius: 8px;
    padding: 5px 8px;
}}
.QLineEdit:focus, QPlainTextEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QDateEdit:focus {{
    border-color: {ACCENT};
}}
.QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{ color: {TEXT_FAINT}; }}
QCalendarWidget QWidget {{ background-color: {CARD}; alternate-background-color: {FIELD}; }}
QCalendarWidget QToolButton {{ background: transparent; color: {TEXT}; }}

QTableView, QTreeView, QListView {{
    background-color: {FIELD};
    alternate-background-color: {WINDOW};
    border: none;
    border-radius: 8px;
    gridline-color: {CARD};
    outline: 0;
}}
QTableView::item, QTreeView::item {{ padding: 4px; }}
QTableView::item:selected, QTreeView::item:selected {{ background-color: {SELECTION}; }}
QTreeView::item:hover {{ background-color: {CARD}; }}
QHeaderView {{ background: transparent; }}
QHeaderView::section {{
    background-color: {CARD};
    color: {TEXT_DIM};
    border: none;
    border-right: 1px solid {FIELD};
    padding: 6px 8px;
    font-weight: 600;
}}
QTableCornerButton::section {{ background-color: {CARD}; border: none; }}
QTreeView::indicator, QTableView::indicator {{
    width: 14px; height: 14px;
    border-radius: 4px;
    background-color: {WINDOW};
    border: 1px solid {TEXT_FAINT};
}}
QTreeView::indicator:checked, QTableView::indicator:checked {{
    background-color: #D087DF;
    border-color: #D087DF;
}}

QAbstractSpinBox > .QLineEdit, QDateEdit > .QLineEdit {{ background: transparent; border: none; padding: 0; }}

QSplitter::handle {{ background: transparent; height: 8px; }}

.QPushButton {{
    background-color: {BUTTON};
    color: {TEXT};
    border: none;
    border-bottom: 3px solid {CARD};
    border-radius: 8px;
    padding: 6px 16px;
    min-width: 64px;
}}
.QPushButton:hover {{ background-color: {RAISED}; }}
.QPushButton:default {{ background-color: {ACCENT}; border-bottom-color: #52389a; }}
.QPushButton:disabled {{ color: {TEXT_FAINT}; }}

.QProgressBar {{
    background-color: {FIELD};
    border: none;
    border-radius: 3px;
    max-height: 6px;
    color: transparent;
}}
.QProgressBar::chunk {{ background-color: {ACCENT_TEXT}; border-radius: 3px; }}

.QScrollBar {{ background: transparent; border: none; }}
.QScrollBar:vertical {{ width: 10px; }}
.QScrollBar:horizontal {{ height: 10px; }}
.QScrollBar::handle {{ background-color: #50FFFFFF; border-radius: 4px; margin: 2px; }}
.QScrollBar::handle:vertical {{ min-height: 24px; }}
.QScrollBar::handle:horizontal {{ min-width: 24px; }}
.QScrollBar::add-line, .QScrollBar::sub-line {{ width: 0; height: 0; }}
.QScrollBar::add-page, .QScrollBar::sub-page {{ background: transparent; }}
"""


def font() -> QFont:
    """The app font: SiliconUI's family list and size."""
    app_font = QFont()
    app_font.setFamilies(FONT_FAMILIES)
    app_font.setPixelSize(FONT_PIXELS)
    return app_font


def palette() -> QPalette:
    """The same colours as a QPalette, for whatever the stylesheet doesn't reach
    (scroll-area viewports, item views' defaults)."""
    colors = QPalette()
    for role, color in [
        (QPalette.Window, WINDOW),
        (QPalette.WindowText, TEXT),
        (QPalette.Base, FIELD),
        (QPalette.AlternateBase, WINDOW),
        (QPalette.Text, TEXT),
        (QPalette.PlaceholderText, TEXT_FAINT),
        (QPalette.Button, BUTTON),
        (QPalette.ButtonText, TEXT),
        (QPalette.Highlight, SELECTION),
        (QPalette.HighlightedText, TEXT),
        (QPalette.ToolTipBase, BUTTON),
        (QPalette.ToolTipText, TEXT),
        (QPalette.Link, ACCENT_TEXT),
    ]:
        colors.setColor(role, QColor(color))
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        colors.setColor(QPalette.Disabled, role, QColor(TEXT_FAINT))
    return colors


def stylesheet() -> str:
    """The application stylesheet."""
    return STYLESHEET + LOG_PANE_STYLE


def apply(app) -> None:
    """Styles app (font, palette and stylesheet), so every window and dialog follows it.

    Call it before building widgets: SiliconUI widgets read the app font when
    they are created.
    """
    app.setFont(font())
    app.setPalette(palette())
    app.setStyleSheet(stylesheet())
