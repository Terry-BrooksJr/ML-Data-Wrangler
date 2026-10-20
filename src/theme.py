"""The app's look: QDarkStyleSheet's dark or light theme, or the platform's own.

Picked with the LASSO_THEME setting (Settings > Appearance) and applied to
the whole QApplication, so every window and dialog follows it. The log pane
always stays a dark console whatever the theme: LogHighlighter's colours
(yellow, cyan, green) are only readable on a dark background.
"""

import os
from typing import Optional

from loguru import logger

THEMES = ("dark", "light", "native")
DEFAULT_THEME = "dark"
# The QTextEdit that shows the log; app.py gives it this objectName.
LOG_PANE = "logPane"
LOG_PANE_STYLE = f"""
QTextEdit#{LOG_PANE} {{
    background-color: #19232D;
    color: #E0E1E3;
    font-family: "Menlo", "Consolas", "DejaVu Sans Mono", monospace;
    font-size: 12px;
}}
"""


def theme_name(value: Optional[str]) -> Optional[str]:
    """value as a theme name (any case; blank is the default), or None if it isn't one."""
    value = (value or "").strip().lower() or DEFAULT_THEME
    return value if value in THEMES else None


def stylesheet(name: str) -> str:
    """The application stylesheet for a theme name from THEMES."""
    if name == "native":
        return LOG_PANE_STYLE
    import qdarkstyle
    from qdarkstyle.dark.palette import DarkPalette
    from qdarkstyle.light.palette import LightPalette

    palette = DarkPalette if name == "dark" else LightPalette
    return qdarkstyle.load_stylesheet(qt_api="pyqt5", palette=palette) + LOG_PANE_STYLE


def apply(app, value: Optional[str] = None) -> str:
    """Styles app with the theme in value (default: LASSO_THEME); returns the theme used.

    An unknown name falls back to the default, with a warning, rather than
    leaving the app unstyled.
    """
    value = os.environ.get("LASSO_THEME") if value is None else value
    name = theme_name(value)
    if name is None:
        logger.warning(
            f"Unknown theme {value!r}; using {DEFAULT_THEME}. Choose one of: {', '.join(THEMES)}."
        )
        name = DEFAULT_THEME
    app.setStyleSheet(stylesheet(name))
    return name
