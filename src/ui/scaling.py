"""The Windows display scale, set before the QApplication exists.

Qt reads QT_SCALE_FACTOR once, when the QApplication is created. SiliconUI
sets it from the Windows DPI when it is imported (siui.gui.scale), but app.py
creates the QApplication early, for the splash, before ui.theme imports
SiliconUI. use_windows_scale_factor() does the same set-up first, through
ctypes rather than importing SiliconUI, which is slow.
"""

import ctypes
import os
import sys


def use_windows_scale_factor() -> None:
    """Sets QT_SCALE_FACTOR from the system DPI, as SiliconUI does. Call before creating the QApplication.

    Only on Windows, and only if the user hasn't set QT_SCALE_FACTOR themselves.
    """
    if sys.platform != "win32" or "QT_SCALE_FACTOR" in os.environ:
        return
    try:
        user32 = ctypes.windll.user32
        user32.SetProcessDPIAware()
        factor = user32.GetDpiForSystem() / 96.0
    except Exception:  # SiliconUI falls back to 1 too (GetDpiForSystem needs Windows 10 1607)
        factor = 1
    os.environ["QT_SCALE_FACTOR"] = str(factor)
