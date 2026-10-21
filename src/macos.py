"""The name macOS shows for the app when it runs from source.

The bold app name in the macOS menu bar comes from the main bundle's
CFBundleName, not from QApplication.setApplicationName. Run from source, the
main bundle is the python executable's folder, so the menu says "python".
Setting CFBundleName before the QApplication exists makes it say the app's
name. The packaged Lasso.app has its own Info.plist and doesn't need this.

Uses the Objective-C runtime through ctypes, so it needs no PyObjC.
"""

import ctypes
import ctypes.util
import sys

from loguru import logger


def _runtime():
    objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
    ctypes.cdll.LoadLibrary(ctypes.util.find_library("Foundation"))
    objc.objc_getClass.restype = ctypes.c_void_p
    objc.objc_getClass.argtypes = [ctypes.c_char_p]
    objc.sel_registerName.restype = ctypes.c_void_p
    objc.sel_registerName.argtypes = [ctypes.c_char_p]
    address = ctypes.cast(objc.objc_msgSend, ctypes.c_void_p).value

    def send(restype, *argtypes):
        # objc_msgSend must be called with the method's exact signature (arm64).
        return ctypes.CFUNCTYPE(restype, ctypes.c_void_p, ctypes.c_void_p, *argtypes)(
            address
        )

    return objc.objc_getClass, objc.sel_registerName, send


def set_app_name(name: str) -> bool:
    """Makes the macOS menu bar show name instead of "python". Call before creating the QApplication.

    Returns:
        True if the name was set; False off macOS or if it couldn't be (never raises).
    """
    if sys.platform != "darwin":
        return False
    try:
        cls, sel, send = _runtime()
        bundle = send(ctypes.c_void_p)(cls(b"NSBundle"), sel(b"mainBundle"))
        info = send(ctypes.c_void_p)(bundle, sel(b"infoDictionary"))
        if not info or not send(ctypes.c_bool, ctypes.c_void_p)(
            info, sel(b"respondsToSelector:"), sel(b"setObject:forKey:")
        ):
            return False
        string = send(ctypes.c_void_p, ctypes.c_char_p)
        value = string(
            cls(b"NSString"), sel(b"stringWithUTF8String:"), name.encode("utf-8")
        )
        key = string(cls(b"NSString"), sel(b"stringWithUTF8String:"), b"CFBundleName")
        send(None, ctypes.c_void_p, ctypes.c_void_p)(
            info, sel(b"setObject:forKey:"), value, key
        )
        return True
    except (OSError, AttributeError, TypeError) as e:
        logger.debug(f"Couldn't set the macOS app name: {e}")
        return False
