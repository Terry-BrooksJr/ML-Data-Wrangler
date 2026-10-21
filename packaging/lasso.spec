# PyInstaller spec for the desktop app. Build with packaging/build.sh, which
# sets up the environment this expects (see the notes there).
#
# One-folder build (plus a .app on macOS) rather than one-file: the spaCy
# model alone is ~600 MB, and a one-file build would unpack it on every launch.

import pathlib
import sys

from PyInstaller.utils.hooks import collect_all, copy_metadata

ROOT = pathlib.Path(SPECPATH).parent
NAME = "Lasso"
# icon.png is the 1024px source; icon.icns (macOS) and icon.ico (Windows) are
# generated from it.
ICON = str(ROOT / "packaging" / ("icon.icns" if sys.platform == "darwin" else "icon.ico"))

# app.py sets icon.png as the window icon (taskbar on Windows/Linux).
datas, binaries, hiddenimports = [(str(ROOT / "packaging" / "icon.png"), "assets")], [], []
# The Tag Normalizer copies these scripts out for a person to run against Zendesk.
datas.append((str(ROOT / "src" / "zendesk_retag"), "zendesk_retag"))

# en_core_web_lg is loaded by import and ships its weights as package data.
# spacy_legacy / spacy_loggers register functions through entry points, which
# spaCy resolves from package metadata at runtime. pyLDAvis ships JS and
# templates its saved HTML charts need. qdarkstyle reads its .qss files and
# imports its compiled icon resources by name.
for package in ("en_core_web_lg", "spacy_legacy", "spacy_loggers", "pyLDAvis", "qdarkstyle"):
    package_datas, package_binaries, package_hiddenimports = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hiddenimports

for dist in ("spacy", "thinc", "en_core_web_lg", "spacy-legacy", "spacy-loggers", "weasel", "confection"):
    datas += copy_metadata(dist)

a = Analysis(
    [str(ROOT / "src" / "app.py")],
    pathex=[str(ROOT / "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["tkinter", "pytest", "mongomock", "IPython", "notebook", "PyQt6", "PySide2", "PySide6"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=NAME,
    console=False,
    icon=ICON,
    # Strip/UPX break code signatures on macOS and gain little here.
    strip=False,
    upx=False,
)

coll = COLLECT(exe, a.binaries, a.datas, name=NAME, strip=False, upx=False)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name=f"{NAME}.app",
        icon=ICON,
        bundle_identifier="com.learnosity.lasso",
        info_plist={
            "CFBundleShortVersionString": "0.1.0",
            "NSHighResolutionCapable": True,
        },
    )
