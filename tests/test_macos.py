import subprocess
import sys
import textwrap
from pathlib import Path

import macos
import pytest


def test_does_nothing_off_macos(monkeypatch):
    monkeypatch.setattr(macos.sys, "platform", "linux")
    assert macos.set_app_name("Lasso") is False


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS only")
def test_sets_the_bundle_name_macos_shows():
    # In a child process: the main bundle's name is process-wide.
    script = textwrap.dedent("""
        import ctypes, macos
        assert macos.set_app_name("Lasso")
        cls, sel, send = macos._runtime()
        bundle = send(ctypes.c_void_p)(cls(b"NSBundle"), sel(b"mainBundle"))
        string = send(ctypes.c_void_p, ctypes.c_char_p)
        key = string(cls(b"NSString"), sel(b"stringWithUTF8String:"), b"CFBundleName")
        value = send(ctypes.c_void_p, ctypes.c_void_p)(bundle, sel(b"objectForInfoDictionaryKey:"), key)
        print(ctypes.string_at(send(ctypes.c_void_p)(value, sel(b"UTF8String"))).decode())
    """)
    src = Path(__file__).resolve().parents[1] / "src"
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=src,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip() == "Lasso"
