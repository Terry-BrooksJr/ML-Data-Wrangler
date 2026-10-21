"""Opens pyLDAvis charts in the browser inside a sandbox.

A chart is HTML with script, and one downloaded from model storage (or
pointed at by a run record) can't be assumed safe. Opening it as a file://
page would run its script with the user's local-file origin. Instead each
chart is served once-registered from 127.0.0.1 at an unguessable URL, with a
Content-Security-Policy that:

- sandboxes it (an opaque origin: no access to local files, cookies or other
  pages' storage; no popups, forms or top-level navigation of other pages);
- allows script and style only inline and from the exact d3 / LDAvis URLs
  pyLDAvis loads, and blocks every network request (connect, images, fonts,
  frames, plugins, base URL changes).

The chart's bytes are read when it is registered, so swapping the file
afterwards changes nothing.
"""

import http.server
import secrets
import threading
import webbrowser
from typing import Dict, Optional, Tuple

from loguru import logger

# The scripts and stylesheet pyLDAvis 3.4 charts load (pyLDAvis.urls).
D3_URL = "https://d3js.org/d3.v5.js"
LDAVIS_URL = (
    "https://cdn.jsdelivr.net/gh/bmabey/pyLDAvis@3.4.0/pyLDAvis/js/ldavis.v3.0.0.js"
)
LDAVIS_CSS_URL = (
    "https://cdn.jsdelivr.net/gh/bmabey/pyLDAvis@3.4.0/pyLDAvis/js/ldavis.v1.0.0.css"
)
MAX_CHART_BYTES = 50 * 1024 * 1024

CSP = "; ".join(
    [
        "sandbox allow-scripts",
        "default-src 'none'",
        f"script-src 'unsafe-inline' {D3_URL} {LDAVIS_URL}",
        f"style-src 'unsafe-inline' {LDAVIS_CSS_URL}",
        "img-src data:",
        "connect-src 'none'",
        "font-src 'none'",
        "frame-src 'none'",
        "object-src 'none'",
        "base-uri 'none'",
        "form-action 'none'",
        "frame-ancestors 'none'",
    ]
)
HEADERS = {
    "Content-Type": "text/html; charset=utf-8",
    "Content-Security-Policy": CSP,
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
    "Cross-Origin-Opener-Policy": "same-origin",
}


class _Handler(http.server.BaseHTTPRequestHandler):
    server: "_ChartServer"

    def do_GET(self) -> None:  # noqa: N802 (http.server's naming)
        host = (self.headers.get("Host") or "").split(":")[0]
        body = (
            self.server.charts.get(self.path)
            if host in ("127.0.0.1", "localhost")
            else None
        )
        if body is None:
            self.send_error(404)
            return
        self.send_response(200)
        for name, value in HEADERS.items():
            self.send_header(name, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(
        self, format, *args
    ) -> None:  # quiet: no request lines in the app log
        pass


class _ChartServer(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.charts: Dict[str, bytes] = {}


_server: Optional[_ChartServer] = None
_lock = threading.Lock()


def _ensure_server() -> _ChartServer:
    global _server
    with _lock:
        if _server is None:
            _server = _ChartServer()
            threading.Thread(
                target=_server.serve_forever, name="chart-viewer", daemon=True
            ).start()
        return _server


def register(chart_path: str) -> str:
    """Serves the chart's current bytes at a new unguessable 127.0.0.1 URL and returns it.

    Raises:
        OSError: the chart can't be read, or is larger than MAX_CHART_BYTES.
    """
    with open(chart_path, "rb") as f:
        body = f.read(MAX_CHART_BYTES + 1)
    if len(body) > MAX_CHART_BYTES:
        raise OSError(
            f"{chart_path} is over {MAX_CHART_BYTES // (1024 * 1024)} MB; not opening it."
        )
    server = _ensure_server()
    path = f"/{secrets.token_urlsafe(24)}/chart.html"
    server.charts[path] = body
    host, port = server.server_address[:2]
    return f"http://{host}:{port}{path}"


def open_chart(chart_path: str) -> str:
    """Opens the chart in the default browser, sandboxed. Returns the URL used."""
    url = register(chart_path)
    logger.info(f"Opening the topic chart (sandboxed) at {url}")
    webbrowser.open(url)
    return url


def shutdown() -> None:
    """Stops the server (on app exit); later charts start a new one."""
    global _server
    with _lock:
        if _server is not None:
            _server.shutdown()
            _server.server_close()
            _server = None


def address() -> Tuple[str, int]:
    return _ensure_server().server_address[:2]
