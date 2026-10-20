import http.client
import urllib.parse

import chart_viewer
import pytest


@pytest.fixture(autouse=True)
def fresh_server():
    yield
    chart_viewer.shutdown()


def _get(url, host=None):
    parts = urllib.parse.urlsplit(url)
    conn = http.client.HTTPConnection(parts.hostname, parts.port, timeout=5)
    conn.request("GET", parts.path, headers={"Host": host} if host else {})
    response = conn.getresponse()
    return response.status, dict(response.getheaders()), response.read()


def test_chart_is_served_sandboxed(tmp_path):
    chart = tmp_path / "lda_model.html"
    chart.write_text("<html><script>1</script></html>")
    url = chart_viewer.register(str(chart))
    assert url.startswith("http://127.0.0.1:")

    status, headers, body = _get(url)
    assert status == 200 and body == chart.read_bytes()
    csp = headers["Content-Security-Policy"]
    assert csp.startswith("sandbox allow-scripts;")
    assert (
        "allow-same-origin" not in csp
        and "allow-popups" not in csp
        and "allow-top-navigation" not in csp
    )
    for directive in (
        "default-src 'none'",
        "connect-src 'none'",
        "object-src 'none'",
        "base-uri 'none'",
        "form-action 'none'",
        "frame-ancestors 'none'",
    ):
        assert directive in csp
    assert (
        f"script-src 'unsafe-inline' {chart_viewer.D3_URL} {chart_viewer.LDAVIS_URL}"
        in csp
    )
    assert headers["X-Content-Type-Options"] == "nosniff"


def test_only_registered_urls_and_local_hosts_are_served(tmp_path):
    chart = tmp_path / "c.html"
    chart.write_text("x")
    url = chart_viewer.register(str(chart))
    base = url.rsplit("/", 2)[0]
    assert _get(base + "/guess/chart.html")[0] == 404
    assert _get(base + "/")[0] == 404
    # DNS rebinding: a page on another site can't read charts through its own host name.
    assert _get(url, host="evil.example")[0] == 404


def test_each_chart_gets_an_unguessable_url(tmp_path):
    chart = tmp_path / "c.html"
    chart.write_text("x")
    first, second = chart_viewer.register(str(chart)), chart_viewer.register(str(chart))
    assert first != second
    assert len(first.split("/")[3]) >= 32


def test_bytes_are_snapshotted_when_registered(tmp_path):
    chart = tmp_path / "c.html"
    chart.write_text("original")
    url = chart_viewer.register(str(chart))
    chart.write_text("<script>swapped</script>")
    assert _get(url)[2] == b"original"


def test_huge_charts_are_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(chart_viewer, "MAX_CHART_BYTES", 10)
    chart = tmp_path / "c.html"
    chart.write_text("x" * 11)
    with pytest.raises(OSError, match="over"):
        chart_viewer.register(str(chart))


def test_server_listens_only_on_loopback(tmp_path):
    chart = tmp_path / "c.html"
    chart.write_text("x")
    chart_viewer.register(str(chart))
    assert chart_viewer.address()[0] == "127.0.0.1"
