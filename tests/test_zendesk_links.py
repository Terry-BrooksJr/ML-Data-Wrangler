from ui import zendesk_links


def test_ticket_url_takes_ticket_numbers_only(monkeypatch):
    assert (
        zendesk_links.ticket_url("30882")
        == "https://learnositysupport.zendesk.com/agent/tickets/30882"
    )
    assert zendesk_links.ticket_url(30882).endswith("/tickets/30882")
    assert zendesk_links.ticket_url("12/../admin") is None
    assert zendesk_links.ticket_url(None) is None

    opened = []
    monkeypatch.setattr(zendesk_links.webbrowser, "open_new", opened.append)
    assert zendesk_links.open_ticket(" 42 ") and opened[-1].endswith("/tickets/42")
    assert not zendesk_links.open_ticket("x")
