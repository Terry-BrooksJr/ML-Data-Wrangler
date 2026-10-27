import json

import pytest
from core import wrangler
from test_wrangler_mongo import COMMENTS, TICKET
from core.wrangler import DataWrangler, process_progress


@pytest.mark.parametrize(
    "stage, done, total, expected",
    [
        ("reshape", 0, 4, 0),
        ("reshape", 2, 4, 12),
        ("reshape", 4, 4, 25),
        ("bind", 1, 4, 36),
        ("bind", 4, 4, 70),
        ("corpus", 4, 4, 95),
        ("corpus", 9, 4, 95),
        ("bind", 0, 0, 70),  # no tickets: the stage is simply done
    ],
)
def test_process_progress(stage, done, total, expected):
    assert process_progress(stage, done, total) == expected


def test_progress_only_moves_forward():
    values = [
        process_progress(stage, done, 3)
        for stage in ("reshape", "bind", "corpus")
        for done in range(4)
    ]
    assert (
        values == sorted(values)
        and values[0] == 0
        and values[-1] == wrangler.PROCESS_STAGES["corpus"][1]
    )


@pytest.fixture
def three_tickets(tmp_path):
    (tmp_path / "comments").mkdir()
    tickets = [{**TICKET, "id": 101 + i} for i in range(3)]
    (tmp_path / "tickets-only.json").write_text(json.dumps(tickets))
    (tmp_path / "comments" / "101.json").write_text(json.dumps(COMMENTS))
    return DataWrangler(
        comments_dir=tmp_path / "comments", ticket_file=tmp_path / "tickets-only.json"
    )


def test_every_step_reports_per_ticket_progress(three_tickets):
    events = []
    three_tickets.tickets_loaded.connect(lambda n: events.append(("loaded", n)))
    three_tickets.ticket_progress.connect(lambda n: events.append(("reshape", n)))
    three_tickets.binding_progress.connect(lambda n: events.append(("bind", n)))
    three_tickets.corpus_creation_progress.connect(
        lambda n: events.append(("corpus", n))
    )

    assert three_tickets.tickets_reshaped()
    assert three_tickets.comments_bound()
    assert three_tickets.create_corpus()

    assert events == [("loaded", 3)] + [
        (step, n) for step in ("reshape", "bind", "corpus") for n in (1, 2, 3)
    ]
