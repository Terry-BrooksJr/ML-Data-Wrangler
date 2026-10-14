import mongomock
import pytest
from pymongo.errors import OperationFailure

from decision_engine import search


class FakeSearchCollection:
    """Stands in for a tickets collection on Atlas: mongomock has no search indexes."""

    def __init__(self, indexes=None, results=None, unsupported=False):
        self.indexes = list(indexes or [])
        self.created = []
        self.pipelines = []
        self.results = results or []
        self.unsupported = unsupported

    def list_search_indexes(self):
        if self.unsupported:
            raise OperationFailure("Search index commands are only supported with Atlas", code=115)
        return iter(self.indexes)

    def create_search_index(self, model):
        self.created.append(model.document)
        self.indexes.append({"name": model.document["name"], "queryable": False, "status": "BUILDING"})
        return model.document["name"]

    def aggregate(self, pipeline):
        if self.unsupported:
            raise OperationFailure("Unrecognized pipeline stage name: '$search'", code=40324)
        self.pipelines.append(pipeline)
        return iter(self.results)


def test_ensure_creates_text_index():
    tickets = FakeSearchCollection()
    assert search.ensure_search_indexes(tickets) == [search.TEXT_INDEX]

    created = tickets.created[0]
    assert created["name"] == "tickets_text" and created["type"] == "search"
    fields = created["definition"]["mappings"]["fields"]
    assert fields["subject"]["analyzer"] == "lucene.english"
    assert fields["transcript"]["fields"]["body"]["type"] == "string"
    assert fields["status"]["type"] == "token"
    assert created["definition"]["mappings"]["dynamic"] is False


def test_ensure_is_idempotent():
    tickets = FakeSearchCollection(indexes=[{"name": search.TEXT_INDEX, "queryable": True}])
    assert search.ensure_search_indexes(tickets) == []
    assert tickets.created == []


def test_ensure_on_plain_mongod_explains():
    with pytest.raises(search.SearchUnavailable, match="mongodb-atlas-local"):
        search.ensure_search_indexes(FakeSearchCollection(unsupported=True))


def test_wait_until_queryable_polls(monkeypatch):
    tickets = FakeSearchCollection(indexes=[{"name": search.TEXT_INDEX, "queryable": False}])
    sleeps = []

    def build_finishes(seconds):
        sleeps.append(seconds)
        tickets.indexes[0]["queryable"] = True

    monkeypatch.setattr(search.time, "sleep", build_finishes)
    search.wait_until_queryable(tickets, timeout=60, poll_seconds=1)
    assert sleeps == [1]


def test_wait_until_queryable_times_out(monkeypatch):
    tickets = FakeSearchCollection(indexes=[{"name": search.TEXT_INDEX, "queryable": False, "status": "BUILDING"}])
    monkeypatch.setattr(search.time, "sleep", lambda s: None)
    with pytest.raises(TimeoutError, match="BUILDING"):
        search.wait_until_queryable(tickets, timeout=0)


def test_pipeline_without_status():
    pipeline = search.build_search_pipeline("import fails", limit=5)

    stage = pipeline[0]["$search"]
    assert list(pipeline[0]) == ["$search"]  # must be the first stage
    assert stage["index"] == "tickets_text"
    exact, fuzzy = stage["compound"]["should"]
    assert exact["text"]["query"] == "import fails" and exact["text"]["path"] == ["subject", "transcript.body"]
    assert exact["text"]["score"] == {"boost": {"value": search.EXACT_BOOST}}
    assert fuzzy["text"]["fuzzy"] == {"maxEdits": 1, "prefixLength": search.FUZZY_PREFIX_LENGTH}
    assert stage["compound"]["minimumShouldMatch"] == 1
    assert "filter" not in stage["compound"]
    assert pipeline[1] == {"$limit": 5}
    assert pipeline[2]["$project"]["score"] == {"$meta": "searchScore"}


def test_pipeline_with_status_filters_exactly():
    stage = search.build_search_pipeline("sso", status="open")[0]["$search"]
    assert stage["compound"]["should"][0]["text"]["query"] == "sso"
    assert stage["compound"]["filter"] == [{"equals": {"path": "status", "value": "open"}}]


def test_search_tickets_runs_pipeline():
    result = {"zendesk_id": "1", "subject": "Import fails", "score": 3.2}
    tickets = FakeSearchCollection(results=[result])

    assert search.search_tickets(tickets, "  import  ") == [result]
    assert tickets.pipelines[0][0]["$search"]["compound"]["should"][0]["text"]["query"] == "import"


def test_search_tickets_rejects_empty_query():
    with pytest.raises(ValueError):
        search.search_tickets(FakeSearchCollection(), "   ")


def test_search_on_plain_mongod_explains():
    with pytest.raises(search.SearchUnavailable):
        search.search_tickets(FakeSearchCollection(unsupported=True), "import")


def test_search_on_mongomock_reports_unavailable_or_unsupported():
    # mongomock doesn't know $search either; it must not silently return nothing.
    with pytest.raises(Exception):
        search.search_tickets(mongomock.MongoClient().db.tickets, "import")


def test_highlight_text_marks_hits():
    result = {"highlights": [{"texts": [{"value": "The ", "type": "text"}, {"value": "import", "type": "hit"},
                                        {"value": " fails", "type": "text"}]}]}
    assert search.highlight_text(result) == "The [import] fails"


# --- review fixes ------------------------------------------------------------


def test_permission_error_is_not_reported_as_unsupported():
    tickets = FakeSearchCollection()
    tickets.list_search_indexes = lambda: (_ for _ in ()).throw(
        OperationFailure("not authorized on support_db to execute command", code=13))
    with pytest.raises(OperationFailure) as raised:
        search.ensure_search_indexes(tickets)
    assert not isinstance(raised.value, search.SearchUnavailable) and raised.value.code == 13


def test_index_created_concurrently_counts_as_success():
    tickets = FakeSearchCollection()

    def lose_the_race(model):
        tickets.indexes.append({"name": search.TEXT_INDEX, "queryable": False})
        raise OperationFailure("Index already exists: tickets_text", code=68)

    tickets.create_search_index = lose_the_race
    assert search.ensure_search_indexes(tickets) == []


def test_other_create_errors_are_raised():
    tickets = FakeSearchCollection()
    tickets.create_search_index = lambda model: (_ for _ in ()).throw(
        OperationFailure("not authorized", code=13))
    with pytest.raises(OperationFailure):
        search.ensure_search_indexes(tickets)


def test_unsupported_by_code_name():
    tickets = FakeSearchCollection()
    tickets.list_search_indexes = lambda: (_ for _ in ()).throw(
        OperationFailure("search not enabled", code=31082, details={"codeName": "SearchNotEnabled"}))
    with pytest.raises(search.SearchUnavailable):
        search.ensure_search_indexes(tickets)


def test_stalled_listing_raises_timeout_error():
    from pymongo.errors import NetworkTimeout

    tickets = FakeSearchCollection()
    tickets.list_search_indexes = lambda: (_ for _ in ()).throw(NetworkTimeout("timed out"))
    with pytest.raises(TimeoutError, match="timed out"):
        search.wait_until_queryable(tickets, timeout=5)


@pytest.mark.parametrize("limit", [0, -1, 2.5, True, "10"])
def test_invalid_limit_is_rejected(limit):
    tickets = FakeSearchCollection()
    with pytest.raises(ValueError, match="limit"):
        search.search_tickets(tickets, "import", limit=limit)
    assert tickets.pipelines == []


def test_cli_dangling_status_shows_usage(capsys):
    with pytest.raises(SystemExit) as raised:
        search.main(["query", "import", "--status"])
    assert raised.value.code == 2
    assert "expected one argument" in capsys.readouterr().err


def test_terminal_safe_escapes_control_sequences():
    assert search.terminal_safe("ok\x1b]52;c;ZXZpbA==\x07 red\x1b[31m‮evil\nnext") == (
        "ok\\x1b]52;c;ZXZpbA==\\x07 red\\x1b[31m\\u202eevil next")
    assert search.terminal_safe(None) == ""


def test_cli_output_is_terminal_safe(capsys, monkeypatch):
    from decision_engine import persistence

    result = {"zendesk_id": "1", "subject": "Hi\x1b[2J", "status": "open", "score": 1.0,
              "highlights": [{"texts": [{"value": "boom\x1b]0;x\x07", "type": "hit"}]}]}
    monkeypatch.setattr(persistence, "get_tickets_collection", lambda: FakeSearchCollection(results=[result]))

    assert search.main(["query", "boom"]) == 0
    out = capsys.readouterr().out
    assert "\x1b" not in out and "\x07" not in out
    assert "Hi\\x1b[2J" in out
