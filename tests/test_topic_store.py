import json
import pathlib

import mongomock
import pytest
from gensim.corpora.dictionary import Dictionary

import topic_store
from LDA_logic import LatentDirichletAllocator
from test_wrangler_mongo import data_dir, wrangler  # noqa: F401 (fixtures)

TOKENS = [
    ["import", "fail", "csv", "upload", "error"],
    ["import", "csv", "error", "row", "fail"],
    ["login", "password", "reset", "sso", "account"],
    ["password", "login", "sso", "locked", "account"],
    ["report", "score", "student", "export", "grade"],
    ["score", "report", "grade", "student", "missing"],
] * 3
IDS = [str(1000 + i) for i in range(len(TOKENS))]


@pytest.fixture
def trained(monkeypatch):
    allocator = LatentDirichletAllocator(num_of_topics=3, seed=7)
    allocator._tokens = TOKENS
    allocator.prelemma_corpus = [" ".join(t) for t in TOKENS]
    allocator.document_ids = IDS
    allocator.id2word = Dictionary(TOKENS)
    allocator.corpus = [allocator.id2word.doc2bow(t) for t in TOKENS]
    assert allocator.model_trained(iterations=20, passes=3, num_of_topics=3)

    def fake_charts(output_dir=None):
        chart, plot = pathlib.Path(output_dir) / "lda_model.html", pathlib.Path(output_dir) / "coherence.png"
        chart.write_text("<html></html>")
        plot.write_bytes(b"png")
        return str(chart), str(plot)

    monkeypatch.setattr(allocator, "visualize_results", fake_charts)
    return allocator


class FakeNlp:
    meta = {"lang": "en", "name": "core_web_lg", "version": "3.7.1"}


@pytest.fixture
def db():
    return mongomock.MongoClient().support_db


def test_save_run_writes_model_dictionary_and_details(trained, tmp_path):
    run = topic_store.save_run(trained, FakeNlp(), ["the", "a"], base_dir=tmp_path)

    run_dir = pathlib.Path(run["model_dir"])
    assert run_dir.parent == tmp_path
    for name in (topic_store.MODEL_FILE, topic_store.DICTIONARY_FILE, topic_store.RUN_FILE,
                 "lda_model.html", "coherence.png"):
        assert (run_dir / name).exists(), name
    assert run["settings"]["iterations"] == 20 and run["settings"]["seed"] == 7
    assert run["preprocessing"]["spacy_model"] == "en_core_web_lg"
    assert run["preprocessing"]["stopwords_count"] == 2
    assert [c["num_topics"] for c in run["coherence"]] == [2, 3]
    assert run["dataset"]["ticket_count"] == len(TOKENS)
    assert run["topics"] and run["best_num_topics"] in (2, 3)


def test_charts_are_per_run(trained, tmp_path):
    first = topic_store.save_run(trained, FakeNlp(), [], base_dir=tmp_path)
    second = topic_store.save_run(trained, FakeNlp(), [], base_dir=tmp_path)
    assert first["charts"]["lda_html"] != second["charts"]["lda_html"]


def test_save_run_needs_a_model(tmp_path):
    with pytest.raises(ValueError):
        topic_store.save_run(LatentDirichletAllocator(num_of_topics=2), FakeNlp(), [], base_dir=tmp_path)


def test_saved_model_reloads_and_assigns_without_retraining(trained, tmp_path, db):
    run = topic_store.save_run(trained, FakeNlp(), [], base_dir=tmp_path)
    model, dictionary = topic_store.load_run_model(run)

    bow = dictionary.doc2bow(["password", "login", "sso"])
    # Inference starts from a random point, so compare approximately.
    reloaded, original = model.get_document_topics(bow), trained.lda_model.get_document_topics(bow)
    assert [t for t, _ in reloaded] == [t for t, _ in original]
    assert [p for _, p in reloaded] == pytest.approx([p for _, p in original], abs=1e-3)


def test_run_history_in_mongo_and_on_disk(trained, tmp_path, db):
    runs = db[topic_store.RUNS_COLLECTION]
    first = topic_store.save_run(trained, FakeNlp(), [], base_dir=tmp_path)
    second = topic_store.save_run(trained, FakeNlp(), [], base_dir=tmp_path)
    topic_store.record_run(runs, first)
    topic_store.record_run(runs, second)
    topic_store.record_run(runs, second)  # re-recording replaces

    listed = topic_store.list_runs(runs)
    assert [r["run_id"] for r in listed] == [second["run_id"], first["run_id"]]
    assert listed[0]["coherence"] == second["coherence"]

    on_disk = topic_store.list_runs(base_dir=tmp_path)
    assert {r["run_id"] for r in on_disk} == {first["run_id"], second["run_id"]}
    assert "topics" in topic_store.describe_run(on_disk[0]) or on_disk[0]["best_num_topics"]


def test_list_runs_with_no_directory(tmp_path):
    assert topic_store.list_runs(base_dir=tmp_path / "missing") == []


def test_assign_trained_topics_writes_ticket_topic(trained, tmp_path, db):
    tickets = db.tickets
    tickets.insert_many([{"zendesk_id": zendesk_id} for zendesk_id in IDS[:4]])
    run = topic_store.save_run(trained, FakeNlp(), [], base_dir=tmp_path)

    summary = topic_store.assign_trained_topics(tickets, run, trained)

    assert summary == {"assigned": 4, "empty": 0, "not_in_mongo": len(IDS) - 4}
    topic = tickets.find_one({"zendesk_id": IDS[0]})["topic"]
    assert topic["run_id"] == run["run_id"]
    assert 0 <= topic["main_topic"] < trained.best_num_topics
    assert topic["mix"] and topic["words"]


def test_assign_from_saved_run_reads_tickets_from_mongo(trained, tmp_path, db, monkeypatch):
    tickets = db.tickets
    tickets.insert_many([
        {"zendesk_id": "1", "subject": "SSO login", "transcript": [{"body": "password reset for locked account"}]},
        {"zendesk_id": "2", "subject": "zzz", "transcript": []},
    ])
    run = topic_store.save_run(trained, FakeNlp(), [], base_dir=tmp_path)
    monkeypatch.setattr(topic_store, "tokenize", lambda nlp, stopwords, texts: [t.lower().split() for t in texts])

    summary = topic_store.assign_from_saved_run(tickets, run, FakeNlp(), [])

    assert summary["assigned"] == 1 and summary["empty"] == 1
    assert tickets.find_one({"zendesk_id": "1"})["topic"]["run_id"] == run["run_id"]
    assert "topic" not in tickets.find_one({"zendesk_id": "2"})


def test_topic_survives_reingestion(wrangler):  # noqa: F811
    tickets = mongomock.MongoClient().support_db.tickets
    wrangler.write_ticket_to_mongo(101, tickets)
    tickets.update_one({"zendesk_id": "101"}, {"$set": {"topic": {"run_id": "r1", "main_topic": 2}}})

    wrangler.write_ticket_to_mongo(101, tickets)

    assert tickets.find_one({"zendesk_id": "101"})["topic"] == {"run_id": "r1", "main_topic": 2}


def test_mongo_ticket_document_matches_training_document(wrangler):  # noqa: F811
    wrangler.tickets_reshaped()
    wrangler.comments_bound()
    doc = wrangler.ticket_to_document(101)

    assert wrangler.mongo_ticket_document(doc) == wrangler.ticket_documents()[0]
    assert wrangler.ticket_ids() == ["101"]


def test_dataset_fingerprint_changes_with_content():
    a = topic_store.dataset_fingerprint(["x y"], ["1"])
    assert a == topic_store.dataset_fingerprint(["x y"], ["1"])
    assert a != topic_store.dataset_fingerprint(["x z"], ["1"])


def test_run_file_is_valid_json(trained, tmp_path):
    run = topic_store.save_run(trained, FakeNlp(), [], base_dir=tmp_path)
    data = json.loads((pathlib.Path(run["model_dir"]) / topic_store.RUN_FILE).read_text())
    assert data["run_id"] == run["run_id"]
