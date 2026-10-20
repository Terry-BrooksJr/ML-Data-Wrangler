# Lasso (ML-Data-Wrangler)
Data Wrangler and Reshaper for Downstream LDA And Zero-Shot Classification ML FIne Tuning 

The repo holds two separate pieces of work that read the same Zendesk ticket data:

| Component | Location | What it does |
|---|---|---|
| **Data wrangler + LDA** | `src/wrangler.py`, `src/LDA_logic.py`, `src/app.py` | A PyQt5 desktop app. It reshapes Zendesk API exports (`tickets-only.json`, `comments/`) into a cleaned corpus, trains gensim LDA topic models and visualises them. `DataWrangler.ticket_to_xml()` also emits tickets as XML that validates against `ticket_schema.xsd`. |
| **Triage decision engine** | `src/decision_engine/` | An LLM classifier that reads one mapped ticket from Mongo and recommends a single triage action. |

The two are **not** the same concern. LDA is unsupervised, offline topic
modelling. The decision engine is a live classifier against a fixed taxonomy.
No zero-shot or fine-tuning code exists yet. See
[DISCOVERY_NOTES.md](DISCOVERY_NOTES.md) for the full repo survey.

## Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/) and pinned in
`uv.lock`. This installs everything, including the dev tools and the spaCy
`en_core_web_lg` model (about 560 MB, fetched from its GitHub release):

```bash
uv sync
cp .env.example .env   # then fill in values
```

Instead of `.env`, you can enter the same values in the app under
**Settings** (**File > Settings…** on Windows and Linux, **Lasso > Preferences…**
on macOS). They're saved to
`settings.json` in your user config folder (`~/Library/Application
Support/Lasso/` on macOS, `%APPDATA%\Lasso\` on
Windows), readable only by you, and override `.env` and the shell. On launch
the app offers to open Settings if any required value is missing. Most
changes apply immediately; the collection names and `DECISION_ENGINE_MODEL`
need a restart. Blanking a field makes the code's default apply, even if
`.env` or the shell sets it. If the file can't be read, the app renames it to
`settings.json.corrupt-<date>` and asks you to re-enter your settings.

`CHEAPER_INFERENCE_API_URL` must be `https://`, since the API key and ticket
text go there. Plain `http://` is accepted only for `localhost`/loopback, with
a warning; anything else stops the review batch.

`requirements.txt` is exported from the lock for pip users. Regenerate it
after changing dependencies:

```bash
uv export --format requirements-txt --no-dev --no-hashes -o requirements.txt
```

The LDA app also needs the NLTK stopword list, which the package doesn't
include. It downloads on first run; to fetch it up front:

```bash
uv run python -c "import nltk; nltk.download('stopwords')"
```

That fetches from `raw.githubusercontent.com`. If your network breaks SSL to
that host, use the jsDelivr mirror instead:

```bash
mkdir -p ~/nltk_data/corpora && curl -sSfL -o ~/nltk_data/corpora/stopwords.zip https://cdn.jsdelivr.net/gh/nltk/nltk_data@gh-pages/packages/corpora/stopwords.zip && unzip -oq ~/nltk_data/corpora/stopwords.zip -d ~/nltk_data/corpora
```

## Packaging for distribution

`packaging/build.sh` builds a standalone app with PyInstaller, so users need
neither Python nor a `.env`; they enter credentials under Settings.

```bash
packaging/build.sh
```

It syncs a separate environment (`build/pkg-venv`) from `uv.lock` on Python
3.10.21, since PyInstaller refuses the 3.10.0 that `.python-version` pins, and
bundles the NLTK stopwords so the app works offline. Output is
`dist/Lasso.app` on macOS (about 900 MB, mostly the spaCy model)
or `dist/Lasso/` on Windows and Linux. PyInstaller doesn't
cross-compile: build on each OS you ship for.

The packaged app writes its log, wrangled data, charts and saved topic runs
to `~/Documents/Lasso/`. The build is ad-hoc signed only; macOS
Gatekeeper will block it on other machines until it's signed with a
Developer ID and notarized (or users right-click > Open).

## Data wrangler and LDA app

### Running it

Run from the repo root. Output files are written relative to the directory
you launch from.

```bash
uv run python src/app.py
```

1. **Select Path to Ticket File** (`tickets-only.json`) and **Select Path to Comments Directory** (`comments/`).
2. **Process Data** reshapes the tickets, binds each ticket's comments and builds the corpus. It writes `completed/processed_tickets<date>.json` and `completed/corpus_<date>.json`.
3. Enter **Number of Topics**, **Iterations** and **Passes**, then click **Train Model**. All three must be integers: at least 2 topics, fewer than 200 iterations, fewer than 20 passes.

Training runs in a background thread with a progress dialog. It treats each
ticket as one document, then trains a model for every topic count from 2 up
to the number you entered and keeps the one with the highest `c_v`
coherence. When it finishes, it:
- writes `lda_model.html` (an interactive pyLDAvis chart) and `coherence.png`;
- opens a results window with that model's topics, ranked by share of tickets, and the coherence-by-topic-count plot;
- opens `lda_model.html` in the browser. The results window can reopen it.
  Charts are never opened as `file://` pages: the app serves each one from
  `127.0.0.1` at a one-off URL in a browser sandbox (no access to local files,
  cookies or the network beyond the chart's own d3/LDAvis scripts), since a
  chart from a downloaded run is HTML with script.

**Before training**, Train Model looks for a saved run (in MongoDB and this
computer's `topic_models/`) trained on the same tickets, with the same
settings and preprocessing. If there is one, it offers **Use Saved Model**
instead of retraining: the only difference would be the random seed.

### Appearance

The app uses [QDarkStyleSheet](https://github.com/ColinDuquesnoy/QDarkStyleSheet)'s
dark theme by default. Settings > Appearance > Theme switches to `light`, or
`native` for the operating system's own look; the change applies when you
save. The log pane stays a dark console in every theme, so its coloured log
levels stay readable.

### Embedding topics (BERTopic)

**Train BERTopic Model…** groups the tickets in MongoDB by meaning, using the
Voyage embeddings already stored on each ticket (run
`cd src && python -m embeddings backfill` first; tickets whose embedding is
missing or older than their text are left out). It asks for a minimum topic
size (default 1% of tickets, at least 10), a number of topics to merge down to
(default: keep all found), whether to keep outliers as topic -1, an optional
saved LDA run to compare with, and whether to write each ticket's topic to its
`embedding_topic` field (off by default; the LDA `topic` field is never
touched). Training runs in the background; the results window shows the
topics, the `c_v` coherence and topic diversity beside the LDA run's (both
scored on the same tickets), and opens the topic word chart and intertopic
distance map in the browser.

Runs are saved under `bertopic_models/` (`BERTOPIC_MODEL_DIR`), apart from the
LDA runs in `topic_models/`, and kept the same way LDA runs are: recorded in
MongoDB's `bertopic_run_log` (`BERTOPIC_RUNS_COLLECTION`; LDA runs go to
`topic_modeling_run_log`, `TOPIC_RUNS_COLLECTION`) and, with model storage set up, uploaded with a signed manifest to
`s3://<bucket>/<MODEL_PREFIX>bertopic/<run_id>/`. **Show Previous BERTopic
Run…** lists runs from MongoDB and this computer, and downloads and verifies
one whose files aren't here before showing its charts. Retry a failed upload
with `cd src && python -m model_storage upload --bertopic [RUN_ID ...]`. The
same can be run from the command line: `cd src && python -m bertopic_model --help`.

### Saving models to S3

With **Bucket** and **Signing key** set under Settings > Model storage (S3 /
Backblaze B2), each trained run is also uploaded to
`s3://<bucket>/topic-models/<run_id>/`: the model, its dictionary, gensim's
extra files, the charts and `run.json`, then a signed `manifest.json` listing
every file's SHA-256 and size. The location is added to the run in `run.json`
and MongoDB.

- Backblaze B2: set the key ID, secret key and endpoint (e.g.
  `https://s3.us-west-004.backblazeb2.com`); the region is read from it.
  AWS S3: leave the endpoint blank and set the region.
- **Signing key:** use the same key on every computer that shares models
  (`cd src && python -m model_storage new-key` makes one). Model files are
  Python pickles, which run code when loaded, so a downloaded run is only
  opened if its manifest's signature verifies with this key and every file
  matches its listed size and hash. Someone who can edit MongoDB or the bucket
  but doesn't have the key can't get a model opened.
- Runs are only ever loaded from this computer's `topic_models/<run_id>/`,
  never from a path a run record names.
- A failed upload doesn't lose the run: it stays saved locally and the app
  shows a warning. Retry, or upload runs trained before this was set up, with
  `cd src && python -m model_storage upload [RUN_ID ...]` (this also adds the
  run to MongoDB if it isn't there yet).
- A run whose files aren't on this computer (trained elsewhere, or the folder
  was cleared) is downloaded from its upload when you open it with Show
  Previous Run, Assign Topics from Saved Run, or Use Saved Model. Only the
  files the manifest lists are fetched (at most 200 files, 2 GB), into a
  staging folder that replaces the run's folder only once everything checks
  out.

### Data handling

- **Comments** come from `comments/<ticket id>.json`, a JSON list of Zendesk comments. Each file starts with the ticket description, so the description isn't added twice. A ticket without a comment file keeps its description as its only comment.
- **Corpus cleansing** happens only when building the corpus and the LDA documents. `DataWrangler._cleanse` normalizes text and drops emails, URLs, UUIDs, MD5 hashes, IPv4 addresses and words with no letters or digits. Everything else keeps the original text.

### Using it from Python

```python
from wrangler import DataWrangler

wrangler = DataWrangler()               # tickets-only.json + comments/ in the current directory
wrangler.tickets_reshaped()
wrangler.comments_bound()

wrangler.ticket_to_xml(30882, output_path="30882.xml", xsd_path="ticket_schema.xsd")
wrangler.wrangled_tickets[0].output_to_dict()   # writes completed/ticket_<id>.json
documents = wrangler.ticket_documents()          # one cleansed document per ticket
```

- **`ticket_to_xml(ticket, output_path=None, xsd_path=None)`** takes a ticket ID or a raw Zendesk ticket dict and returns XML that validates against `ticket_schema.xsd`. Text is verbatim: nothing is cleansed. People and groups are numeric Zendesk IDs, because the export has no names or emails. Validation against the schema runs only if `lxml` is installed.
- **`Ticket.output_to_dict(output_path=None)`** writes the ticket as JSON and returns the same data as a dict.

### Known issues

- **Training workers re-import `app.py`.** gensim's `LdaMulticore` starts worker processes, and on macOS each one re-runs `app.py` from the top. So every worker loads the spaCy model and builds a hidden copy of the window. Training still works, but it's slow and uses a lot of memory. The fix is to move the module-level `nlp`, `app` and `window` setup under `if __name__ == "__main__":`.
- **Chart files aren't gitignored.** `coherence.png` and `lda_model.html` land in the directory you launch from and show up as untracked files. (`completed/` and `*.log` are already ignored.)

## Writing tickets to MongoDB

`DataWrangler` can load the export into the `tickets` collection that the
decision engine reads. It uses the same `MONGO_URI`, `MONGO_DB` and
`MONGO_COLLECTION` settings as the engine.

```python
from wrangler import DataWrangler

wrangler = DataWrangler()                  # tickets-only.json + comments/
wrangler.write_tickets_to_mongo()          # every ticket in the export
wrangler.write_ticket_to_mongo(30882)      # one ticket
```

Writes are keyed by `zendesk_id` and are safe to re-run:
- **New tickets** are inserted with `awaiting_review: true`.
- **Existing tickets** get only unseen comments appended, and their status, priority, assignee and other mutable fields refreshed.
- **`awaiting_review`** goes back to `true` only when new comments arrive.
- **Review history** (`agent_reviews`) is never touched.

Known limitations:
- Authors and assignees are numeric Zendesk IDs. Only the requester's comments get a `role`.
- `customer_id` and `jira_ids` stay empty until open items #8 and #7 are resolved.

## Decision engine

### What it does

For every ticket document with `awaiting_review: true`, the engine:

1. builds a JSON payload of the mapped fields. It wraps each comment body in `<customer_text>` tags as an injection guard, and never sends `raw_xml`.
2. calls the model through Cheaper Inference (`https://api.cheaperinference.com`, Anthropic-compatible), with the long static system prompt cached.
3. appends the result to `agent_reviews[]` and sets `awaiting_review: false`.

It **never** writes to Zendesk. A human acts on every recommendation.
Write-back (Module 3) and ingestion into Mongo (Module 1) are not built.

### The five categories (precedence order)

The model evaluates these top to bottom and stops at the first match.

1. **`escalate_leadership`**: crosses a departmental, relationship or systemic boundary. Examples are churn risk, SLA breach, data risk, contractual exposure, platform risk, or needing a CSM or the VP of Product.
2. **`escalation_internal`**: needs an action only the ticket's Assignee (the Support Owner) can take.
3. **`flag_reengage`**: the ticket is stalled because someone hasn't responded. At most 2 attempts, at least 7 days apart.
4. **`flag_solve`**: recommend closing. Requires confidence of at least 75, and the code enforces this as a hard backstop.
5. **`manual_triage`**: an honest abstention when information is too thin or confidence is genuinely low. Invalid JSON from the model also routes here.

The verbatim prompt is in `src/decision_engine/prompts.py`.

### Running it

Requires `CHEAPER_INFERENCE_API_KEY` and `MONGO_URI`. `MONGO_DB`,
`MONGO_COLLECTION` and `DECISION_ENGINE_MODEL` are optional overrides; see
`.env.example`.

```bash
cd src && python -m decision_engine.persistence
```

or, from Python:

```python
from decision_engine.persistence import run_review_batch, process_one_ticket
run_review_batch()               # every ticket with awaiting_review: true
process_one_ticket("30882")      # a single ticket
```

### Feedback and the weekly review

On the Decision Engine tab, select a review and use **Mark Correct** or
**Mark Misclassified…**. A misclassification is written to
`misclassification_log` (`MISCLASSIFICATION_COLLECTION`) and marks the
review's outcome "bad".

**Weekly Review…** opens one week's log (Monday to Sunday, UTC). Enter your
name as Reviewer, select entries and **Mark Selected Reviewed**. Only
reviewed entries feed a summary: **Suggest from Reviewed** drafts one from
them for you to edit, and **Save Draft** stores it, unapproved, in
`lessons_learned` (`LESSONS_COLLECTION`) with the IDs of the entries it's
built on. Select a draft and **Approve Selected** to record who approved it.
Approved lessons aren't fed back into the prompt yet.

### Ticket embeddings (Voyage AI)

**Embed Tickets (Voyage AI)…** on the Decision Engine tab stores a Voyage
embedding on each ticket, then creates the Atlas vector index
`tickets_vector` (cosine, filterable by `status`). Set the key under Settings >
Embeddings (Voyage AI); the endpoint, model (`voyage-3.5`) and dimensions
(1024) have defaults. The default endpoint is Voyage on MongoDB Atlas,
`https://ai.mongodb.com/v1/embeddings`, for an Atlas model API key;
`EMBEDDING_API_ENDPOINT=api.voyageai.com` (or a full `EMBEDDING_API_URL`
ending in `/embeddings`) uses Voyage's own API instead. Requests go through
the `voyageai` Python client, with its retries off (only rate limits,
502/503/504 and refused connections are retried, since a timed-out request
may already have been billed) and redirects refused.

- It embeds each ticket's subject and first requester comment (internal
  notes never; the subject alone if no comment is marked as the requester's).
  It lists the tickets first and asks; only those approved are sent. These
  are paid calls.
- Each ticket gets `embedding` and `embedding_meta` (endpoint, model,
  dimensions, a hash of the embedded text, time). Re-running only embeds
  tickets whose text or any of those settings changed, so it is cheap to run
  after each Write Tickets to MongoDB. A ticket whose text is now empty loses
  its embedding.
- Work is saved batch by batch: closing the app or an API error keeps what
  was already embedded. A ticket edited while it was being embedded isn't
  overwritten with the old text's vector; the next run picks it up.
- Only one backfill runs at a time (a lock in the `embedding_jobs`
  collection, which expires after 15 minutes if its holder crashed).
- The vector index is checked **before** any paid call. If it exists with a
  different definition (dimensions, path, similarity, filters), nothing is
  sent: drop it in Atlas first. It is created only once every approved ticket
  is embedded, never after a cancelled or partial run.
- Requests are never redirected (that would resend the key and ticket text).
  Only failures where nothing was processed are retried (429, 502–504,
  connection refused); timeouts aren't, since Voyage may already have billed
  them. A batch Voyage rejects as too many tokens is split and retried.
- Without Atlas Search (a plain `mongod`), embeddings are still stored; only
  the index is skipped.
- Nothing queries the vectors yet.

From the command line:

```bash
cd src && python -m embeddings check      # one tiny call to test the key; no ticket text
cd src && python -m embeddings backfill   # same as the button
```

### Comparing embedding models

`src/embedding_eval.py` compares Voyage with other models on labelled tickets. It embeds every ticket in a labelled evaluation set
(the same file `decision_engine.evaluation` uses) with each model and checks
how often a ticket's nearest neighbours share its classification.

```bash
cd src && python -m embedding_eval ../eval_sets/backlog_sample.json --text opening full --show-neighbours 5
```

- Models: spaCy `en_core_web_lg` (free, local), Voyage (the app's embedding
  Settings; `--voyage-model` tries another model) and OpenAI
  `text-embedding-3-small` (`--openai-model`). Voyage needs
  `EMBEDDING_API_KEY` and OpenAI `OPENAI_API_KEY`; each is skipped without its
  key. They are paid calls that
  send ticket text to that vendor, so the script asks first (`--yes` skips
  the question).
- `--text opening` embeds the subject and first requester comment; `full`
  embeds every public comment (internal notes are never sent).
- Columns: **top-1** (nearest neighbour has the same label), **P@k** (share of
  the k neighbours with the same label) and **kNN@k** (the neighbours' majority
  label is right), with a random-neighbours baseline and P@k per category.
- Embeddings are cached in `embedding_cache/` (gitignored), so re-runs are free.
- Use a few hundred labelled tickets with several per category; a category with
  one ticket can never score.

### Tests

```bash
uv run pytest
```

The tests stub the model and use `mongomock`, so they need no credentials.
The live eval sends all five category fixtures plus a precedence-conflict
ticket to the real model. It spends money, so it only runs on request:

```bash
RUN_LIVE_EVAL=1 CHEAPER_INFERENCE_API_KEY=... uv run pytest -k live
```

### Known gaps (open items)

These are unresolved by design. Nothing in the code invents a value for them.

1. Leadership `@mention` recipient-selection logic (VP or one of the two Team Leads). This is a `[PLACEHOLDER]` in the prompt.
2. A real Escalate-to-Support-Owner edge-case ticket for evals. The current fixtures are synthetic.
3. The notification channel for leadership escalations.
4. A confirmed list of available signals. This is a `[PLACEHOLDER]` in the prompt. The data shows tags and priority, no SLA due date, and almost no CSAT; see the discovery notes.
5. A taxonomy glossary.
6. Confidence thresholds for every category other than Solve.
7. The Jira-integration account's display name and a confirmed comment format. The data suggests user `1031430881` with a `[JIRA->ZD Notification]` prefix, which is unconfirmed.
8. A stable customer identifier. `organization_id` is a candidate, but it's not in the XSD.
9. Whether `TicketOutcome` is usable ground truth. The discovery notes indicate **no**: it records root cause, and the export has no closed tickets.

Two more items came out of the integration itself:
- The model ID on Cheaper Inference is unconfirmed. It defaults to the brief's `claude-opus-4.8`.
- The Mongo database and collection names are unconfirmed. They default to `support_db.tickets`.
