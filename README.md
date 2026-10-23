# Lasso (ML-Data-Wrangler)
Data Wrangler and Reshaper for Downstream LDA And Zero-Shot Classification ML FIne Tuning 

The repo holds two separate pieces of work that read the same Zendesk ticket data:

| Component | Location | What it does |
|---|---|---|
| **Data wrangler + LDA** | `src/wrangler.py`, `src/LDA_logic.py`, `src/app.py` | A PyQt5 desktop app. It reshapes Zendesk API exports (`tickets-only.json`, `comments/`) into a cleaned corpus, trains gensim LDA topic models and visualises them. `DataWrangler.ticket_to_xml()` also emits tickets as XML that validates against `ticket_schema.xsd`. |
| **Triage decision engine** | `src/decision_engine/` | An LLM classifier that reads one mapped ticket from Mongo and recommends a single triage action. |

The two are **not** the same concern. LDA is unsupervised, offline topic
modelling. The decision engine is a live classifier against a fixed taxonomy.
No zero-shot or fine-tuning code exists yet.

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
the app offers to open Settings if any required value is missing. Every
change applies as soon as you save; nothing needs a restart. Blanking a field makes the code's default apply, even if
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

The app is styled with [PyQt-SiliconUI](https://github.com/ChinaIceF/PyQt-SiliconUI)
(GPL-3.0, like PyQt5 itself). Its buttons, combo boxes, checkboxes, progress
bars, scroll bars and cards are SiliconUI widgets (`src/widgets.py`); the
plain Qt widgets around them (tabs, tables, text boxes, dialogs) get a
stylesheet built from SiliconUI's colour tokens (`src/theme.py`). SiliconUI's
widgets come only in dark, so there is no longer a light or native theme and
the Theme setting is gone; a saved `LASSO_THEME` is ignored.

SiliconUI isn't on PyPI in a current version, so `pyproject.toml` installs it
from GitHub at a pinned commit. To try a newer one, change `rev` under
`[tool.uv.sources]` and run `uv lock`.

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

## Normalizing tags

The **Tag Normalizer** tab finds tags that mean the same thing and maps each
to one tag, so "oil change", "tire_rotation" and "Car Maintenance" all become
`car_maintenance`. It never writes to Zendesk; the result is a mapping you
export, and **Save Run** keeps it (see [Saved runs and normalized tags](#saved-runs-and-normalized-tags)).

1. Paste tags (comma- or newline-separated, or a JSON list), or **Load
   File…**. A CSV such as a Zendesk tag export (`name,count`) uses its first
   column; the count is ignored.
2. Pick **Voyage AI** (the embedding Settings; sends only the tag names),
   **Language model** (see below) or
   **spaCy** (local and free, but misses looser synonyms), then **Find
   Similar Tags**. Each tag is embedded once per session; re-runs only embed
   new tags.
3. Adjust **Merge at similarity ≥** to regroup instantly. Voyage scores short
   tags high (unrelated pairs often reach 0.8), so its default is 0.90;
   spaCy's is 0.80.
4. Double-click a group's tag to rename it, untick any tag to keep it out of
   its group, then **Export Mapping…** as CSV (`tag,canonical_tag,uses,similarity`)
   or JSON.

Before embedding, tags are cleaned: underscores, hyphens and camelCase become
spaces and case is ignored, so `car_maintenance` and `Car Maintenance` are
one tag. A group is named by its most central tag, in its most common
spelling.

Structured `facet__value` tags (`sentiment__positive`, `intent_confidence__low`,
`intent__software__sync__issue`) come from a fixed taxonomy and share long
prefixes, so opposite values embed as near-identical. Two of them are never
grouped together; a free-form tag can still join one (`central_2fa_reset` →
`intent__software__verification_code__2fa_reset`), and the structured tag
names the group. Untick **Keep facet__value tags apart** to allow it.

The same from the command line:

```bash
cd src && python -m tag_normalizer tags.csv --out tag_mapping.csv
cd src && python -m tag_normalizer tags.txt --backend spacy --threshold 0.8
```

### Saved runs and normalized tags

**Save Run** keeps a run the way LDA and BERTopic runs are kept: its folder
under `tag_normalizer_runs/` (`TAG_RUN_DIR`) holds the tags, their vectors and
the mapping (CSV and JSON); with model storage set up it is uploaded with a
signed manifest to `s3://<bucket>/<MODEL_PREFIX>tag_normalizer/<run_id>/`
(retry a failed upload with `cd src && python -m model_storage upload --tags RUN_ID`);
and it is recorded in MongoDB's `tag_normalization_run_log`
(`TAG_RUNS_COLLECTION`) with its settings, your renames and unticked tags,
and every tag it changed. **Previous Runs…** lists runs from MongoDB and
this computer and reopens one exactly as saved (downloading and verifying it
first if needed), without embedding anything again.

Saving also records the run's decisions in `normalized_tags`
(`NORMALIZED_TAGS_COLLECTION`), one document per successor tag:

```json
{
  "tag": "password_reset",
  "text": "password reset",
  "replaces": [
    {"tag": "pw_reset", "text": "pw reset", "run_id": "20261020T...", "replaced_at": "..."}
  ],
  "run_ids": ["20261020T..."],
  "created_at": "...", "updated_at": "..."
}
```

- A deprecated tag belongs to one successor (unique index on `replaces.tag`).
- Successors never chain: merging successor A into B moves A and everything
  A replaced to B, and A's document goes.
- The latest run wins: a tag deprecated before but kept as itself (unticked)
  or chosen as a successor now is taken off its old successor; a successor
  left replacing nothing is removed.

**Find Similar Tags** reads `normalized_tags` back. Every saved successor
joins the grouping (embedded with your tags), so new synonyms join it and
it keeps its name, shown as "(saved successor)" when it isn't in your input;
two saved successors are never grouped together (rename a group to merge
them). A tag already deprecated goes straight to its successor's group.
Without `MONGO_URI` the tab works as before and Save Run keeps the folder
(and upload) only.

### Normalizing with a language model

The third option, **Language model**, sends the tags (with how often each was
seen) to a model that follows
[`src/tag_normalization_reference.md`](src/tag_normalization_reference.md).
That file holds the rules R1-R11, the tag registry, known overlap clusters and
the JSON output contract, and is sent as the model's instructions.
[`src/tag_normalization_priorities.md`](src/tag_normalization_priorities.md)
follows it and wins where they differ. The goal is discoverability: merge tags
that name the same concept even when they share no words, coin a new tag when
that names the concept best, and never rename a tag only to change its format.
It also loosens two of the reference's rules: a specific tag merges into its
general tag (`nvda_screenreader` into `screen_readers`), and a `uses_X` tag
into `X` (`uses_data_api` into `data_api`), when that's where people would
search. Different products, teams and facet values still stay apart. Edit
either file to change how tags are normalized.

**Tags starting with `intent` are Zendesk AI tags and are left alone**: they
are never embedded or sent to the model, nothing may be merged into one (a
reply that tries is reverted and flagged), and they appear unchanged in the
mapping, so the retag scripts never touch them.

Before asking the model, Voyage embeds the other tags and proposes candidate
pairs: every pair at 0.87 similarity or more (two facet__value tags are never
paired), closest first, up to 300. The model judges each pair, one tag
against the other, and may also merge tags no pair holds. Without Voyage
configured, the model runs without them and says so; `--no-candidates` skips
them on the command line. The model maps every tag to a canonical tag,
explains its non-trivial decisions (shown as tooltips) and raises flags for a
person to decide (shown below the groups). Rename, untick and both exports
work as for embeddings; the JSON export also holds the model's `decisions`,
`flags`, `model`, the `candidates` it judged and the `ignored` intent tags.
There is no threshold: the rules decide. Language-model runs can't be saved
with **Save Run** yet (saved runs keep a vector per tag and regroup by
embeddings when reopened), and they don't apply earlier `normalized_tags`
decisions; use **Export Mapping** to keep the result.

The reply is checked before it's shown:
- Every input tag must be mapped. A tag the model leaves out keeps its name
  and is flagged.
- Registry tags marked `automated: true` or `pending` are never renamed (R3),
  whatever the model answers.
- Merges the model rates below 0.8 confidence are kept apart and flagged.
- A rename that only changes a tag's form (`sentiment__positive` →
  `sentiment_positive`) to a name no other tag shares is undone.

Requests use the Decision Engine's Cheaper Inference settings (key, https
URL). The model is **Tag Normalizer model** (`TAG_NORMALIZER_MODEL`), or the
primary model when that's blank. Only tag names and counts are sent. On the
195-tag example export, one run takes one to two minutes and about 25k tokens. One
request holds at most 1,000 tags; split longer lists, since splitting hides
merges across the parts.

```bash
cd src && python -m tag_normalizer tags.csv --backend llm --out tag_mapping.json
```

### Retagging tickets in Zendesk

**Export Zendesk Scripts…** (or `--scripts FOLDER` on the command line)
writes a folder for a person to run by hand; Lasso itself never calls
Zendesk. It holds `tag_changes.csv` (`old_tag,new_tag`, only tags that
change) and three standard-library Python scripts, run in order:

1. `1_find_tickets.py` (read-only) searches for tickets with each old tag
   (`GET /api/v2/search/export`, cursor-paginated) and writes
   `tickets_to_retag.csv`.
2. `2_remove_old_tags.py --apply` removes each old tag from those tickets.
3. `3_add_new_tags.py --apply` adds the new tag to the same tickets.

Steps 2 and 3 use `PUT /api/v2/tickets/update_many.json` (100 tickets per
request, `remove_tags` / `additional_tags`, so other tags are untouched) and
wait for each job to finish. They print a dry run unless given `--apply`,
read the ticket list from step 1 rather than searching again, skip closed
tickets (Zendesk won't update them) and log each ticket's result. They read
`ZENDESK_SUBDOMAIN`, `ZENDESK_EMAIL` and `ZENDESK_API_TOKEN` from the
environment. A new tag with spaces or capitals is written as a valid Zendesk
tag (`Vehicle Care` → `vehicle_care`). Each update is a normal ticket update,
so it can fire triggers that look at tags. The folder's `README.txt` has the
full steps.

```bash
cd src && python -m tag_normalizer tags.csv --scripts ../zendesk_retag_run
```

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
run_review_batch()               # every ticket with awaiting_review: true, oldest first
process_one_ticket("30882")      # a single ticket
```

In the app, the **Decision Engine** tab's **Run Batch…** asks how many of the
tickets awaiting review to classify (1 to all of them, oldest Zendesk ID
first), then shows progress after each ticket:
- **Progress:** how many are done of how many, and what happened to them
  (classified, to manual triage, skipped, errored).
- **Speed and time left:** the average seconds per ticket so far, and the
  estimated time left at that speed.
- **Recent Reviews:** each new review appears in the list as soon as it's
  saved, and the review you're reading stays selected.

**Stop After This Ticket** lets the ticket being reviewed finish and save, then
stops; the rest stay awaiting review for the next batch.

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
- The model ID on Cheaper Inference is unconfirmed. It defaults to `claude-opus-4.8` (`DECISION_ENGINE_MODEL`).
- The Mongo database and collection names are unconfirmed. They default to `support_db.tickets`.
