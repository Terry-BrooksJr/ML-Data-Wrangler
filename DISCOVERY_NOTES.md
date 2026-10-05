# Discovery Notes: Decision Engine Integration (Phase 0)

Findings from inspecting this repo before merging the Zendesk triage decision
engine described in `AI_CODER_INTEGRATION_BRIEF.md`. Recorded 2026-10-05
against `main` @ `8199bbc`, with uncommitted changes to `src/` in the working tree.

## 1. What's in `src/`

| File | Role |
|---|---|
| `wrangler.py` | `DataWrangler` (PyQt `QObject`) reshapes Zendesk API JSON (`tickets.json` + `comments/<id>.json`) into `Ticket`/`Comment` objects, builds an LDA corpus, and has `ticket_to_xml()`, which **produces** `ticket_schema.xsd`-shaped XML from raw Zendesk JSON (with optional lxml validation). |
| `LDA_logic.py` | `LatentDirichletAllocator`: gensim LDA topic modelling plus pyLDAvis/seaborn visualisation. |
| `app.py` | PyQt5 desktop GUI that drives wrangling and LDA training. |
| `utility.py` | Log widgets/highlighter, text helpers. |
| `__init__.py` | Was misnamed `__init.py__` until 2026-10-05; renamed. Modules still import each other as top-level names (`from wrangler import DataWrangler`), so scripts run from inside `src/`. |

**There is no zero-shot classification, transformers, Mongo, Anthropic, or
ingestion-into-a-database code anywhere.** The "zero-shot" in the README
describes intent, not something the code does.

**Relationship to the decision engine:** the two are separate concerns. LDA is
unsupervised, offline topic modelling, while the decision engine is a live
LLM classifier. They share one data path, though: `ticket_to_xml()` is a
natural producer of the XML that Module 1 (ingestion) would consume. The
decision engine should live as its own module, sitting next to the LDA code
rather than merged into it.

## 2. Dependency source of truth

`pyproject.toml` (Poetry, `package-mode = false`) is authoritative.
`requirements.txt` is in `poetry export` format (pinned, with environment
markers), so it is a generated artifact. New dependencies go through
`poetry add`, then a re-export. Two gaps:
- `lxml` is not declared, but `ticket_to_xml()` uses it for XSD validation, and validation is skipped silently when it's missing.
- There is no test runner (pytest) yet.

Other environment facts:
- There's no `.env.example` and no secrets convention. `.env` is gitignored.
- `Dockerfile` is empty (0 bytes).
- There's no CI config.

## 3. `ticket_schema.xsd` vs. brief §4.4

The **element names and order match §4.4 exactly.** The differences are in
what the real producer (`ticket_to_xml`) puts into those elements:

| XSD field | Brief assumed | What the data actually provides |
|---|---|---|
| `Requester/Name`, `CC/Name` | Name + Email + Title | **Numeric Zendesk user IDs**. `Email` and `Title` are never emitted. The "email as interim customer_id" plan has no email to use. |
| `Assignee`, `Group`, `Comment/Author` | Display names | **Numeric IDs** (`assignee_id`, `group_id`, `author_id`). The model would see "assignee 387134954297" unless a users lookup is added. |
| `StatusCategory` / `TicketStatus` | Distinct fields | Both are set to the same Zendesk `status`. |
| `RequestType` | May encode taxonomy | Custom field `27124341`: `developer_support` (370), `author_support` (255), null (6). This is a product line. It is **not** taxonomy-like. |
| `TicketOutcome` | Possible ground truth | Custom field `27219391`: `learnosity_bug` (308), `possible_feature_enhancement` (196), `client_education` (76), null (42), and a few others. This is a **root-cause category**. It is **not** a triage decision. |
| `Comment/IsInternalNote` | Per comment | Derived from `!public`. Preserved per comment, which is good. |

`wrangler.py` reads `RequestType`/`TicketOutcome` by **list position**
(`fields[0]`, `fields[2]`). That is correct for every ticket in the current
dump, but it would be safer keyed by custom-field ID.

## 4. `tickets-only.json` and `comments/`

- `tickets-only.json` holds 631 raw Zendesk API ticket objects, created between 2019-06-21 and 2026-08-24. `comments/` holds 632 files: one per ticket, plus a stray copy of `tickets-only.json` inside it.
- Every ticket is `type: incident`. **Status: 606 hold, 13 pending, 12 open, and 0 solved or closed.**
- So this is a live **open backlog** snapshot. It is not a training or eval set of resolved tickets.
- Comments: 7,190 real comments (4,381 public, 2,809 internal). `comments/` also holds `tickets-only.json`, a byte-identical copy of the root file containing the 631 ticket objects. `_raw_comments_for` only matches files named by ticket ID, so the copy is never read as comments. It's harmless but can be deleted.
- `DataWrangler`'s default `ticket_file` was `tickets.json`, which doesn't exist. It now defaults to `tickets-only.json` (fixed 2026-10-05).

**Bearing on open item #9:** `TicketOutcome` is **not usable** as ground
truth for `outcomes`. It records root cause, not one of the five triage
decisions, and none of these tickets are closed anyway.

## 5. Evidence relevant to other open items (for the owner to confirm, not resolved here)

- **#4 Available signals:**
  - **Present:** tags (246 distinct). They include tier-like `enterprise`/`standard`, region `emea`/`apac`, `jira_escalated` (383), `closed_no_response`, and AI-derived `sentiment__*` and `intent__*` tags.
  - **Set on 581 tickets:** `priority`.
  - **Set on 629 tickets:** `organization_id`.
  - **Effectively absent:** SLA due date (`due_at` is never set) and CSAT (590 unoffered, 40 offered, 1 bad).
  - **Not mapped:** none of these signals is mapped into the XSD today.
- **#7 Jira integration account:** user ID `1031430881` posts internal comments via `api` (279 of them), prefixed `[JIRA-&gt;ZD Notification]` (HTML-escaped), e.g. "Jira issue LRN-25010 is now closed." / "Issue LRN-25009 Update: The development team has started working on this issue." The display name isn't in the data. The owner still needs to confirm before a parser is built.
- **#8 Stable customer ID:** Zendesk `organization_id` is a candidate. The XSD doesn't carry it.

## 6. Decisions taken for Phases 1–5

The owner approved these defaults:
- **Dependencies:** added through Poetry (`pyproject.toml`). `poetry.lock` and `requirements.txt` still need regenerating: `poetry lock && poetry export -f requirements.txt -o requirements.txt`.
- **Layout:** `src/decision_engine/` is a self-contained package. The existing flat modules are untouched.
- **Mongo:** `support_db.tickets` by default, overridable with `MONGO_DB` and `MONGO_COLLECTION`.
- **Model:** set by `DECISION_ENGINE_MODEL`, defaulting to `claude-opus-4.8` (with a TODO in `classify.py`).

Deviations from the brief's reference code. All are mechanical; none changes classification logic:
- The API client and Mongo connection are built lazily, so modules import without secrets (needed for tests).
- `classify_ticket` reads the first text block rather than `content[0]`, which may be a thinking block.
- Each review entry gets a `review_id: ObjectId`. Schema §4.2 requires it, the reference code omitted it, and `outcomes` references it.
- The update filter also matches `awaiting_review: True`, so concurrent runs can't double-append.
- `run_review_batch` tolerates already-reviewed tickets and logs with loguru, not print.
- The payload's `json.dumps` uses `default=str`, because `reengage_attempts` holds datetimes.

## 7. Original pre-Phase-1 questions

1. Confirm Poetry as source of truth (and add `pytest`/`mongomock` as dev dependencies).
2. Package layout: keep `src/` flat and add `src/decision_engine/` as the first real package (run as `python -m decision_engine.persistence` from `src/`), or fix `__init.py__` and make `src/` a package.
3. Mongo database/collection names (the brief uses `support_db.tickets` as a placeholder).
4. Model ID on Cheaper Inference: the brief's `claude-opus-4.8` needs confirming against the gateway's model list.
5. Whether to add a user/group ID→name lookup so `assignee`/`author` are meaningful to the model. That work belongs to ingestion (Module 1), not the engine.
