# ML-Data-Wrangler
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

```bash
poetry install
cp .env.example .env   # then fill in values
```

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

### Tests

```bash
poetry run pytest
```

The tests stub the model and use `mongomock`, so they need no credentials.
The live eval sends all five category fixtures plus a precedence-conflict
ticket to the real model. It spends money, so it only runs on request:

```bash
RUN_LIVE_EVAL=1 CHEAPER_INFERENCE_API_KEY=... poetry run pytest -k live
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
