---
title: Zendesk Tag Normalization Reference
purpose: >
  Reference for an LLM that receives a list of Zendesk ticket tags and normalizes it:
  grouping near-synonyms and tags with shared semantic meaning under one canonical tag.
sources:
  - "Support Tagging: How to Get Started with Zendesk Tagging (PDF)"
  - "Tagging Scope Prefix Reference-old - Sheet1 (PDF)"
compiled: 2026-10-20
format_notes: >
  Section 1-3 are prose rules. Section 4 is the tag registry as YAML. Section 5 lists known
  overlap clusters. Section 6 is the required output contract (compatible with
  src/tag_normalizer.py's JSON output). Fields marked `source: inferred` were not stated in
  the source documents and should be treated as lower-confidence guidance.
---

# Zendesk Tag Normalization Reference

## 1. Your task

You will receive a list of tags (possibly with usage counts). Produce a mapping from every
input tag to a canonical tag, so that tags expressing the same meaning collapse to one form.

Why this matters (from the source guide): duplicate tagging such as `billing_issue` and
`payment_problem` on the same underlying issue skews analytics, triggers redundant
workflows, misroutes tickets, gives agents conflicting categorizations, and inflates
metrics. Normalization enforces a controlled vocabulary by consolidating synonymous terms.

## 2. Tag model

There are two classes of tags by origin: **Automated** (added by conditions, events or
business rules; no engineer involvement) and **User** (added by an engineer). By shape,
tags fall into these kinds:

| kind | shape | meaning |
|---|---|---|
| `context` | `<scope_prefix><domain>[_<entity>]` | Per-ticket, customizable. Tracks patterns and metrics about a specific concept. |
| `static` | fixed string | Same on every ticket where it applies. Flags an event (e.g. a new Zendesk user request) or a specialty category for external stakeholders. Mostly applied by automation, but may be added manually. |
| `dynamic_id` | `<prefix><identifier>` | Prefix plus a machine identifier (e.g. a UUID). |

### Anatomy of a context tag

1. **Scope prefix** (tag root): chosen from the predefined list in Section 4. It is the
   category or classification of the tag and always ends in `_`.
2. **Domain / concept**: the primary subject, issue or domain of the ticket (high-level).
3. **Entity** (optional): the specific component or element. Many tags stop at
   prefix + domain because that is specific enough.

Example: `activity_player_settings_accessibility` = prefix `activity_player_settings_` + domain `accessibility`.
Example: `client_education_async_scoring` = prefix `client_education_` + domain `async_scoring`.

### Formatting conventions

- Lowercase `snake_case`; words joined by `_`. Zendesk tags cannot contain spaces.
- Prefixes in the registry end with `_`; a full context tag never ends with `_`.

## 3. Normalization rules

Apply in order. Cite the rule ID in your output for each non-trivial decision.

- **R1 Surface variants are the same tag.** Differences only in case, separators
  (`-`, ` `, `.`, `__`), singular/plural, or obvious typos map to one form.
  `Call-Followup`, `call followup`, `call_followups` → `call_followup`.
- **R2 Prefer the registry.** If any member of a group is in the registry (Section 4), or
  is a context tag built on a registry prefix, that form is canonical. Use the registry's
  exact spelling, even where it is misspelled (see R8).
- **R3 Protected tags.** Tags with `automated: true` or `automated: pending` are driven by
  Zendesk automation and business rules. Never rename them or merge them into another tag.
  Other tags may be merged *into* them when they mean the same thing. Manual tagging
  complements automation and must not disrupt it.
- **R4 Same meaning, different words.** Merge true synonyms and near-synonyms that refer to
  the same underlying issue (`billing_issue` + `payment_problem`). Without a registry
  member, pick the clearest, most general, most frequently used spelling as canonical.
- **R5 Do not merge across prefixes.** The scope prefix carries meaning. Tags with the same
  domain under different prefixes are different tags: `troubleshooting_scoring`,
  `client_education_scoring` and `best_practice_scoring` stay separate.
- **R6 Do not merge across specificity levels.** A tag with an entity is not a duplicate of
  its parent: `author_site_settings` is not `author_site_settings_permissions`. Only merge
  when the entity is a synonym (`sdk_php` and `sdk_php_sdk` merge).
- **R7 Do not merge distinct targets of a shared family.** Escalation, initiation and
  `uses_*` tags name different teams or products. `csm_escalated`, `product_escalation` and
  `support_vp_escalation` are distinct even though all are escalations.
- **R8 Flag, don't fix, registry misspellings.** Some registry entries are misspelled
  (`spelling_issue` field). Map variants to the registry spelling and add a flag with the
  suggested correction. Correcting the registry is a human decision.
- **R9 Static vs context.** Never merge a static tag into a context prefix or vice versa
  unless Section 5 says to. Example: `reports` (unknown) vs `report_<REPORT_TYPE>` (context)
  vs `uses_reports` (static) are kept apart.
- **R10 Dynamic IDs.** `content_cue_<UUID>` and similar tags are unique per item. Never merge
  two different IDs. Only normalize surface form (R1).
- **R11 When unsure, keep and flag.** If two tags may or may not mean the same thing, map
  each to itself and add a flag with your reasoning. A wrong merge corrupts data; a missed
  merge is easy to fix later.

## 4. Tag registry

Field meanings:
- `kind`: `context` (prefix, ends in `_`), `static`, `dynamic_id`, or `unknown` (not
  classified in the source).
- `automated`: `true`, `false`, `pending` (automation planned), or `unknown`.
- `spelling_issue`: present when the registry spelling looks wrong; the value is the
  likely intended spelling (`source: inferred`).
- `definition` / `example` / `notes`: verbatim intent of the source, with typos fixed.

```yaml
registry:
  - {tag: "<accessibility tag; name lost in source export>", kind: static, automated: true, definition: "The issue, topic or subject matter relates to digital accessibility.", notes: "Tag name rendered as #ERROR! in the source sheet. Likely candidates: accessibility, a11y. Confirm before use."}
  - {tag: activity_player_settings_, kind: context, automated: false, definition: "A specific setting of the assessment (activity) player is the subject.", example: activity_player_settings_accessibility}
  - {tag: agent_ooo, kind: static, automated: true, definition: "The ticket owner is marked out of office in Zendesk."}
  - {tag: author_site_, kind: context, automated: false, definition: "The main issue relates to the suffixed element of the Author Site, not only the authoring process.", example: author_site_settings}
  - {tag: "<authoring-process tag; name lost in source export>", kind: context, automated: false, definition: "Unlike author_site_, the main issue involves the process of authoring.", notes: "Tag name missing in source. Do not merge with author_site_."}
  - {tag: best_practice_, kind: context, automated: false, definition: "Advice on the widely recognized most effective and efficient way to achieve an outcome.", example: best_practice_altering_live_content, notes: "Used to review how customers were advised in the past on common topics."}
  - {tag: bulk_update, kind: static, automated: true, definition: "A bulk content update. Common with HMH."}
  - {tag: call_followup, kind: static, automated: false, definition: "A call was completed and a summary or outstanding action items were provided."}
  - {tag: call_author_training, kind: static, automated: false, definition: "A request for an author training call.", notes: "Definition truncated in source; completion is inferred."}
  - {tag: central_2fa_reset, kind: static, automated: true, definition: "Request to reset a two-factor authentication device registration."}
  - {tag: central_data_update_, kind: context, automated: false}
  - {tag: client_churn, kind: static, automated: pending, definition: "Created by Customer Success to remove a customer's access to Learnosity services."}
  - {tag: client_authoring_only, kind: static, automated: true}
  - {tag: client_education_, kind: context, automated: false, definition: "Enabling customers to make informed implementation and API usage decisions. The domain is the primary topic the support engineer focused on.", example: client_education_async_scoring, notes: "Required when the ticket outcome is 'Client Education'."}
  - {tag: client_poc_work, kind: static, automated: false, definition: "Helps Sales Engineering validate a prospect's proposed use case or implementation approach."}
  - {tag: closed, kind: static, automated: true, definition: "Ticket entered the irreversible 'closed' status."}
  - {tag: closed_no_response, kind: static, automated: true, definition: "Moved from pending to solved by the business rule after 7 calendar days with no activity from the requester or external parties."}
  - {tag: compatibility_, kind: context, automated: false}
  - {tag: configruation_, kind: static, automated: unknown, spelling_issue: configuration_, notes: "Ends in _ (prefix shape) but listed as static; classification unclear."}
  - {tag: "content_cue_<UUID>", kind: dynamic_id, automated: unknown, example: content_cue_08469e2e-0600-1ee7-be77-9ff39f772c83}
  - {tag: content_update_, kind: context, automated: unknown}
  - {tag: cross_client, kind: static, automated: unknown}
  - {tag: csat, kind: static, automated: false}
  - {tag: csm_escalated, kind: static, automated: unknown, definition: "Resolution requires strategic involvement of, or escalation to, the Customer Success Manager (CSM)."}
  - {tag: csm-initated, kind: static, automated: unknown, spelling_issue: csm_initiated}
  - {tag: custom, kind: context, automated: unknown, notes: "Possibly custom_ prefix; see cluster C5."}
  - {tag: customization, kind: static, automated: unknown}
  - {tag: design, kind: unknown, automated: unknown}
  - {tag: developer, kind: unknown, automated: unknown}
  - {tag: duplicate_ticket, kind: static, automated: unknown}
  - {tag: duplicate_shared_passages, kind: static, automated: unknown}
  - {tag: dynamic_content, kind: unknown, automated: unknown}
  - {tag: edge_case, kind: static, automated: false}
  - {tag: engineering_escalations, kind: static, automated: false}
  - {tag: fair_use, kind: unknown, automated: false}
  - {tag: feature_, kind: unknown, automated: false}
  - {tag: feature_enablement_, kind: context, automated: false}
  - {tag: feature_disablement_, kind: context, automated: unknown}
  - {tag: feedback_aide, kind: static, automated: false}
  - {tag: firehose_init_request, kind: static, automated: pending}
  - {tag: google, kind: static, automated: true}
  - {tag: groom, kind: static, automated: unknown}
  - {tag: implementation_, kind: context, automated: unknown}
  - {tag: implementation_advisory_, kind: context, automated: unknown}
  - {tag: intergration_, kind: context, automated: unknown, spelling_issue: integration_}
  - {tag: issue_, kind: context, automated: unknown}
  - {tag: item_configruation_, kind: context, automated: unknown, spelling_issue: item_configuration_}
  - {tag: items_api, kind: static, automated: unknown}
  - {tag: jira, kind: static, automated: true}
  - {tag: learning_outcomes, kind: unknown, automated: unknown}
  - {tag: learnosity_toolkit_recommenderd, kind: unknown, automated: unknown, spelling_issue: learnosity_toolkit_recommended}
  - {tag: load_test, kind: static, automated: false}
  - {tag: local_device, kind: static, automated: false}
  - {tag: lss, kind: unknown, automated: unknown}
  - {tag: lts, kind: unknown, automated: unknown}
  - {tag: manual_grading, kind: unknown, automated: unknown}
  - {tag: math, kind: unknown, automated: unknown}
  - {tag: migration, kind: unknown, automated: unknown}
  - {tag: migration_session, kind: unknown, automated: unknown}
  - {tag: onboarding_init_request, kind: static, automated: unknown}
  - {tag: parter, kind: unknown, automated: unknown, spelling_issue: partner}
  - {tag: partners_escalation, kind: static, automated: pending}
  - {tag: partnerships_, kind: context, automated: false}
  - {tag: possible_feature, kind: static, automated: true}
  - {tag: post_call_followup, kind: static, automated: unknown}
  - {tag: product_escalation, kind: static, automated: false}
  - {tag: qt_, kind: context, automated: false, notes: "Likely 'question type' (source: inferred)."}
  - {tag: question, kind: unknown, automated: unknown}
  - {tag: questions_api, kind: static, automated: true}
  - {tag: random, kind: unknown, automated: unknown}
  - {tag: re-opened, kind: static, automated: true, definition: "A ticket in solved status was reopened.", notes: "Hyphen is the real spelling; protected by R3. Map reopened / re_opened here."}
  - {tag: redacted, kind: static, automated: unknown}
  - {tag: regular, kind: unknown, automated: unknown}
  - {tag: "report_<REPORT_TYPE>", kind: context, automated: false, notes: "Prefix is report_; the domain is a report type."}
  - {tag: reports, kind: unknown, automated: unknown}
  - {tag: requested_deletion, kind: static, automated: false}
  - {tag: root_cause_, kind: context, automated: unknown}
  - {tag: sdk_, kind: context, automated: false, definition: "SDK used by the customer, when material to resolution.", example: sdk_php}
  - {tag: session_duplication, kind: unknown, automated: unknown}
  - {tag: signature, kind: unknown, automated: unknown}
  - {tag: single_page_app, kind: static, automated: false, definition: "A single page application (SPA) was material to the cause, investigation or resolution."}
  - {tag: special_handling, kind: static, automated: unknown}
  - {tag: support_call_request, kind: static, automated: false}
  - {tag: support_engineer_initiated, kind: static, automated: pending}
  - {tag: support_internal_transfer, kind: static, automated: pending}
  - {tag: support_lateral_escalation, kind: static, automated: pending}
  - {tag: support_leads_escalation, kind: static, automated: pending}
  - {tag: support_vp_escalation, kind: static, automated: pending}
  - {tag: tagging, kind: unknown, automated: unknown}
  - {tag: tags_vetted, kind: static, automated: unknown, notes: "Marked 'Other - See Notes' in source; no notes given. Likely a process marker that tags were reviewed (source: inferred)."}
  - {tag: tech, kind: unknown, automated: unknown}
  - {tag: templates_question, kind: unknown, automated: unknown}
  - {tag: top, kind: unknown, automated: unknown}
  - {tag: troubleshooting_, kind: context, automated: unknown}
  - {tag: unconventional, kind: unknown, automated: unknown}
  - {tag: unscheduled_load, kind: unknown, automated: unknown}
  - {tag: upgrade, kind: unknown, automated: unknown}
  - {tag: user, kind: unknown, automated: unknown}
  - {tag: uses_assess, kind: static, automated: unknown}
  - {tag: uses_author, kind: static, automated: unknown}
  - {tag: uses_data, kind: static, automated: unknown}
  - {tag: uses_question, kind: static, automated: unknown}
  - {tag: uses_reports, kind: static, automated: unknown}
  - {tag: venn_diagram, kind: unknown, automated: unknown}
  - {tag: video, kind: unknown, automated: unknown}
  - {tag: wcag_level, kind: unknown, automated: unknown}
  - {tag: zd, kind: unknown, automated: unknown}
  - {tag: zendesk_new_request, kind: static, automated: false}
```

## 5. Known overlap clusters

All clusters are `source: inferred` from the registry. `decision` is the default to apply;
`flag` means keep the tags separate and raise a flag for human review.

```yaml
clusters:
  - id: C1
    tags: [call_followup, post_call_followup]
    decision: merge
    canonical: call_followup
    reason: "Both mean a call happened and follow-up was sent. call_followup has a definition; post_call_followup has none."
  - id: C2
    tags: [re-opened, reopened, re_opened]
    decision: merge
    canonical: re-opened
    reason: "Surface variants (R1) of a protected automated tag (R3)."
  - id: C3
    tags: [feature_, feature_enablement_, feature_disablement_, possible_feature]
    decision: flag
    reason: "Enablement and disablement are opposite actions; possible_feature is a protected automated flag for feature requests. feature_ is ambiguous."
  - id: C4
    tags: [implementation_, implementation_advisory_, client_education_]
    decision: flag
    reason: "Related advisory work, but different prefixes (R5). client_education_ is required for the 'Client Education' outcome."
  - id: C5
    tags: [custom, customization, unconventional, edge_case]
    decision: flag
    reason: "custom vs customization look like near-synonyms, but one is listed as context and one as static (R9)."
  - id: C6
    tags: [author_site_, uses_author, "<authoring-process tag>", call_author_training]
    decision: keep_separate
    reason: "Source states author_site_ differs from the authoring-process tag. uses_author is a product-usage flag."
  - id: C7
    tags: [questions_api, uses_question, question, templates_question, qt_]
    decision: keep_separate
    reason: "API product flag vs usage flag vs question-type prefix. Only surface variants merge."
  - id: C8
    tags: ["report_<REPORT_TYPE>", reports, uses_reports]
    decision: flag
    reason: "reports (unclassified) may be a bare report_ tag with no domain. Suggest report_<type> when the type is known."
  - id: C9
    tags: [migration, migration_session, upgrade]
    decision: flag
    reason: "migration_session is more specific than migration (R6). upgrade may overlap migration."
  - id: C10
    tags: [csm_escalated, engineering_escalations, partners_escalation, product_escalation, support_lateral_escalation, support_leads_escalation, support_vp_escalation]
    decision: keep_separate
    reason: "Distinct escalation targets (R7). Merge only surface variants, e.g. product_escalations → product_escalation."
  - id: C11
    tags: [csm-initated, support_engineer_initiated, onboarding_init_request, firehose_init_request]
    decision: keep_separate
    reason: "Different initiators/request types. Map csm_initiated / csm-initiated / csm_initated → csm-initated (R2, R8)."
  - id: C12
    tags: [parter, partnerships_, partners_escalation]
    decision: flag
    reason: "parter is probably a misspelled partner tag; partnerships_ is a context prefix; partners_escalation is an escalation (R7)."
  - id: C13
    tags: ["<accessibility tag>", wcag_level, activity_player_settings_accessibility]
    decision: keep_separate
    reason: "General accessibility flag vs WCAG conformance level vs a specific player setting (R6)."
  - id: C14
    tags: [zd, zendesk_new_request, google, jira]
    decision: flag
    reason: "zd is an unclassified abbreviation of Zendesk; do not merge into zendesk_new_request without evidence."
  - id: C15
    tags: [duplicate_ticket, duplicate_shared_passages, session_duplication]
    decision: keep_separate
    reason: "Duplicate ticket (process) vs duplicated content vs duplicated sessions (product behavior)."
  - id: C16
    tags: [closed, closed_no_response]
    decision: keep_separate
    reason: "Both protected automated tags with different triggers (R3)."
  - id: C17
    tags: [call_followup, support_call_request]
    decision: keep_separate
    reason: "Request for a call vs follow-up after a call."
```

## 6. Output contract

Return one JSON object and nothing else. `mapping` must contain **every** input tag
exactly once (use the input spelling as the key). `groups` lists only groups with more than
one member, canonical first. This matches the JSON emitted by `src/tag_normalizer.py`,
with `decisions` and `flags` added.

```json
{
  "mapping": {"<input tag>": "<canonical tag>"},
  "groups": [{"canonical": "<canonical tag>", "tags": ["<canonical tag>", "<merged tag>"]}],
  "decisions": [
    {
      "tag": "<input tag>",
      "canonical": "<canonical tag>",
      "action": "keep | merge | normalize_form",
      "rules": ["R1"],
      "confidence": 0.0,
      "reason": "<one sentence>"
    }
  ],
  "flags": [
    {
      "tags": ["<tag>", "<tag>"],
      "type": "possible_synonym | registry_misspelling | unknown_prefix | protected_tag | missing_entity",
      "suggestion": "<what a human should decide>",
      "rules": ["R11"]
    }
  ]
}
```

- `keep`: canonical equals the input tag.
- `normalize_form`: only case, separator, plural or typo changed (R1).
- `merge`: the input tag is a different word or phrase with the same meaning (R4).
- `confidence`: 0-1. Merges below 0.8 should be flags instead.
- `decisions` may omit tags whose action is `keep` and need no explanation.

### Worked example

Input:

```json
["billing_issue", "payment_problem", "Call-Followup", "post_call_followup", "reopened",
 "troubleshooting_scoring", "client_education_scoring", "intergration_lti",
 "integration_lti", "product_escalations", "csm_escalated", "feature_enablement_math"]
```

Output:

```json
{
  "mapping": {
    "billing_issue": "billing_issue",
    "payment_problem": "billing_issue",
    "Call-Followup": "call_followup",
    "post_call_followup": "call_followup",
    "reopened": "re-opened",
    "troubleshooting_scoring": "troubleshooting_scoring",
    "client_education_scoring": "client_education_scoring",
    "intergration_lti": "intergration_lti",
    "integration_lti": "intergration_lti",
    "product_escalations": "product_escalation",
    "csm_escalated": "csm_escalated",
    "feature_enablement_math": "feature_enablement_math"
  },
  "groups": [
    {"canonical": "billing_issue", "tags": ["billing_issue", "payment_problem"]},
    {"canonical": "call_followup", "tags": ["call_followup", "Call-Followup", "post_call_followup"]},
    {"canonical": "re-opened", "tags": ["re-opened", "reopened"]},
    {"canonical": "intergration_lti", "tags": ["intergration_lti", "integration_lti"]},
    {"canonical": "product_escalation", "tags": ["product_escalation", "product_escalations"]}
  ],
  "decisions": [
    {"tag": "payment_problem", "canonical": "billing_issue", "action": "merge", "rules": ["R4"], "confidence": 0.9, "reason": "Same underlying issue; the source guide uses this exact pair as its duplicate-tag example."},
    {"tag": "Call-Followup", "canonical": "call_followup", "action": "normalize_form", "rules": ["R1", "R2"], "confidence": 1.0, "reason": "Case and separator variant of a registry tag."},
    {"tag": "post_call_followup", "canonical": "call_followup", "action": "merge", "rules": ["R4"], "confidence": 0.85, "reason": "Cluster C1."},
    {"tag": "reopened", "canonical": "re-opened", "action": "normalize_form", "rules": ["R1", "R3"], "confidence": 1.0, "reason": "Variant of a protected automated tag; the registry spelling is hyphenated."},
    {"tag": "client_education_scoring", "canonical": "client_education_scoring", "action": "keep", "rules": ["R5"], "confidence": 1.0, "reason": "Different prefix from troubleshooting_scoring."},
    {"tag": "integration_lti", "canonical": "intergration_lti", "action": "normalize_form", "rules": ["R2", "R8"], "confidence": 0.95, "reason": "Registry prefix is spelled intergration_."},
    {"tag": "product_escalations", "canonical": "product_escalation", "action": "normalize_form", "rules": ["R1"], "confidence": 1.0, "reason": "Plural variant."}
  ],
  "flags": [
    {"tags": ["intergration_", "integration_"], "type": "registry_misspelling", "suggestion": "Rename the registry prefix intergration_ to integration_ and retag.", "rules": ["R8"]},
    {"tags": ["billing_issue"], "type": "unknown_prefix", "suggestion": "Not in the registry. Consider a registry prefix (e.g. issue_billing).", "rules": ["R11"]}
  ]
}
```
