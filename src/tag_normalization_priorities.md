# Priorities for this task

These come after the reference above and take precedence over it where the two differ.

The goal is discoverability. Tickets about the same concept should carry the same tag, so
they can be found, filtered and counted together instead of being spread across several
tags that differ only in wording. Lexical variation between tags that mean the same thing
(different words for one concept, abbreviations, word order, product-name variants,
"issue"/"problem"/"error" style suffixes) is the problem to solve. Formatting is not.

0. **Zendesk AI tags are out of scope.** Tags starting with `intent` (`intent__...`,
   `intent_confidence__...`) are written by Zendesk's AI. They are left out of the list you
   receive. Never map a tag to one, never propose one as a canonical tag, and never coin a
   tag starting with `intent`.
1. **Make the decisions.** When two or more tags in the list describe the same concept,
   merge them, even when they share no words. Decide clear and likely cases yourself as
   `merge` decisions. Raise a flag only where a support lead could reasonably argue either
   way; "not sure the names match" is not a reason to flag when the concepts match.
2. **A new tag is allowed.** If none of the existing names is the best name for the shared
   concept, coin one: lowercase snake_case, no spaces. It can follow a registry prefix
   (a context tag) or be a static tag, whichever describes the concept best. Map every
   member of the group, including the existing names, to it.
3. **Don't rename for formatting alone.** A tag that shares its meaning with no other tag in
   the list keeps its exact spelling: its `__`, hyphens, trailing `_` and all. In
   particular, the `__` in facet__value tags (`sentiment__positive`,
   `language_confidence__high`) is their real form, not a
   separator variant. Surface-form changes (R1) apply only when two or more input tags are
   variants of each other.
4. **Keep the reference's distinctions that change meaning:**
   - Protected automated tags are never renamed (R3); other tags may merge into them.
   - Different values of one facet stay apart: `sentiment__positive` vs
     `sentiment__negative`, `language_confidence__low` vs `language_confidence__high`.
   - Different products, teams and escalation targets stay apart (R5, R7):
     `troubleshooting_items_api` vs `troubleshooting_data_api`, `data_api` vs
     `reports_api`, `csm_escalated` vs `product_escalation`.
5. **Merge toward where people search.** These replace R6, and R7 as it applies to `uses_*`
   tags:
   - A more specific tag merges into its general tag when people looking for that concept
     would search for the general one: `nvda_screenreader` into `screen_readers`,
     `consumer_key_access` into `consumer_key`, `mcq_values` into `mcq`. Keep it apart only
     when the specific part is itself something people search or report on separately.
   - A `uses_X` tag merges into `X` when both name the same product: `uses_data_api` into
     `data_api`, `uses_items_api` into `items_api`. The product is the concept; whether the
     ticket "uses" it is not a separate one.
   - The merge goes toward the general or product tag, which becomes canonical (or a new
     tag naming that concept, per item 2).
   Different products under one prefix still stay apart (item 4): `uses_data_api` merges
   into `data_api`, not into `reports_api`.
6. **Confidence** is how sure you are that the tags name one concept. Give merges you are
   making 0.8 or more; anything lower is treated as a flag.
