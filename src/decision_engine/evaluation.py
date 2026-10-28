"""Evaluation against hand-labelled tickets, reported as a confusion matrix.

An evaluation set is a JSON list of {"ticket": <tickets document>,
"expected": <classification>, "note": optional}. Build one by exporting a
sample with export_labelling_sample, filling in each "expected", and keeping
the file out of git if it holds real customer text.

    python -m decision_engine.evaluation sample OUT.json [N]   # needs MONGO_URI
    python -m decision_engine.evaluation balanced OUT.json [PER_CATEGORY]
    python -m decision_engine.evaluation agree A.json B.json   # two labellers
    python -m decision_engine.evaluation run EVAL_SET.json     # paid model calls

A random sample rarely holds a leadership case, so `balanced` picks up to
PER_CATEGORY (default 15) reviewable tickets per category the engine last
gave them, and every leadership case. `agree` compares two people's labels
of the same tickets: where they disagree, the taxonomy is unclear.
"""

import json
import random
import sys
from collections import Counter
from typing import Callable, Dict, Iterable, List, Optional

from pymongo.collection import Collection

from .classify import classify_ticket, EngineError
from .guards import CLASSIFICATIONS
from .payload import build_ticket_payload

ERROR_LABEL = "engine_error"
# Fields that identify a customer or aren't part of the model's input.
_DROPPED_FIELDS = (
    "_id",
    "raw_xml",
    "requester_id",
    "organization_id",
    "cc_list",
    "customer_id",
    "agent_reviews",
    "outcomes",
    "review_claim",
    "last_engine_error",
    "embedding",
    "embedding_meta",
    "ab_test_verdicts",
)


def load_eval_set(path: str) -> List[dict]:
    with open(path) as f:
        cases = json.load(f)
    unlabelled = [
        i for i, case in enumerate(cases) if case.get("expected") not in CLASSIFICATIONS
    ]
    if unlabelled:
        raise ValueError(
            f"{path}: cases {unlabelled[:10]} have no valid 'expected' classification"
        )
    return cases


def export_labelling_sample(
    tickets: Collection, path: str, size: int = 50, seed: Optional[int] = None
) -> int:
    """Writes a random sample of tickets for hand labelling.

    Identifier fields are dropped, but comment bodies are kept verbatim and may
    still name people: review the file before sharing it.

    Returns:
        The number of tickets written.
    """
    ids = [doc["zendesk_id"] for doc in tickets.find({}, {"zendesk_id": 1})]
    chosen = random.Random(seed).sample(ids, min(size, len(ids)))
    cases = []
    for zendesk_id in chosen:
        doc = tickets.find_one({"zendesk_id": zendesk_id})
        for field in _DROPPED_FIELDS:
            doc.pop(field, None)
        cases.append({"ticket": doc, "expected": None, "note": ""})
    with open(path, "w") as f:
        json.dump(cases, f, indent=2, default=str)
    return len(cases)


PER_CATEGORY = 15


def _labelling_case(tickets: Collection, zendesk_id: str) -> dict:
    doc = tickets.find_one({"zendesk_id": zendesk_id})
    for field in _DROPPED_FIELDS:
        doc.pop(field, None)
    return {"ticket": doc, "expected": None, "note": ""}


def export_balanced_sample(
    tickets: Collection,
    path: str,
    per_category: int = PER_CATEGORY,
    seed: Optional[int] = None,
) -> Dict[str, int]:
    """Writes a labelling sample balanced across categories.

    Tickets are grouped by their latest review's classification; only
    reviewable ones (not Solved or Closed) with a review are used. Up to
    per_category come from each group, and every escalate_leadership one.
    The engine's answer is left out (agent_reviews is dropped), so it can't
    sway the labeller; cases are shuffled for the same reason.

    Returns:
        How many tickets were taken from each category.
    """
    from .persistence import REVIEWABLE

    groups: Dict[str, List[str]] = {c: [] for c in CLASSIFICATIONS}
    query = {"agent_reviews.0": {"$exists": True}, **REVIEWABLE}
    for doc in tickets.find(query, {"zendesk_id": 1, "agent_reviews": 1}):
        latest = doc["agent_reviews"][-1].get("classification")
        if latest in groups:
            groups[latest].append(doc["zendesk_id"])
    rng = random.Random(seed)
    chosen: Dict[str, List[str]] = {}
    for category, ids in groups.items():
        ids.sort()
        take = len(ids) if category == "escalate_leadership" else per_category
        chosen[category] = rng.sample(ids, min(take, len(ids)))
    cases = [_labelling_case(tickets, i) for ids in chosen.values() for i in ids]
    rng.shuffle(cases)
    with open(path, "w") as f:
        json.dump(cases, f, indent=2, default=str)
    return {category: len(ids) for category, ids in chosen.items()}


def agreement(cases_a: Iterable[dict], cases_b: Iterable[dict]) -> dict:
    """How far two labellers agree on the tickets both labelled.

    Returns:
        {"compared", "agreement" (share of equal labels), "kappa" (Cohen's,
        agreement beyond chance; None when undefined), "disagreements":
        [{"zendesk_id", "a", "b"}]}.
    """

    def labels(cases):
        return {
            str(case["ticket"]["zendesk_id"]): case.get("expected")
            for case in cases
            if case.get("expected") in CLASSIFICATIONS
        }

    a, b = labels(cases_a), labels(cases_b)
    shared = sorted(set(a) & set(b))
    if not shared:
        return {"compared": 0, "agreement": None, "kappa": None, "disagreements": []}
    observed = sum(a[t] == b[t] for t in shared) / len(shared)
    count_a, count_b = Counter(a[t] for t in shared), Counter(b[t] for t in shared)
    expected = sum(count_a[c] * count_b[c] for c in CLASSIFICATIONS) / len(shared) ** 2
    return {
        "compared": len(shared),
        "agreement": observed,
        "kappa": None if expected == 1 else (observed - expected) / (1 - expected),
        "disagreements": [
            {"zendesk_id": t, "a": a[t], "b": b[t]} for t in shared if a[t] != b[t]
        ],
    }


def format_agreement(result: dict) -> str:
    if not result["compared"]:
        return "No ticket was labelled in both files."
    kappa = "n/a" if result["kappa"] is None else f"{result['kappa']:.2f}"
    lines = [
        f"{result['compared']} tickets labelled by both: "
        f"{result['agreement']:.0%} agree (Cohen's kappa {kappa})."
    ]
    lines += [f"  {d['zendesk_id']}: {d['a']} vs {d['b']}" for d in result["disagreements"]]
    return "\n".join(lines)


def evaluate(
    cases: Iterable[dict], classify: Callable[[str], dict] = classify_ticket
) -> dict:
    """Classifies each case and tallies expected vs predicted.

    Engine errors are counted under the "engine_error" prediction rather than
    stopping the run.

    Returns:
        {"matrix": {expected: Counter(predicted)}, "per_category": {...},
         "accuracy": float, "total": int, "mistakes": [...]}.
    """
    matrix: Dict[str, Counter] = {c: Counter() for c in CLASSIFICATIONS}
    mistakes = []
    for case in cases:
        ticket = case["ticket"]
        try:
            predicted = classify(build_ticket_payload(ticket))["classification"]
        except EngineError as e:
            predicted, reason = ERROR_LABEL, str(e)
        else:
            reason = ""
        matrix[case["expected"]][predicted] += 1
        if predicted != case["expected"]:
            mistakes.append(
                {
                    "ticket_id": ticket.get("zendesk_id"),
                    "expected": case["expected"],
                    "predicted": predicted,
                    "error": reason,
                }
            )

    total = sum(sum(row.values()) for row in matrix.values())
    correct = sum(matrix[c][c] for c in CLASSIFICATIONS)
    per_category = {}
    for c in CLASSIFICATIONS:
        predicted_as_c = sum(matrix[e][c] for e in CLASSIFICATIONS)
        actual_c = sum(matrix[c].values())
        per_category[c] = {
            "support": actual_c,
            "precision": matrix[c][c] / predicted_as_c if predicted_as_c else None,
            "recall": matrix[c][c] / actual_c if actual_c else None,
        }
    return {
        "matrix": matrix,
        "per_category": per_category,
        "accuracy": correct / total if total else None,
        "total": total,
        "mistakes": mistakes,
    }


def format_report(report: dict) -> str:
    columns = list(CLASSIFICATIONS) + [ERROR_LABEL]
    short = {
        c: c.replace("escalate_", "esc_").replace("escalation_", "esc_")[:13]
        for c in columns
    }
    lines = [
        "expected \\ predicted".ljust(22) + "".join(short[c].rjust(14) for c in columns)
    ]
    for expected in CLASSIFICATIONS:
        row = report["matrix"][expected]
        lines.append(
            expected.ljust(22) + "".join(str(row[c]).rjust(14) for c in columns)
        )
    lines.append("")
    for c, stats in report["per_category"].items():
        fmt = lambda v: f"{v:.0%}" if v is not None else "-"
        lines.append(
            f"{c:<22} support {stats['support']:>3}  precision {fmt(stats['precision']):>4}  "
            f"recall {fmt(stats['recall']):>4}"
        )
    if report["accuracy"] is not None:
        lines.append(
            f"\nOverall accuracy: {report['accuracy']:.0%} of {report['total']}"
        )
    return "\n".join(lines)


def main(argv: List[str]) -> int:
    if len(argv) >= 2 and argv[0] == "run":
        print(format_report(evaluate(load_eval_set(argv[1]))))
        return 0
    if len(argv) >= 2 and argv[0] == "sample":
        from .persistence import get_tickets_collection

        count = export_labelling_sample(
            get_tickets_collection(), argv[1], int(argv[2]) if len(argv) > 2 else 50
        )
        print(f"Wrote {count} tickets to {argv[1]}; fill in each 'expected'.")
        return 0
    if len(argv) >= 2 and argv[0] == "balanced":
        from .persistence import get_tickets_collection

        counts = export_balanced_sample(
            get_tickets_collection(),
            argv[1],
            int(argv[2]) if len(argv) > 2 else PER_CATEGORY,
        )
        shown = ", ".join(f"{c} {n}" for c, n in counts.items())
        print(f"Wrote {sum(counts.values())} tickets to {argv[1]} ({shown}).")
        return 0
    if len(argv) >= 3 and argv[0] == "agree":
        # Partly labelled files are fine: only tickets both labelled count.
        with open(argv[1]) as a, open(argv[2]) as b:
            print(format_agreement(agreement(json.load(a), json.load(b))))
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
