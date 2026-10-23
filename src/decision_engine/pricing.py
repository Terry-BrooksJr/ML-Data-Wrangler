"""What a model call costs, in US dollars.

Cheaper Inference reports each call's actual charge in its reply
(`billing.billed_cost_usd`, after its discount); classify and decisions keep
it in the call's audit as "billed_usd". When a call has none (it failed
before a reply, say), its cost is estimated from its token usage and the
model's prices per million tokens.

Prices come from GET /v1/models (`pricing`: the discounted
input/output/cache prices Cheaper Inference charges now). An A/B test
saves the prices it was created with, after the reviewer has had the
chance to correct them, so its estimates don't move when prices do.
"""

import math
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Optional

from . import decisions

# The pricing fields of GET /v1/models, and the usage keys they price.
_LISTED = {
    "input": "input_per_million",
    "output": "output_per_million",
    "cache_read": "cache_read_input_per_million",
    "cache_write": "cache_write_input_per_million",
}
_USAGE = {
    "input": "input_tokens",
    "output": "output_tokens",
    "cache_read": "cache_read_input_tokens",
    "cache_write": "cache_creation_input_tokens",
}
# How a cost was arrived at, shown beside it.
BILLED, ESTIMATED, PARTLY_ESTIMATED = "billed", "estimated", "partly estimated"


@dataclass(frozen=True)
class Price:
    """US dollars per million tokens. Cache prices default to the input price.

    Attributes:
        source: "live" (read from Cheaper Inference) or "entered" (typed in).
    """

    input: float
    output: float
    cache_read: Optional[float] = None
    cache_write: Optional[float] = None
    source: str = "live"

    def to_dict(self) -> dict:
        return asdict(self)

    def __post_init__(self):
        for name in ("input", "output", "cache_read", "cache_write"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(
                    f"The {name} price must be a finite number of 0 or more, got {value}."
                )

    @classmethod
    def from_dict(cls, values: dict) -> "Price":
        return cls(
            input=float(values["input"]),
            output=float(values["output"]),
            cache_read=_number(values.get("cache_read")),
            cache_write=_number(values.get("cache_write")),
            source=values.get("source") or "entered",
        )

    def cost(self, usage: dict) -> float:
        """What usage costs at these prices."""
        rates = {
            "input": self.input,
            "output": self.output,
            "cache_read": self.input if self.cache_read is None else self.cache_read,
            "cache_write": self.input if self.cache_write is None else self.cache_write,
        }
        return sum(
            (usage.get(_USAGE[kind]) or 0) * rate / 1_000_000
            for kind, rate in rates.items()
        )


def _number(value) -> Optional[float]:
    """value as a finite float, or None (missing, not a number, nan or inf)."""
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def parse_listing(models: Iterable[dict]) -> Dict[str, Price]:
    """Model ID -> Price for every listed model with valid input and output
    prices (finite and not negative)."""
    prices = {}
    for model in models:
        pricing = model.get("pricing") or {}
        values = {kind: _number(pricing.get(field)) for kind, field in _LISTED.items()}
        if values["input"] is None or values["output"] is None:
            continue
        try:
            prices[model["id"]] = Price(**values)
        except ValueError:  # a negative price: leave the model unpriced
            continue
    return prices


def live_prices(client, models: Iterable[str]) -> Dict[str, Price]:
    """Today's prices for the given models (read fresh); models without a
    listed price are left out."""
    listed = parse_listing(decisions.model_list(client, refresh=True))
    return {m: listed[m] for m in models if m in listed}


def billed(audit: Optional[dict]) -> Optional[float]:
    """The charge a reply reported, from its audit, or None."""
    return _number((audit or {}).get("billed_cost_usd"))


def calls_cost(calls: Iterable[dict], prices: Dict[str, Price]) -> dict:
    """The cost of the model calls behind one answer.

    Each call counts its billed charge if the reply gave one, else an
    estimate from prices and its token usage. Returns {"cost_usd": float, or
    None when a call has no charge and either no price or no usage,
    "cost_basis": BILLED, ESTIMATED or PARTLY_ESTIMATED}.
    """
    total, bases = 0.0, set()
    for call in calls:
        charge = _number(call.get("billed_usd"))
        if charge is not None:
            bases.add(BILLED)
        else:
            price = prices.get(call.get("model"))
            usage = call.get("usage") or {}
            # A call with no usage and no charge (a failed request, say) may
            # still have been billed: its cost is unknown, not zero.
            if price is None or not any(usage.get(key) for key in _USAGE.values()):
                return {"cost_usd": None, "cost_basis": None}
            charge = price.cost(usage)
            bases.add(ESTIMATED)
        total += charge
    basis = (
        PARTLY_ESTIMATED
        if len(bases) > 1
        else next(iter(bases), BILLED)  # no calls: nothing was charged
    )
    return {"cost_usd": total, "cost_basis": basis}


def format_usd(amount: Optional[float]) -> str:
    """$1.23, or $0.000123 for the tiny amounts single calls cost."""
    if amount is None:
        return "unknown"
    if amount == 0:
        return "$0"
    if abs(amount) >= 0.01:
        return f"${amount:,.2f}"
    return f"${amount:.6f}".rstrip("0")
