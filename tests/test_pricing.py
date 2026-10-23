import pytest
from decision_engine import classify, pricing
from test_decision_engine import FakeClient, model_reply

LISTING = [
    {
        "id": "claude-opus-5.5",
        "pricing": {
            "input_per_million": "2.897228",
            "output_per_million": "14.486140",
            "cache_read_input_per_million": "0.144861402000000000000000",
            "cache_write_input_per_million": "3.621535050000000000000000",
        },
    },
    {
        "id": "jev-1.13.0",
        "pricing": {"input_per_million": "0.0357", "output_per_million": "0"},
    },
    {"id": "no-price", "pricing": {}},
    {"id": "half-price", "pricing": {"input_per_million": "1"}},
]


def test_parse_listing_reads_prices_and_skips_incomplete_ones():
    prices = pricing.parse_listing(LISTING)
    assert set(prices) == {"claude-opus-5.5", "jev-1.13.0"}
    opus = prices["claude-opus-5.5"]
    assert (opus.input, opus.output) == (2.897228, 14.48614)
    assert opus.cache_read == pytest.approx(0.144861402)
    assert opus.source == "live"


def test_a_price_costs_every_kind_of_token():
    price = pricing.Price(input=4, output=20, cache_read=0.2, cache_write=5)
    usage = {
        "input_tokens": 1_000,
        "output_tokens": 500,
        "cache_read_input_tokens": 4_000,
        "cache_creation_input_tokens": 200,
    }
    assert price.cost(usage) == pytest.approx(0.004 + 0.01 + 0.0008 + 0.001)
    # Without cache prices, cached tokens cost the input price.
    assert pricing.Price(input=4, output=20).cost(usage) == pytest.approx(
        0.004 + 0.01 + 0.016 + 0.0008
    )


def test_calls_prefer_the_billed_charge_and_estimate_the_rest():
    prices = {"m": pricing.Price(input=1, output=2)}
    billed = {"model": "m", "usage": {"input_tokens": 10**6}, "billed_usd": "0.5"}
    estimated = {"model": "m", "usage": {"output_tokens": 10**6}}
    assert pricing.calls_cost([billed], prices) == {
        "cost_usd": 0.5,
        "cost_basis": "billed",
    }
    assert pricing.calls_cost([estimated], prices) == {
        "cost_usd": 2.0,
        "cost_basis": "estimated",
    }
    assert pricing.calls_cost([billed, estimated], prices) == {
        "cost_usd": 2.5,
        "cost_basis": "partly estimated",
    }
    unpriced = {"model": "other", "usage": {"input_tokens": 5}}
    assert pricing.calls_cost([billed, unpriced], prices) == {
        "cost_usd": None,
        "cost_basis": None,
    }


def test_prices_round_trip_and_format():
    price = pricing.Price(input=1.5, output=3, source="entered")
    assert pricing.Price.from_dict(price.to_dict()) == price
    assert pricing.format_usd(None) == "unknown"
    assert pricing.format_usd(0) == "$0"
    assert pricing.format_usd(12.345) == "$12.35"
    assert pricing.format_usd(0.000123) == "$0.000123"


def test_the_engine_records_the_billed_charge(monkeypatch):
    client = FakeClient(model_reply("flag_solve"))
    original = client._create

    def with_billing(**kwargs):
        response = original(**kwargs)
        response.billing = {"billed_cost_usd": "0.004321"}
        return response

    client.messages.create = with_billing
    monkeypatch.setattr(classify, "get_client", lambda: client)
    result = classify.classify_ticket("p", config=classify.EngineConfig("m"))
    assert result["audit"]["billed_cost_usd"] == "0.004321"
    assert result["model_calls"][0]["billed_usd"] == "0.004321"


def test_live_prices_reads_the_list_fresh(monkeypatch):
    from decision_engine import decisions

    decisions.reset()
    reads = []

    class Client:
        def get(self, path, cast_to=None):
            reads.append(path)
            return {"data": LISTING}

    try:
        prices = pricing.live_prices(Client(), ["claude-opus-5.5", "no-price"])
        pricing.live_prices(Client(), ["claude-opus-5.5"])
    finally:
        decisions.reset()
    assert set(prices) == {"claude-opus-5.5"} and len(reads) == 2


def test_a_call_with_no_usage_and_no_charge_has_an_unknown_cost():
    prices = {"m": pricing.Price(input=1, output=2)}
    failed = {"model": "m", "usage": {}, "latency_ms": 5}
    assert pricing.calls_cost([failed], prices) == {
        "cost_usd": None,
        "cost_basis": None,
    }
    billed = {"model": "m", "usage": {}, "billed_usd": "0"}
    assert pricing.calls_cost([billed], prices)["cost_usd"] == 0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0])
def test_prices_must_be_finite_and_not_negative(bad):
    with pytest.raises(ValueError, match="finite"):
        pricing.Price(input=bad, output=1)
    listing = [
        {
            "id": "m",
            "pricing": {"input_per_million": str(bad), "output_per_million": "1"},
        }
    ]
    assert pricing.parse_listing(listing) == {}
