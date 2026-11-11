"""Integration tests for bucketed market-quote DV01.

Uses the real curve bootstrap and vanilla swap pricer, without mocks.
Run from the project root:
    python -m pytest tests/test_bucketed_quote_dv01.py -v

Market pillars: 0.5Y deposit, 1Y FRA, 1.5Y analytical swap,
2.5Y numerical swap (2Y coupon needs interpolation).
"""

import copy
import math

import pytest

from curves.bootstrap import build_yield_curve
from instruments.swap import VanillaSwap
from pricing.swap_pricer import price_swap
from risk.dv01 import bucketed_quote_dv01, parallel_quote_dv01

BP = 1e-4
NOTIONAL = 1_000_000
KEYS = [
    ("deposit", 0.5),
    ("fra", 1.0),
    ("swap", 1.5),
    ("swap", 2.5),
]


@pytest.fixture
def market_quotes():
    return [
        {"type": "deposit", "maturity": 0.5, "rate": 0.0300, "alpha": 0.5},
        {"type": "fra", "start": 0.5, "maturity": 1.0, "rate": 0.0320, "alpha": 0.5},
        {
            "type": "swap", "maturity": 1.5, "rate": 0.0340,
            "payment_times": [0.5, 1.0, 1.5],
            "accrual_fractions": [0.5] * 3,
        },
        {
            "type": "swap", "maturity": 2.5, "rate": 0.0360,
            "payment_times": [0.5, 1.0, 1.5, 2.0, 2.5],
            "accrual_fractions": [0.5] * 5,
        },
    ]


@pytest.fixture
def payer_swap():
    return VanillaSwap(
        notional=NOTIONAL,
        fixed_rate=0.033,
        payment_times=[0.5, 1.5, 2.5],
        accrual_fractions=[0.5, 1.0, 1.0],
        direction="payer",
    )


def swap_with(swap, *, direction=None, notional=None):
    return VanillaSwap(
        notional=swap.notional if notional is None else notional,
        fixed_rate=swap.fixed_rate,
        payment_times=swap.payment_times,
        accrual_fractions=swap.accrual_fractions,
        direction=swap.direction if direction is None else direction,
    )


def independently_calculated_bucket(swap, quotes, selected_index, bump=BP):
    """Calculate one bucket independently, using real bootstrap + pricer."""
    up_quotes = copy.deepcopy(quotes)
    down_quotes = copy.deepcopy(quotes)
    up_quotes[selected_index]["rate"] += bump
    down_quotes[selected_index]["rate"] -= bump
    up_value = price_swap(swap, build_yield_curve(up_quotes))
    down_value = price_swap(swap, build_yield_curve(down_quotes))
    return (up_value - down_value) * BP / (2 * bump)


def test_result_has_one_entry_per_market_quote(market_quotes, payer_swap):
    result = bucketed_quote_dv01(payer_swap, market_quotes)
    assert isinstance(result, dict)
    assert list(result) == KEYS
    assert all(isinstance(v, (int, float)) and math.isfinite(v)
               for v in result.values())


@pytest.mark.parametrize("index", range(4), ids=[
    "deposit", "fra", "analytical_swap", "numerical_swap"
])
def test_each_bucket_matches_independent_rebootstrap(market_quotes, payer_swap, index):
    actual = bucketed_quote_dv01(payer_swap, market_quotes)[KEYS[index]]
    expected = independently_calculated_bucket(payer_swap, market_quotes, index)
    assert actual == pytest.approx(expected, rel=1e-9, abs=1e-7)


def test_buckets_sum_to_parallel_quote_dv01(market_quotes, payer_swap):
    """Directional derivative identity; small O(bump**2) discretisation error."""
    buckets = bucketed_quote_dv01(payer_swap, market_quotes)
    parallel = parallel_quote_dv01(payer_swap, market_quotes)
    assert sum(buckets.values()) == pytest.approx(parallel, rel=1e-4, abs=1e-3)


def test_payer_receiver_symmetry_per_bucket(market_quotes, payer_swap):
    payer = bucketed_quote_dv01(payer_swap, market_quotes)
    receiver = bucketed_quote_dv01(
        swap_with(payer_swap, direction="receiver"), market_quotes
    )
    for key in KEYS:
        assert payer[key] == pytest.approx(-receiver[key], rel=1e-9, abs=1e-7)


@pytest.mark.parametrize("factor", [0.5, 2.0, 10.0])
def test_notional_scales_every_bucket(market_quotes, payer_swap, factor):
    base = bucketed_quote_dv01(payer_swap, market_quotes)
    scaled = bucketed_quote_dv01(
        swap_with(payer_swap, notional=NOTIONAL * factor), market_quotes
    )
    for key in KEYS:
        assert scaled[key] == pytest.approx(factor * base[key], rel=1e-9, abs=1e-6)


@pytest.mark.parametrize("bump", [0.5e-4, 1e-4, 2e-4])
def test_normalised_to_one_basis_point(market_quotes, payer_swap, bump):
    reference = bucketed_quote_dv01(payer_swap, market_quotes, bump=BP)
    current = bucketed_quote_dv01(payer_swap, market_quotes, bump=bump)
    for key in KEYS:
        assert current[key] == pytest.approx(reference[key], rel=1e-4, abs=1e-3)


def test_original_quotes_and_swap_unchanged(market_quotes, payer_swap):
    initial_quotes = copy.deepcopy(market_quotes)
    initial_swap = copy.deepcopy(vars(payer_swap))
    bucketed_quote_dv01(payer_swap, market_quotes)
    assert market_quotes == initial_quotes
    assert vars(payer_swap) == initial_swap


def test_result_independent_of_input_quote_order(market_quotes, payer_swap):
    reordered = list(reversed(copy.deepcopy(market_quotes)))
    original = bucketed_quote_dv01(payer_swap, market_quotes)
    result = bucketed_quote_dv01(payer_swap, reordered)
    assert set(result) == set(original)
    for key in KEYS:
        assert result[key] == pytest.approx(original[key], rel=1e-9, abs=1e-7)


def test_shorter_swap_has_zero_exposure_to_later_quote(market_quotes):
    """2.5Y quote cannot affect a swap finishing at 1.5Y in sequential bootstrap."""
    short_swap = VanillaSwap(
        notional=NOTIONAL, fixed_rate=0.033,
        payment_times=[0.5, 1.0, 1.5],
        accrual_fractions=[0.5, 0.5, 0.5], direction="payer",
    )
    result = bucketed_quote_dv01(short_swap, market_quotes)
    assert result[("swap", 2.5)] == pytest.approx(0.0, abs=1e-8)


def test_single_quote_deposit_bucket_matches_manual_formula():
    """Independently benchmark one bucket without calling bootstrap for expected value."""
    quotes = [{"type": "deposit", "maturity": 0.5, "rate": 0.03, "alpha": 0.5}]
    swap = VanillaSwap(
        notional=NOTIONAL, fixed_rate=0.04,
        payment_times=[0.5], accrual_fractions=[0.5], direction="payer",
    )
    def manual_pv(rate):
        df = 1 / (1 + rate * 0.5)
        return NOTIONAL * (1 - df - 0.04 * 0.5 * df)
    expected = (manual_pv(0.03 + BP) - manual_pv(0.03 - BP)) / 2
    result = bucketed_quote_dv01(swap, quotes)
    assert result[("deposit", 0.5)] == pytest.approx(expected, rel=1e-9, abs=1e-7)


@pytest.mark.parametrize("bad_bump", [0, -BP, float("nan"), float("inf"), -float("inf")])
def test_invalid_bump_value(market_quotes, payer_swap, bad_bump):
    with pytest.raises(ValueError):
        bucketed_quote_dv01(payer_swap, market_quotes, bump=bad_bump)


@pytest.mark.parametrize("bad_bump", [True, False, "0.0001", None, []])
def test_invalid_bump_type(market_quotes, payer_swap, bad_bump):
    with pytest.raises(TypeError):
        bucketed_quote_dv01(payer_swap, market_quotes, bump=bad_bump)


def test_invalid_swap(market_quotes):
    with pytest.raises(TypeError):
        bucketed_quote_dv01(None, market_quotes)


@pytest.mark.parametrize("bad_quotes", [None, {}, (), "bad", 42])
def test_market_quotes_must_be_list(bad_quotes, payer_swap):
    with pytest.raises(TypeError):
        bucketed_quote_dv01(payer_swap, bad_quotes)


def test_empty_market_quotes(payer_swap):
    with pytest.raises(ValueError):
        bucketed_quote_dv01(payer_swap, [])


def test_market_quote_must_be_dict(market_quotes, payer_swap):
    market_quotes[1] = 42
    with pytest.raises(TypeError):
        bucketed_quote_dv01(payer_swap, market_quotes)


@pytest.mark.parametrize("missing", ["type", "maturity", "rate"])
def test_missing_required_quote_field(market_quotes, payer_swap, missing):
    del market_quotes[0][missing]
    with pytest.raises(ValueError):
        bucketed_quote_dv01(payer_swap, market_quotes)


def test_invalid_quote_type(market_quotes, payer_swap):
    market_quotes[0]["type"] = "future"
    with pytest.raises(ValueError):
        bucketed_quote_dv01(payer_swap, market_quotes)


@pytest.mark.parametrize("field,bad_value,exc", [
    ("maturity", 0, ValueError),
    ("maturity", -1, ValueError),
    ("maturity", float("inf"), ValueError),
    ("maturity", True, TypeError),
    ("maturity", "0.5", TypeError),
    ("rate", float("nan"), ValueError),
    ("rate", float("inf"), ValueError),
    ("rate", True, TypeError),
    ("rate", "0.03", TypeError),
])
def test_invalid_quote_fields(market_quotes, payer_swap, field, bad_value, exc):
    market_quotes[0][field] = bad_value
    with pytest.raises(exc):
        bucketed_quote_dv01(payer_swap, market_quotes)


def test_duplicate_bucket_identifier_rejected(market_quotes, payer_swap):
    # Duplicate (type, maturity) keys would silently overwrite a dict result.
    market_quotes.append(copy.deepcopy(market_quotes[0]))
    with pytest.raises(ValueError, match="Duplicate"):
        bucketed_quote_dv01(payer_swap, market_quotes)
