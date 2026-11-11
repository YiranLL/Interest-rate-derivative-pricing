"""Tests for risk.dv01.parallel_quote_dv01.

These tests use a controlled curve-builder replacement to isolate the quote-bump,
rebootstrap, and swap-repricing workflow. They DO NOT assert that the project's
real build_yield_curve() accepts any particular market-quote schema; add an
integration test using your actual quote fixtures separately.

Run from the project root:
    python -m pytest tests/test_parallel_quote_dv01.py -v
"""

import copy
import math

import pytest

import risk.dv01 as dv01_module
from curves.yield_curve import YieldCurve
from instruments.swap import VanillaSwap
from pricing.swap_pricer import price_swap
from risk.dv01 import parallel_quote_dv01

BP = 1e-4
N = 1_000_000


def make_swap(*, direction="payer", notional=N, fixed_rate=0.04):
    return VanillaSwap(
        notional=notional,
        fixed_rate=fixed_rate,
        payment_times=[1.0, 2.0, 3.0],
        accrual_fractions=[1.0, 1.0, 1.0],
        direction=direction,
    )


@pytest.fixture
def quotes():
    # The stub builder below intentionally accepts only these common fields.
    return [
        {"type": "deposit", "maturity": 1.0, "rate": 0.03},
        {"type": "deposit", "maturity": 2.0, "rate": 0.035},
        {"type": "swap", "maturity": 3.0, "rate": 0.04},
    ]


@pytest.fixture
def controlled_builder(monkeypatch):
    """Replace bootstrap with a deterministic *independent* market-to-DF map.

    Quote at maturity T is interpreted as a continuously compounded zero rate:
        P(0,T) = exp(-rate * T).
    This is not the deposit/FRA/swap bootstrap algorithm; it gives us an exact
    benchmark for testing whether parallel_quote_dv01 bumps quotes correctly.
    """
    calls = []

    def builder(market_quotes):
        calls.append(copy.deepcopy(market_quotes))
        maturities = [0.0] + [q["maturity"] for q in market_quotes]
        dfs = [1.0] + [
            math.exp(-q["rate"] * q["maturity"]) for q in market_quotes
        ]
        return YieldCurve(maturities, dfs)

    monkeypatch.setattr(dv01_module, "build_yield_curve", builder)
    return calls


def reference_dv01(swap, quotes, bump=BP):
    """Independent expected DV01 by explicitly constructing bumped DFs."""
    def curve_for_shift(shift):
        return YieldCurve(
            [0.0] + [q["maturity"] for q in quotes],
            [1.0] + [
                math.exp(-(q["rate"] + shift) * q["maturity"])
                for q in quotes
            ],
        )

    up = price_swap(swap, curve_for_shift(bump))
    down = price_swap(swap, curve_for_shift(-bump))
    return (up - down) / (2 * bump) * BP


def analytical_dv01(swap, quotes):
    """Exact first-order derivative for an all-pillar parallel zero shift.

    Because the controlled builder treats the input rates as continuously
    compounded pillar zero rates, parallel quote bump is parallel zero bump.
    """
    curve = YieldCurve(
        [0.0] + [q["maturity"] for q in quotes],
        [1.0] + [math.exp(-q["rate"] * q["maturity"]) for q in quotes],
    )
    t_last = swap.payment_times[-1]
    derivative = swap.notional * (
        t_last * curve.discount_factor(t_last)
        + swap.fixed_rate * sum(
            alpha * t * curve.discount_factor(t)
            for t, alpha in zip(swap.payment_times, swap.accrual_fractions)
        )
    )
    return BP * derivative * (1 if swap.direction == "payer" else -1)


# ---------- Correctness and rebootstrap workflow ----------

def test_matches_independent_bump_and_reprice_reference(quotes, controlled_builder):
    swap = make_swap()
    actual = parallel_quote_dv01(swap, quotes)
    assert actual == pytest.approx(reference_dv01(swap, quotes), rel=1e-10, abs=1e-8)


def test_matches_analytical_derivative(quotes, controlled_builder):
    actual = parallel_quote_dv01(make_swap(), quotes)
    assert actual == pytest.approx(analytical_dv01(make_swap(), quotes), rel=1e-7, abs=1e-5)


def test_rebootstraps_twice_with_opposite_bumps(quotes, controlled_builder):
    parallel_quote_dv01(make_swap(), quotes)
    calls = controlled_builder
    assert len(calls) == 2
    for original, upper, lower in zip(quotes, calls[0], calls[1]):
        assert upper["rate"] == pytest.approx(original["rate"] + BP)
        assert lower["rate"] == pytest.approx(original["rate"] - BP)
        assert upper["type"] == original["type"] == lower["type"]
        assert upper["maturity"] == original["maturity"] == lower["maturity"]


def test_does_not_mutate_original_quotes(quotes, controlled_builder):
    original = copy.deepcopy(quotes)
    parallel_quote_dv01(make_swap(), quotes)
    assert quotes == original


def test_builder_receives_independent_quote_objects(quotes, monkeypatch):
    ids = []

    def builder(market_quotes):
        ids.append((id(market_quotes), *(id(q) for q in market_quotes)))
        return YieldCurve(
            [0.0] + [q["maturity"] for q in market_quotes],
            [1.0] + [math.exp(-q["rate"] * q["maturity"]) for q in market_quotes],
        )

    monkeypatch.setattr(dv01_module, "build_yield_curve", builder)
    parallel_quote_dv01(make_swap(), quotes)
    assert len(ids) == 2
    assert all(a != b for a, b in zip(ids[0], ids[1]))
    assert ids[0][0] != id(quotes)
    assert ids[1][0] != id(quotes)


# ---------- Financial properties ----------

@pytest.mark.parametrize("fixed_rate", [0.0, 0.02, 0.04, 0.08])
def test_normal_payer_dv01_is_positive(quotes, controlled_builder, fixed_rate):
    assert parallel_quote_dv01(make_swap(fixed_rate=fixed_rate), quotes) > 0


def test_payer_receiver_symmetry(quotes, controlled_builder):
    payer = parallel_quote_dv01(make_swap(direction="payer"), quotes)
    receiver = parallel_quote_dv01(make_swap(direction="receiver"), quotes)
    assert receiver == pytest.approx(-payer, rel=1e-10, abs=1e-8)


def test_notional_scaling(quotes, controlled_builder):
    small = parallel_quote_dv01(make_swap(notional=N), quotes)
    large = parallel_quote_dv01(make_swap(notional=2 * N), quotes)
    assert large == pytest.approx(2 * small, rel=1e-10, abs=1e-8)


@pytest.mark.parametrize("bump", [5e-5, 1e-4, 2e-4])
def test_normalised_to_one_bp(quotes, controlled_builder, bump):
    actual = parallel_quote_dv01(make_swap(), quotes, bump=bump)
    assert actual == pytest.approx(analytical_dv01(make_swap(), quotes), rel=1e-6, abs=1e-4)


def test_negative_fixed_rate_need_not_have_positive_payer_dv01(quotes, controlled_builder):
    # A deliberately extreme negative K establishes the sign is not guaranteed.
    assert parallel_quote_dv01(make_swap(fixed_rate=-1.0), quotes) < 0


# ---------- Input validation ----------

@pytest.mark.parametrize("bad_bump", [0, -BP, float("nan"), float("inf"), -float("inf")])
def test_invalid_bump_values_rejected(quotes, bad_bump):
    with pytest.raises(ValueError):
        parallel_quote_dv01(make_swap(), quotes, bump=bad_bump)


@pytest.mark.parametrize("bad_bump", [True, False, "0.0001", None, object()])
def test_invalid_bump_types_rejected(quotes, bad_bump):
    with pytest.raises(TypeError):
        parallel_quote_dv01(make_swap(), quotes, bump=bad_bump)


def test_invalid_swap_rejected(quotes):
    with pytest.raises(TypeError):
        parallel_quote_dv01("not a swap", quotes)


@pytest.mark.parametrize("bad_quotes", [None, {}, (), "quotes"])
def test_invalid_quote_container_rejected(bad_quotes):
    with pytest.raises(TypeError):
        parallel_quote_dv01(make_swap(), bad_quotes)


def test_empty_quotes_rejected():
    with pytest.raises(ValueError):
        parallel_quote_dv01(make_swap(), [])


def test_quote_entry_must_be_dict(quotes):
    malformed = copy.deepcopy(quotes)
    malformed[0] = "bad quote"
    with pytest.raises(TypeError):
        parallel_quote_dv01(make_swap(), malformed)


@pytest.mark.parametrize("missing_key", ["type", "maturity", "rate"])
def test_missing_required_quote_key_rejected(quotes, missing_key):
    malformed = copy.deepcopy(quotes)
    del malformed[0][missing_key]
    with pytest.raises(ValueError):
        parallel_quote_dv01(make_swap(), malformed)


@pytest.mark.parametrize("bad_rate", [None, "0.03", True, False])
def test_invalid_quote_rate_type_rejected(quotes, bad_rate):
    malformed = copy.deepcopy(quotes)
    malformed[0]["rate"] = bad_rate
    with pytest.raises(TypeError):
        parallel_quote_dv01(make_swap(), malformed)


@pytest.mark.parametrize("bad_rate", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_quote_rate_rejected(quotes, bad_rate):
    malformed = copy.deepcopy(quotes)
    malformed[0]["rate"] = bad_rate
    with pytest.raises(ValueError):
        parallel_quote_dv01(make_swap(), malformed)
