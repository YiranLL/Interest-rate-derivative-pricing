"""Real integration tests: market quotes -> bootstrap -> swap pricing -> DV01.

No monkeypatching or mocked curve builder. Run from the project root:
    python -m pytest tests/test_quote_dv01_integration.py -v

The fixture includes:
    0.5Y deposit, 1Y FRA, 1.5Y analytical swap bootstrap,
    2.5Y numerical swap bootstrap (2Y coupon lies beyond last pillar).
"""

import copy
import math

import pytest

from curves.bootstrap import build_yield_curve
from instruments.swap import VanillaSwap
from pricing.swap_pricer import par_swap_rate, price_swap
from risk.dv01 import parallel_quote_dv01

BP = 1e-4
NOTIONAL = 1_000_000


@pytest.fixture
def market_quotes():
    return [
        {"type": "deposit", "maturity": 0.5, "rate": 0.0300, "alpha": 0.5},
        {"type": "fra", "start": 0.5, "maturity": 1.0, "rate": 0.0320, "alpha": 0.5},
        {
            "type": "swap", "maturity": 1.5, "rate": 0.0340,
            "payment_times": [0.5, 1.0, 1.5],
            "accrual_fractions": [0.5, 0.5, 0.5],
        },
        {
            "type": "swap", "maturity": 2.5, "rate": 0.0360,
            "payment_times": [0.5, 1.0, 1.5, 2.0, 2.5],
            "accrual_fractions": [0.5] * 5,
        },
    ]


@pytest.fixture
def off_market_swap():
    # Different coupon schedule from either quoted swap, to avoid a tautological
    # par-swap repricing check when measuring quote sensitivity.
    return VanillaSwap(
        notional=NOTIONAL,
        fixed_rate=0.033,
        payment_times=[0.5, 1.5, 2.5],
        accrual_fractions=[0.5, 1.0, 1.0],
        direction="payer",
    )


def bumped_quotes(quotes, amount, index=None):
    """Independent test-fixture helper; never alters quotes."""
    result = copy.deepcopy(quotes)
    indexes = range(len(result)) if index is None else [index]
    for i in indexes:
        result[i]["rate"] += amount
    return result


def test_real_bootstrap_deposit_fra_values(market_quotes):
    curve = build_yield_curve(market_quotes)
    p05 = 1.0 / (1.0 + 0.0300 * 0.5)
    p10 = p05 / (1.0 + 0.0320 * 0.5)
    assert curve.discount_factor(0.0) == pytest.approx(1.0)
    assert curve.discount_factor(0.5) == pytest.approx(p05, rel=1e-12)
    assert curve.discount_factor(1.0) == pytest.approx(p10, rel=1e-12)
    assert list(curve.maturities) == pytest.approx([0.0, 0.5, 1.0, 1.5, 2.5])


def test_analytical_swap_pillar_matches_closed_form(market_quotes):
    curve = build_yield_curve(market_quotes)
    k = market_quotes[2]["rate"]
    p05 = curve.discount_factor(0.5)
    p10 = curve.discount_factor(1.0)
    expected = (1 - k * 0.5 * (p05 + p10)) / (1 + k * 0.5)
    assert curve.discount_factor(1.5) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("quote_index", [2, 3], ids=["analytical_swap", "numerical_swap"])
def test_both_swap_market_quotes_are_repriced(market_quotes, quote_index):
    curve = build_yield_curve(market_quotes)
    q = market_quotes[quote_index]
    par_swap = VanillaSwap(
        notional=NOTIONAL, fixed_rate=q["rate"],
        payment_times=q["payment_times"],
        accrual_fractions=q["accrual_fractions"], direction="payer",
    )
    assert par_swap_rate(par_swap, curve) == pytest.approx(q["rate"], abs=1e-10)
    assert price_swap(par_swap, curve) == pytest.approx(0.0, abs=1e-5)


def test_numerical_swap_intermediate_df_is_interpolated(market_quotes):
    curve = build_yield_curve(market_quotes)
    # The 2Y payment date is between the 1.5Y and 2.5Y pillars.
    p15, p25 = curve.discount_factor(1.5), curve.discount_factor(2.5)
    expected = math.sqrt(p15 * p25)  # log-linear midpoint
    assert curve.discount_factor(2.0) == pytest.approx(expected, rel=1e-12)


def test_real_quote_dv01_matches_manual_rebootstrap(market_quotes, off_market_swap):
    up_curve = build_yield_curve(bumped_quotes(market_quotes, BP))
    down_curve = build_yield_curve(bumped_quotes(market_quotes, -BP))
    expected = (price_swap(off_market_swap, up_curve)
                - price_swap(off_market_swap, down_curve)) / 2.0
    actual = parallel_quote_dv01(off_market_swap, market_quotes)
    assert actual == pytest.approx(expected, rel=1e-9, abs=1e-7)
    assert math.isfinite(actual)
    assert abs(actual) > 1e-5


def test_parallel_quote_dv01_has_expected_positive_sign(market_quotes, off_market_swap):
    # With these realistic positive rates and this payer contract, +rate
    # increases value. This is specific to this fixture, not a universal claim.
    assert parallel_quote_dv01(off_market_swap, market_quotes) > 0


def test_payer_receiver_symmetry(market_quotes, off_market_swap):
    receiver = VanillaSwap(
        notional=off_market_swap.notional,
        fixed_rate=off_market_swap.fixed_rate,
        payment_times=off_market_swap.payment_times,
        accrual_fractions=off_market_swap.accrual_fractions,
        direction="receiver",
    )
    assert parallel_quote_dv01(receiver, market_quotes) == pytest.approx(
        -parallel_quote_dv01(off_market_swap, market_quotes), abs=1e-7
    )


def test_notional_scaling(market_quotes, off_market_swap):
    doubled = VanillaSwap(
        notional=2 * off_market_swap.notional,
        fixed_rate=off_market_swap.fixed_rate,
        payment_times=off_market_swap.payment_times,
        accrual_fractions=off_market_swap.accrual_fractions,
        direction="payer",
    )
    assert parallel_quote_dv01(doubled, market_quotes) == pytest.approx(
        2 * parallel_quote_dv01(off_market_swap, market_quotes), rel=1e-9
    )


@pytest.mark.parametrize("bump", [0.5e-4, 1e-4, 2e-4])
def test_dv01_normalised_per_bp(market_quotes, off_market_swap, bump):
    reference = parallel_quote_dv01(off_market_swap, market_quotes, bump=1e-4)
    actual = parallel_quote_dv01(off_market_swap, market_quotes, bump=bump)
    # Second-order central-difference discretisation error is O(bump**2).
    assert actual == pytest.approx(reference, rel=1e-4, abs=1e-3)


def test_market_quotes_and_contract_not_mutated(market_quotes, off_market_swap):
    original_quotes = copy.deepcopy(market_quotes)
    original_contract = vars(off_market_swap).copy()
    parallel_quote_dv01(off_market_swap, market_quotes)
    assert market_quotes == original_quotes
    assert vars(off_market_swap) == original_contract


@pytest.mark.parametrize("quote_index", [0, 1, 2, 3], ids=["deposit", "fra", "analytical_swap", "numerical_swap"])
def test_each_quote_moves_its_own_pillar(market_quotes, quote_index):
    original = build_yield_curve(market_quotes)
    bumped = build_yield_curve(bumped_quotes(market_quotes, BP, quote_index))
    maturity = market_quotes[quote_index]["maturity"]
    assert bumped.discount_factor(maturity) != pytest.approx(
        original.discount_factor(maturity), abs=1e-9
    )
    # Increasing a positive instrument quote lowers its own pillar DF
    # in this calibration fixture.
    assert bumped.discount_factor(maturity) < original.discount_factor(maturity)


def test_quote_order_not_required(market_quotes, off_market_swap):
    # build_yield_curve sorts maturities internally.
    reordered = list(reversed(copy.deepcopy(market_quotes)))
    expected = parallel_quote_dv01(off_market_swap, market_quotes)
    assert parallel_quote_dv01(off_market_swap, reordered) == pytest.approx(
        expected, rel=1e-10, abs=1e-7
    )


def test_zero_and_up_down_bumped_curves_reprice_all_swap_quotes(market_quotes):
    for amount in [0.0, BP, -BP]:
        bumped = bumped_quotes(market_quotes, amount)
        curve = build_yield_curve(bumped)
        for q in bumped:
            if q["type"] != "swap":
                continue
            swap = VanillaSwap(
                notional=NOTIONAL, fixed_rate=q["rate"],
                payment_times=q["payment_times"],
                accrual_fractions=q["accrual_fractions"],
                direction="payer",
            )
            assert price_swap(swap, curve) == pytest.approx(0.0, abs=1e-5)
