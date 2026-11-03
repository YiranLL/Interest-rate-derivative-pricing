
import math
from copy import deepcopy

import pytest

from curves.yield_curve import YieldCurve
from curves.bootstrap import build_yield_curve


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def market_quotes():
    """
    Deliberately unsorted market quotes.

    Deposit: 0.5Y
    FRA:     0.5Y -> 1Y
    Swap:    2Y, analytical bootstrap
    Swap:    3Y, numerical bootstrap

    The 3Y swap has a coupon at 2.5Y, beyond the
    previously constructed 2Y curve.
    """
    return [
        {
            "type": "swap",
            "maturity": 3.0,
            "rate": 0.041,
            "payment_times": [1.0, 2.0, 2.5, 3.0],
            "accrual_fractions": [1.0, 1.0, 0.5, 0.5],
        },
        {
            "type": "fra",
            "start": 0.5,
            "maturity": 1.0,
            "rate": 0.038,
            "alpha": 0.5,
        },
        {
            "type": "swap",
            "maturity": 2.0,
            "rate": 0.040,
            "payment_times": [1.0, 2.0],
            "accrual_fractions": [1.0, 1.0],
        },
        {
            "type": "deposit",
            "maturity": 0.5,
            "rate": 0.035,
            "alpha": 0.5,
        },
    ]


@pytest.fixture
def curve(market_quotes):
    return build_yield_curve(market_quotes)


# ============================================================
# 1. Full curve construction
# ============================================================

def test_build_returns_yield_curve(curve):
    assert isinstance(curve, YieldCurve)


def test_curve_pillars(curve):
    assert curve.maturities == pytest.approx(
        (0.0, 0.5, 1.0, 2.0, 3.0)
    )

    assert len(curve.discount_factors) == 5
    assert curve.discount_factors[0] == pytest.approx(1.0)


def test_discount_factors_are_finite_and_positive(curve):
    assert all(
        math.isfinite(df) and df > 0
        for df in curve.discount_factors
    )


def test_maturities_are_strictly_increasing(curve):
    assert all(
        curve.maturities[i] < curve.maturities[i + 1]
        for i in range(len(curve.maturities) - 1)
    )


# ============================================================
# 2. Independent expected values
# ============================================================

def test_deposit_and_fra_expected_dfs(curve):
    """
    Independently calculate the first two bootstrapped DFs.
    """

    expected_05 = 1.0 / (1.0 + 0.035 * 0.5)

    expected_10 = expected_05 / (1.0 + 0.038 * 0.5)

    assert curve.discount_factor(0.5) == pytest.approx(
        expected_05, abs=1e-12
    )

    assert curve.discount_factor(1.0) == pytest.approx(
        expected_10, abs=1e-12
    )


def test_analytical_swap_expected_df(curve):
    """
    The 2Y swap should use analytical bootstrapping.
    """

    p1 = 1.0 / (1.0 + 0.035 * 0.5) / (1.0 + 0.038 * 0.5)

    expected_p2 = (1.0 - 0.04 * p1) / 1.04

    assert curve.discount_factor(2.0) == pytest.approx(
        expected_p2, abs=1e-12
    )


# ============================================================
# 3. Market quote repricing
# ============================================================

def test_all_market_quotes_reprice(curve, market_quotes):
    """
    Every market instrument should reproduce its input rate.
    """

    for quote in market_quotes:

        if quote["type"] == "deposit":
            p = curve.discount_factor(quote["maturity"])

            implied_rate = (
                (1.0 / p) - 1.0
            ) / quote["alpha"]

        elif quote["type"] == "fra":
            p_start = curve.discount_factor(quote["start"])
            p_end = curve.discount_factor(quote["maturity"])

            implied_rate = (
                p_start / p_end - 1.0
            ) / quote["alpha"]

        elif quote["type"] == "swap":
            annuity = sum(
                alpha * curve.discount_factor(t)
                for t, alpha in zip(
                    quote["payment_times"],
                    quote["accrual_fractions"]
                )
            )

            implied_rate = (
                1.0 - curve.discount_factor(quote["maturity"])
            ) / annuity

        else:
            pytest.fail("Unexpected instrument type in fixture.")

        assert implied_rate == pytest.approx(
            quote["rate"], abs=1e-10
        ), f"Repricing failed for {quote['type']} at {quote['maturity']}Y"


# ============================================================
# 4. Numerical interpolation
# ============================================================

def test_numerical_swap_intermediate_coupon(curve):
    """
    The 2.5Y coupon requires interpolation between
    the 2Y and newly bootstrapped 3Y pillars.
    """

    p2 = curve.discount_factor(2.0)
    p3 = curve.discount_factor(3.0)

    expected_p25 = math.sqrt(p2 * p3)

    assert curve.discount_factor(2.5) == pytest.approx(
        expected_p25, abs=1e-12
    )


# ============================================================
# 5. Input ordering and mutation
# ============================================================

def test_input_order_does_not_matter(market_quotes):
    curve_a = build_yield_curve(market_quotes)

    sorted_quotes = sorted(
        market_quotes, key=lambda q: q["maturity"]
    )
    curve_b = build_yield_curve(sorted_quotes)

    assert curve_a.maturities == curve_b.maturities

    assert curve_a.discount_factors == pytest.approx(
        curve_b.discount_factors
    )


def test_input_market_quotes_are_not_mutated(market_quotes):
    original = deepcopy(market_quotes)

    build_yield_curve(market_quotes)

    assert market_quotes == original


# ============================================================
# 6. Invalid market quotes
# ============================================================

def test_empty_market_quotes():
    with pytest.raises(ValueError):
        build_yield_curve([])


def test_duplicate_maturities():
    quotes = [
        {
            "type": "deposit",
            "maturity": 0.5,
            "rate": 0.03,
            "alpha": 0.5,
        },
        {
            "type": "deposit",
            "maturity": 0.5,
            "rate": 0.04,
            "alpha": 0.5,
        },
    ]

    with pytest.raises(ValueError):
        build_yield_curve(quotes)


def test_invalid_instrument_type():
    quotes = [
        {
            "type": "bond",
            "maturity": 1.0,
            "rate": 0.04,
        }
    ]

    with pytest.raises(ValueError):
        build_yield_curve(quotes)


def test_swap_maturity_mismatch():
    quotes = [
        {
            "type": "deposit",
            "maturity": 1.0,
            "rate": 0.03,
            "alpha": 1.0,
        },
        {
            "type": "swap",
            "maturity": 2.0,
            "rate": 0.04,
            "payment_times": [1.0, 2.5],
            "accrual_fractions": [1.0, 1.0],
        },
    ]

    with pytest.raises(ValueError):
        build_yield_curve(quotes)


def test_invalid_swap_payment_order():
    quotes = [
        {
            "type": "deposit",
            "maturity": 1.0,
            "rate": 0.03,
            "alpha": 1.0,
        },
        {
            "type": "swap",
            "maturity": 3.0,
            "rate": 0.04,
            "payment_times": [2.0, 1.5, 3.0],
            "accrual_fractions": [1.0, 0.5, 1.5],
        },
    ]

    with pytest.raises(ValueError):
        build_yield_curve(quotes)


# ============================================================
# 7. Single-payment analytical swap
# ============================================================

def test_single_payment_swap():
    """
    A single-payment swap should be handled analytically.
    There are no earlier coupon DFs to retrieve.
    """

    quotes = [
        {
            "type": "swap",
            "maturity": 1.0,
            "rate": 0.04,
            "payment_times": [1.0],
            "accrual_fractions": [1.0],
        }
    ]

    curve = build_yield_curve(quotes)

    assert curve.discount_factor(1.0) == pytest.approx(
        1.0 / 1.04, abs=1e-12
    )
