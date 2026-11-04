
import pytest

from curves.yield_curve import YieldCurve
from instruments.swap import VanillaSwap
from pricing.swap_pricer import price_swap, par_swap_rate


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def curve():
    return YieldCurve(
        maturities=[0.0, 1.0, 2.0, 3.0],
        discount_factors=[1.0, 0.96, 0.92, 0.88]
    )


@pytest.fixture
def swap():
    return VanillaSwap(
        notional=1_000_000,
        fixed_rate=0.04,
        payment_times=[1.0, 2.0],
        accrual_fractions=[1.0, 1.0],
        direction="payer"
    )


# ============================================================
# 1. Basic pricing
# ============================================================

def test_payer_swap_price(curve, swap):
    """
    Payer = receive floating, pay fixed.
    """

    annuity = 0.96 + 0.92
    fixed_pv = 1_000_000 * 0.04 * annuity
    floating_pv = 1_000_000 * (1.0 - 0.92)

    expected = floating_pv - fixed_pv

    assert expected == pytest.approx(4_800.0)
    assert price_swap(swap, curve) == pytest.approx(
        expected, abs=1e-8
    )


def test_receiver_swap_price(curve):
    swap = VanillaSwap(
        notional=1_000_000,
        fixed_rate=0.04,
        payment_times=[1.0, 2.0],
        accrual_fractions=[1.0, 1.0],
        direction="receiver"
    )

    assert price_swap(swap, curve) == pytest.approx(
        -4_800.0, abs=1e-8
    )


def test_payer_receiver_symmetry(curve):
    params = dict(
        notional=1_000_000,
        fixed_rate=0.04,
        payment_times=[1.0, 2.0],
        accrual_fractions=[1.0, 1.0]
    )

    payer = VanillaSwap(**params, direction="payer")
    receiver = VanillaSwap(**params, direction="receiver")

    assert price_swap(payer, curve) == pytest.approx(
        -price_swap(receiver, curve), abs=1e-8
    )


# ============================================================
# 2. Par swap rate
# ============================================================

def test_par_swap_rate(curve, swap):
    expected = (1.0 - 0.92) / (0.96 + 0.92)

    assert par_swap_rate(swap, curve) == pytest.approx(
        expected, abs=1e-12
    )


def test_par_swap_has_zero_value(curve, swap):
    fair_rate = par_swap_rate(swap, curve)

    par_swap = VanillaSwap(
        notional=swap.notional,
        fixed_rate=fair_rate,
        payment_times=swap.payment_times,
        accrual_fractions=swap.accrual_fractions,
        direction="payer"
    )

    assert price_swap(par_swap, curve) == pytest.approx(
        0.0, abs=1e-8
    )


def test_par_rate_independent_of_contract_rate(curve):
    swap1 = VanillaSwap(
        1_000_000, 0.02, [1.0, 2.0], [1.0, 1.0], "payer"
    )
    swap2 = VanillaSwap(
        1_000_000, 0.08, [1.0, 2.0], [1.0, 1.0], "payer"
    )

    assert par_swap_rate(swap1, curve) == pytest.approx(
        par_swap_rate(swap2, curve), abs=1e-12
    )


# ============================================================
# 3. Equivalent valuation formula
# ============================================================

def test_payer_value_equals_annuity_times_rate_difference(curve, swap):
    """
    V_payer = N * A * (S - K)
    """

    annuity = sum(
        alpha * curve.discount_factor(t)
        for t, alpha in zip(
            swap.payment_times, swap.accrual_fractions
        )
    )

    market_rate = par_swap_rate(swap, curve)
    expected = swap.notional * annuity * (
        market_rate - swap.fixed_rate
    )

    assert price_swap(swap, curve) == pytest.approx(
        expected, abs=1e-8
    )


# ============================================================
# 4. Sensitivity to contract terms
# ============================================================

def test_notional_scaling(curve, swap):
    larger_swap = VanillaSwap(
        notional=2_000_000,
        fixed_rate=swap.fixed_rate,
        payment_times=swap.payment_times,
        accrual_fractions=swap.accrual_fractions,
        direction="payer"
    )

    assert price_swap(larger_swap, curve) == pytest.approx(
        2 * price_swap(swap, curve), abs=1e-8
    )


def test_higher_fixed_rate_reduces_payer_value(curve):
    low_fixed = VanillaSwap(
        1_000_000, 0.03, [1.0, 2.0], [1.0, 1.0], "payer"
    )
    high_fixed = VanillaSwap(
        1_000_000, 0.05, [1.0, 2.0], [1.0, 1.0], "payer"
    )

    assert price_swap(low_fixed, curve) > price_swap(
        high_fixed, curve
    )


# ============================================================
# 5. Interpolation and irregular schedules
# ============================================================

def test_swap_with_interpolated_payment_dates(curve):
    swap = VanillaSwap(
        notional=1_000_000,
        fixed_rate=0.04,
        payment_times=[0.5, 1.5, 2.5],
        accrual_fractions=[0.5, 1.0, 1.0],
        direction="payer"
    )

    annuity = sum(
        alpha * curve.discount_factor(t)
        for t, alpha in zip(
            swap.payment_times, swap.accrual_fractions
        )
    )

    expected = swap.notional * (
        1.0 - curve.discount_factor(2.5)
        - swap.fixed_rate * annuity
    )

    assert price_swap(swap, curve) == pytest.approx(
        expected, abs=1e-8
    )


# ============================================================
# 6. Negative interest rates
# ============================================================

def test_swap_with_negative_rates():
    negative_curve = YieldCurve(
        maturities=[0.0, 1.0, 2.0],
        discount_factors=[1.0, 1.01, 1.02]
    )

    swap = VanillaSwap(
        notional=1_000_000,
        fixed_rate=-0.005,
        payment_times=[1.0, 2.0],
        accrual_fractions=[1.0, 1.0],
        direction="payer"
    )

    annuity = 1.01 + 1.02

    expected = 1_000_000 * (
        1.0 - 1.02 - (-0.005) * annuity
    )

    assert price_swap(swap, negative_curve) == pytest.approx(
        expected, abs=1e-8
    )

    assert par_swap_rate(swap, negative_curve) < 0


# ============================================================
# 7. Single-payment swap
# ============================================================

def test_single_payment_swap(curve):
    swap = VanillaSwap(
        notional=1_000_000,
        fixed_rate=0.04,
        payment_times=[1.0],
        accrual_fractions=[1.0],
        direction="payer"
    )

    expected = 1_000_000 * (
        1.0 - 0.96 - 0.04 * 0.96
    )

    assert price_swap(swap, curve) == pytest.approx(
        expected, abs=1e-8
    )


# ============================================================
# 8. Input validation
# ============================================================

@pytest.mark.parametrize(
    "overrides",
    [
        {"notional": 0},
        {"notional": -100},
        {"fixed_rate": float("nan")},
        {"payment_times": []},
        {"payment_times": [2.0, 1.0]},
        {"payment_times": [0.0, 1.0]},
        {"accrual_fractions": [1.0]},
        {"accrual_fractions": [1.0, -1.0]},
        {"direction": "receive"},
    ]
)
def test_invalid_swap_contract(overrides):
    params = dict(
        notional=1_000_000,
        fixed_rate=0.04,
        payment_times=[1.0, 2.0],
        accrual_fractions=[1.0, 1.0],
        direction="payer"
    )

    params.update(overrides)

    with pytest.raises(ValueError):
        VanillaSwap(**params)


# ============================================================
# 9. Curve range
# ============================================================

def test_swap_maturity_outside_curve_range(curve):
    swap = VanillaSwap(
        notional=1_000_000,
        fixed_rate=0.04,
        payment_times=[1.0, 4.0],
        accrual_fractions=[1.0, 3.0],
        direction="payer"
    )

    with pytest.raises(ValueError):
        price_swap(swap, curve)
