import math

import pytest

from curves.yield_curve import YieldCurve
from instruments.swap import VanillaSwap
from pricing.swap_pricer import price_swap
from risk.dv01 import bump_zero_curve, parallel_zero_dv01

BP = 1e-4


@pytest.fixture
def curve():
    """Flat 4% continuously compounded zero-rate curve."""
    maturities = [0.0, 1.0, 2.0, 3.0, 5.0]
    discount_factors = [math.exp(-0.04 * t) for t in maturities]
    return YieldCurve(maturities, discount_factors)


@pytest.fixture
def payer_swap():
    return VanillaSwap(
        notional=1_000_000,
        fixed_rate=0.04,
        payment_times=[1.0, 2.0, 3.0, 4.0, 5.0],
        accrual_fractions=[1.0] * 5,
        direction="payer",
    )


def make_swap(*, direction="payer", notional=1_000_000, fixed_rate=0.04):
    return VanillaSwap(
        notional=notional,
        fixed_rate=fixed_rate,
        payment_times=[1.0, 2.0, 3.0, 4.0, 5.0],
        accrual_fractions=[1.0] * 5,
        direction=direction,
    )


def analytical_parallel_zero_dv01(swap, curve):
    """Independent derivative of the spot-starting single-curve swap PV.

    For a payer swap:
        dV/dh = N [T_n P(0,T_n) + K sum_i alpha_i T_i P(0,T_i)]
    where h is a parallel shift in continuously compounded zero rates.
    """
    last_time = swap.payment_times[-1]
    floating_term = last_time * curve.discount_factor(last_time)
    fixed_term = swap.fixed_rate * sum(
        alpha * t * curve.discount_factor(t)
        for t, alpha in zip(swap.payment_times, swap.accrual_fractions)
    )
    payer_dv01 = BP * swap.notional * (floating_term + fixed_term)
    return payer_dv01 if swap.direction == "payer" else -payer_dv01


# ---------------------------------------------------------------------------
# 1. Curve bump tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bump", [BP, -BP, 5e-5, -2e-4, 0.0])
def test_bump_zero_curve_matches_exact_formula(curve, bump):
    shifted = bump_zero_curve(curve, bump)
    for t in curve.maturities:
        expected = curve.discount_factor(t) * math.exp(-bump * t)
        assert shifted.discount_factor(t) == pytest.approx(expected, rel=1e-12)


def test_bump_preserves_time_zero(curve):
    shifted = bump_zero_curve(curve, BP)
    assert shifted.discount_factor(0.0) == pytest.approx(1.0)


def test_bump_does_not_mutate_original_curve(curve):
    before_maturities = tuple(curve.maturities)
    before_dfs = tuple(curve.discount_factors)
    shifted = bump_zero_curve(curve, BP)
    assert shifted is not curve
    assert tuple(curve.maturities) == before_maturities
    assert tuple(curve.discount_factors) == before_dfs


def test_up_bump_reduces_discount_factors(curve):
    shifted_up = bump_zero_curve(curve, BP)
    shifted_down = bump_zero_curve(curve, -BP)
    for t in [1.0, 2.0, 3.0, 5.0]:
        assert shifted_up.discount_factor(t) < curve.discount_factor(t)
        assert shifted_down.discount_factor(t) > curve.discount_factor(t)


def test_bump_applies_between_pillars_with_log_linear_interpolation(curve):
    t = 4.0  # Not a pillar: between years 3 and 5
    shifted = bump_zero_curve(curve, BP)
    expected = curve.discount_factor(t) * math.exp(-BP * t)
    assert shifted.discount_factor(t) == pytest.approx(expected, rel=1e-12)


def test_zero_bump_produces_equivalent_curve(curve):
    shifted = bump_zero_curve(curve, 0.0)
    assert shifted is not curve
    for t in [0.0, 0.5, 1.0, 2.5, 4.5, 5.0]:
        assert shifted.discount_factor(t) == pytest.approx(curve.discount_factor(t))


# ---------------------------------------------------------------------------
# 2. DV01: independent financial and numerical checks
# ---------------------------------------------------------------------------


def test_payer_dv01_matches_analytical_derivative(payer_swap, curve):
    actual = parallel_zero_dv01(payer_swap, curve)
    expected = analytical_parallel_zero_dv01(payer_swap, curve)
    assert actual == pytest.approx(expected, rel=1e-7)


def test_receiver_dv01_matches_analytical_derivative(curve):
    receiver = make_swap(direction="receiver")
    actual = parallel_zero_dv01(receiver, curve)
    expected = analytical_parallel_zero_dv01(receiver, curve)
    assert actual == pytest.approx(expected, rel=1e-7)


def test_payer_receiver_dv01_are_opposites(curve):
    payer = make_swap(direction="payer")
    receiver = make_swap(direction="receiver")
    assert parallel_zero_dv01(payer, curve) == pytest.approx(
        -parallel_zero_dv01(receiver, curve), rel=1e-10, abs=1e-8
    )


def test_dv01_scales_linearly_with_notional(curve):
    small = make_swap(notional=1_000_000)
    large = make_swap(notional=2_000_000)
    assert parallel_zero_dv01(large, curve) == pytest.approx(
        2.0 * parallel_zero_dv01(small, curve), rel=1e-10
    )


@pytest.mark.parametrize("bump", [5e-5, 1e-4, 2e-4])
def test_dv01_is_normalised_to_one_basis_point(curve, payer_swap, bump):
    expected = analytical_parallel_zero_dv01(payer_swap, curve)
    assert parallel_zero_dv01(payer_swap, curve, bump=bump) == pytest.approx(
        expected, rel=5e-7
    )


def test_payer_dv01_positive_with_nonnegative_fixed_rate(curve, payer_swap):
    assert parallel_zero_dv01(payer_swap, curve) > 0


def test_extremely_negative_fixed_rate_can_give_negative_payer_dv01():
    zero_curve = YieldCurve([0.0, 1.0, 2.0], [1.0, 1.0, 1.0])
    swap = VanillaSwap(
        notional=1_000_000,
        fixed_rate=-0.8,
        payment_times=[1.0, 2.0],
        accrual_fractions=[1.0, 1.0],
        direction="payer",
    )
    actual = parallel_zero_dv01(swap, zero_curve)
    expected = BP * 1_000_000 * (2.0 - 0.8 * (1.0 + 2.0))
    assert actual < 0
    assert actual == pytest.approx(expected, rel=1e-7)


def test_dv01_equals_symmetric_repricing_difference_for_one_bp(curve, payer_swap):
    up_pv = price_swap(payer_swap, bump_zero_curve(curve, BP))
    down_pv = price_swap(payer_swap, bump_zero_curve(curve, -BP))
    assert parallel_zero_dv01(payer_swap, curve) == pytest.approx(
        (up_pv - down_pv) / 2.0, rel=1e-10
    )


# ---------------------------------------------------------------------------
# 3. Input validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_bump", [float("nan"), float("inf"), -float("inf")])
def test_curve_bump_rejects_nonfinite_values(curve, bad_bump):
    with pytest.raises(ValueError):
        bump_zero_curve(curve, bad_bump)


@pytest.mark.parametrize("bad_bump", [None, "0.0001", True])
def test_curve_bump_rejects_non_numeric_values(curve, bad_bump):
    with pytest.raises(TypeError):
        bump_zero_curve(curve, bad_bump)


@pytest.mark.parametrize("bad_bump", [0.0, -BP, float("nan"), float("inf")])
def test_dv01_rejects_invalid_step(curve, payer_swap, bad_bump):
    with pytest.raises(ValueError):
        parallel_zero_dv01(payer_swap, curve, bump=bad_bump)


@pytest.mark.parametrize("bad_bump", [None, "0.0001", True])
def test_dv01_rejects_non_numeric_step(curve, payer_swap, bad_bump):
    with pytest.raises(TypeError):
        parallel_zero_dv01(payer_swap, curve, bump=bad_bump)


def test_curve_bump_rejects_wrong_curve_type():
    with pytest.raises(TypeError):
        bump_zero_curve(None, BP)


def test_dv01_rejects_wrong_swap_type(curve):
    with pytest.raises(TypeError):
        parallel_zero_dv01(None, curve)


def test_dv01_rejects_wrong_curve_type(payer_swap):
    with pytest.raises(TypeError):
        parallel_zero_dv01(payer_swap, None)
