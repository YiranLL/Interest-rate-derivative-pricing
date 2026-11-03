
import pytest

from curves.yield_curve import YieldCurve
from instruments.fra import FRA
from pricing.fra_pricer import price_fra


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def curve():
    return YieldCurve(
        maturities=[0.0, 1.0, 1.5, 2.0],
        discount_factors=[1.0, 0.97, 0.95, 0.92]
    )


@pytest.fixture
def fra():
    return FRA(
        notional=1_000_000,
        start_time=1.0,
        end_time=1.5,
        fixed_rate=0.04,
        accrual_fraction=0.5,
        direction="receive"
    )


# ============================================================
# 1. Basic pricing
# ============================================================

def test_fra_price_against_manual_calculation(curve, fra):
    forward = (0.97 / 0.95 - 1.0) / 0.5

    expected = (
        1_000_000 * 0.5 * 0.95 * (forward - 0.04)
    )

    assert price_fra(fra, curve) == pytest.approx(
        expected, abs=1e-8
    )


def test_fra_equivalent_pricing_formula(curve, fra):
    """
    V = N * [P(0,T1) - (1 + alpha*K)*P(0,T2)]
    """

    p1 = curve.discount_factor(fra.start_time)
    p2 = curve.discount_factor(fra.end_time)

    expected = fra.notional * (
        p1 - (1 + fra.accrual_fraction * fra.fixed_rate) * p2
    )

    assert price_fra(fra, curve) == pytest.approx(
        expected, abs=1e-8
    )


# ============================================================
# 2. Fair rate
# ============================================================

def test_fair_rate_fra_has_zero_value(curve):
    forward = curve.forward_rate(1.0, 1.5, 0.5)

    fra = FRA(
        notional=1_000_000,
        start_time=1.0,
        end_time=1.5,
        fixed_rate=forward,
        accrual_fraction=0.5,
        direction="receive"
    )

    assert price_fra(fra, curve) == pytest.approx(
        0.0, abs=1e-8
    )


# ============================================================
# 3. Direction and payoff signs
# ============================================================

def test_receive_and_pay_are_opposites(curve):
    params = dict(
        notional=1_000_000,
        start_time=1.0,
        end_time=1.5,
        fixed_rate=0.04,
        accrual_fraction=0.5
    )

    receive_fra = FRA(**params, direction="receive")
    pay_fra = FRA(**params, direction="pay")

    assert price_fra(receive_fra, curve) == pytest.approx(
        -price_fra(pay_fra, curve), abs=1e-8
    )


@pytest.mark.parametrize(
    "fixed_rate, expected_sign",
    [
        (0.02, 1),
        (0.06, -1),
    ]
)
def test_receive_floating_value_sign(curve, fixed_rate, expected_sign):
    fra = FRA(
        notional=1_000_000,
        start_time=1.0,
        end_time=1.5,
        fixed_rate=fixed_rate,
        accrual_fraction=0.5,
        direction="receive"
    )

    assert price_fra(fra, curve) * expected_sign > 0


# ============================================================
# 4. Notional scaling
# ============================================================

def test_notional_scaling(curve, fra):
    larger_fra = FRA(
        notional=2 * fra.notional,
        start_time=fra.start_time,
        end_time=fra.end_time,
        fixed_rate=fra.fixed_rate,
        accrual_fraction=fra.accrual_fraction,
        direction=fra.direction
    )

    assert price_fra(larger_fra, curve) == pytest.approx(
        2 * price_fra(fra, curve), abs=1e-8
    )


# ============================================================
# 5. Interpolation
# ============================================================

def test_fra_pricing_with_interpolated_dates(curve):
    """
    1.25Y and 1.75Y are not curve pillars.
    The pricer must use interpolated DFs.
    """

    fra = FRA(
        notional=1_000_000,
        start_time=1.25,
        end_time=1.75,
        fixed_rate=0.04,
        accrual_fraction=0.5,
        direction="receive"
    )

    p1 = curve.discount_factor(1.25)
    p2 = curve.discount_factor(1.75)

    expected = fra.notional * (
        p1 - (1 + 0.5 * 0.04) * p2
    )

    assert price_fra(fra, curve) == pytest.approx(
        expected, abs=1e-8
    )


# ============================================================
# 6. Negative interest rates
# ============================================================

def test_negative_forward_rate():
    curve = YieldCurve(
        maturities=[0.0, 1.0, 2.0],
        discount_factors=[1.0, 1.01, 1.02]
    )

    fra = FRA(
        notional=1_000_000,
        start_time=1.0,
        end_time=2.0,
        fixed_rate=0.0,
        accrual_fraction=1.0,
        direction="receive"
    )

    assert curve.forward_rate(1.0, 2.0, 1.0) < 0
    assert price_fra(fra, curve) < 0


# ============================================================
# 7. Instrument validation
# ============================================================

@pytest.mark.parametrize(
    "overrides",
    [
        {"notional": 0},
        {"notional": -1_000},
        {"start_time": -0.5},
        {"start_time": 2.0, "end_time": 1.0},
        {"accrual_fraction": 0},
        {"accrual_fraction": -0.5},
        {"direction": "invalid"},
    ]
)
def test_invalid_fra_contract(overrides):
    params = dict(
        notional=1_000_000,
        start_time=1.0,
        end_time=1.5,
        fixed_rate=0.04,
        accrual_fraction=0.5,
        direction="receive"
    )

    params.update(overrides)

    with pytest.raises(ValueError):
        FRA(**params)


def test_equal_start_and_end_times_rejected():
    """
    A FRA must have a strictly positive accrual period.
    """

    with pytest.raises(ValueError):
        FRA(
            notional=1_000_000,
            start_time=1.0,
            end_time=1.0,
            fixed_rate=0.04,
            accrual_fraction=0.5,
            direction="receive"
        )


# ============================================================
# 8. Curve range
# ============================================================

def test_fra_maturity_outside_curve_range(curve):
    fra = FRA(
        notional=1_000_000,
        start_time=1.5,
        end_time=3.0,
        fixed_rate=0.04,
        accrual_fraction=1.5,
        direction="receive"
    )

    with pytest.raises(ValueError):
        price_fra(fra, curve)
