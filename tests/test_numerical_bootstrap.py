
import math

import pytest

from curves.yield_curve import YieldCurve
from curves.bootstrap import bootstrap_swap, bootstrap_swap_numerical


def test_numerical_matches_analytical():
    """Both methods should produce the same DF when all earlier coupon DFs are known."""

    curve = YieldCurve(
        maturities=[0.0, 1.0, 2.0],
        discount_factors=[1.0, 0.96, 0.92]
    )

    par_rate = 0.04
    payment_times = [1.0, 2.0, 3.0]
    accrual_fractions = [1.0, 1.0, 1.0]

    expected = bootstrap_swap(
        par_rate=par_rate,
        known_dfs=[0.96, 0.92],
        accrual_fractions=accrual_fractions
    )

    actual = bootstrap_swap_numerical(
        par_rate, curve, payment_times, accrual_fractions
    )

    assert actual == pytest.approx(expected, abs=1e-10)


def test_intermediate_coupon_interpolation():
    """Numerical bootstrap must solve for unknown intermediate coupon DFs."""

    curve = YieldCurve(
        maturities=[0.0, 1.0, 2.0],
        discount_factors=[1.0, 0.96, 0.92]
    )

    par_rate = 0.04
    payment_times = [1.0, 1.5, 2.0, 2.5, 3.0]
    accrual_fractions = [0.5] * 5

    final_df = bootstrap_swap_numerical(
        par_rate, curve, payment_times, accrual_fractions
    )

    assert math.isfinite(final_df)
    assert final_df > 0

    extended_curve = YieldCurve(
        [*curve.maturities, 3.0],
        [*curve.discount_factors, final_df]
    )

    # Check interpolation at 2.5Y.
    expected_2_5_df = math.sqrt(0.92 * final_df)

    assert extended_curve.discount_factor(2.5) == pytest.approx(
        expected_2_5_df, abs=1e-10
    )

    # Check the par swap equation.
    annuity = sum(
        alpha * extended_curve.discount_factor(t)
        for t, alpha in zip(payment_times, accrual_fractions)
    )

    fixed_leg_pv = par_rate * annuity
    floating_leg_pv = 1.0 - final_df

    assert fixed_leg_pv == pytest.approx(
        floating_leg_pv, abs=1e-10
    )


@pytest.mark.parametrize(
    "par_rate",
    [0.01, 0.03, 0.05, -0.005]
)
def test_repricing_across_rates(par_rate):
    """The bootstrapped curve should reproduce positive and negative par rates."""

    curve = YieldCurve(
        [0.0, 1.0, 2.0],
        [1.0, 0.98, 0.96]
    )

    payment_times = [1.0, 2.0, 2.5, 3.0]
    accrual_fractions = [1.0, 1.0, 0.5, 0.5]

    final_df = bootstrap_swap_numerical(
        par_rate, curve, payment_times, accrual_fractions
    )

    extended_curve = YieldCurve(
        [*curve.maturities, 3.0],
        [*curve.discount_factors, final_df]
    )

    annuity = sum(
        alpha * extended_curve.discount_factor(t)
        for t, alpha in zip(payment_times, accrual_fractions)
    )

    implied_par_rate = (1.0 - final_df) / annuity

    assert implied_par_rate == pytest.approx(
        par_rate, abs=1e-10
    )


def test_original_curve_is_not_mutated():
    """Root finding should not modify the original curve."""

    curve = YieldCurve(
        [0.0, 1.0, 2.0],
        [1.0, 0.96, 0.92]
    )

    before_maturities = curve.maturities
    before_dfs = curve.discount_factors

    bootstrap_swap_numerical(
        0.04,
        curve,
        [1.0, 2.0, 2.5, 3.0],
        [1.0, 1.0, 0.5, 0.5]
    )

    assert curve.maturities == before_maturities
    assert curve.discount_factors == before_dfs


def test_invalid_accrual_lengths():
    curve = YieldCurve([0.0, 1.0], [1.0, 0.97])

    with pytest.raises(ValueError):
        bootstrap_swap_numerical(
            0.04, curve, [1.0, 2.0], [1.0]
        )


def test_invalid_payment_order():
    curve = YieldCurve([0.0, 1.0], [1.0, 0.97])

    with pytest.raises(ValueError):
        bootstrap_swap_numerical(
            0.04, curve, [1.5, 1.0, 2.0], [0.5, 0.5, 1.0]
        )


def test_no_valid_bracket():
    """An excessively high par rate can put the root outside the fixed bracket."""

    curve = YieldCurve([0.0, 1.0], [1.0, 0.97])

    with pytest.raises(ValueError, match="Root is not bracketed"):
        bootstrap_swap_numerical(
            1_000_000.0,
            curve,
            [1.0, 2.0],
            [1.0, 1.0]
        )
