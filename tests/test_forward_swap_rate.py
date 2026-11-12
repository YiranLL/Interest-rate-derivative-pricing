"""Tests for pricing.swap_pricer.forward_swap_rate.

Run from the project root:
    python -m pytest tests/test_forward_swap_rate.py -v

These tests use the real YieldCurve, VanillaSwap and pricing implementations.
"""
import copy
import math

import pytest

from curves.yield_curve import YieldCurve
from instruments.swap import VanillaSwap
from pricing.swap_pricer import forward_swap_rate, par_swap_rate, price_swap


@pytest.fixture
def curve():
    return YieldCurve([0.0, 1.0, 2.0, 3.0, 4.0], [1.0, 0.97, 0.94, 0.90, 0.86])


def test_hand_calculated_two_year_forward(curve):
    actual = forward_swap_rate(curve, 1.0, [2.0, 3.0], [1.0, 1.0])
    assert actual == pytest.approx((0.97 - 0.90) / (0.94 + 0.90), abs=1e-13)


def test_nonannual_accrual_fractions(curve):
    expected = (0.97 - 0.90) / (0.5 * 0.94 + 0.75 * 0.90)
    assert forward_swap_rate(curve, 1.0, [2.0, 3.0], [0.5, 0.75]) == pytest.approx(expected)


def test_single_payment_forward_swap(curve):
    assert forward_swap_rate(curve, 1.0, [2.0], [1.0]) == pytest.approx((0.97 - 0.94) / 0.94)


def test_spot_swap_equals_existing_par_swap_rate(curve):
    times = [1.0, 2.0, 3.0]
    alphas = [1.0, 1.0, 1.0]
    swap = VanillaSwap(1_000_000, 0.04, times, alphas, 'payer')
    assert forward_swap_rate(curve, 0.0, times, alphas) == pytest.approx(
        par_swap_rate(swap, curve), abs=1e-13
    )


@pytest.mark.parametrize('direction', ['payer', 'receiver'])
def test_spot_par_swap_has_zero_value(curve, direction):
    times = [1.0, 2.0, 3.0]
    alphas = [1.0, 1.0, 1.0]
    rate = forward_swap_rate(curve, 0.0, times, alphas)
    swap = VanillaSwap(1_000_000, rate, times, alphas, direction)
    assert price_swap(swap, curve) == pytest.approx(0.0, abs=1e-8)


def test_forward_start_between_pillars(curve):
    # Log-linear DF interpolation: P(0, 1.5) = sqrt(P(0,1)*P(0,2)).
    start_df = math.sqrt(0.97 * 0.94)
    expected = (start_df - 0.90) / (0.94 + 0.90)
    assert forward_swap_rate(curve, 1.5, [2.0, 3.0], [1.0, 1.0]) == pytest.approx(expected)


def test_interpolated_payment_date(curve):
    df_25 = math.sqrt(0.94 * 0.90)
    expected = (0.97 - 0.90) / (0.5 * df_25 + 0.5 * 0.90)
    assert forward_swap_rate(curve, 1.0, [2.5, 3.0], [0.5, 0.5]) == pytest.approx(expected)


def test_flat_continuous_zero_curve():
    r = 0.035
    pillars = [0.0, 1.0, 2.0, 3.0, 4.0]
    flat_curve = YieldCurve(pillars, [math.exp(-r*t) for t in pillars])
    times = [2.0, 3.0, 4.0]
    expected = (math.exp(-r) - math.exp(-4*r)) / sum(math.exp(-r*t) for t in times)
    assert forward_swap_rate(flat_curve, 1.0, times, [1.0]*3) == pytest.approx(expected)


def test_flat_zero_rate_curve():
    flat_curve = YieldCurve([0.0, 1.0, 2.0, 3.0], [1.0]*4)
    assert forward_swap_rate(flat_curve, 1.0, [2.0, 3.0], [1.0, 1.0]) == pytest.approx(0.0)


def test_negative_forward_rate_from_increasing_dfs():
    negative_curve = YieldCurve([0.0, 1.0, 2.0, 3.0], [1.0, 1.01, 1.025, 1.04])
    actual = forward_swap_rate(negative_curve, 1.0, [2.0, 3.0], [1.0, 1.0])
    assert actual == pytest.approx((1.01 - 1.04) / (1.025 + 1.04))
    assert actual < 0


def test_curve_not_modified(curve):
    before_maturities = tuple(curve.maturities)
    before_dfs = tuple(curve.discount_factors)
    forward_swap_rate(curve, 1.0, [2.0, 3.0], [1.0, 1.0])
    assert tuple(curve.maturities) == before_maturities
    assert tuple(curve.discount_factors) == before_dfs


def test_payment_and_accrual_inputs_not_modified(curve):
    times = [2.0, 3.0]
    alphas = [1.0, 1.0]
    original_times, original_alphas = copy.deepcopy(times), copy.deepcopy(alphas)
    forward_swap_rate(curve, 1.0, times, alphas)
    assert times == original_times
    assert alphas == original_alphas


def test_accepts_tuple_schedules(curve):
    assert forward_swap_rate(curve, 1.0, (2.0, 3.0), (1.0, 1.0)) == pytest.approx(
        (0.97 - 0.90) / (0.94 + 0.90)
    )


@pytest.mark.parametrize('bad_curve', [None, 'curve', 4, {}, []])
def test_curve_type_validation(bad_curve):
    with pytest.raises(TypeError):
        forward_swap_rate(bad_curve, 1.0, [2.0], [1.0])


@pytest.mark.parametrize('bad_start', [True, False, '1', None, [], {}])
def test_start_time_type_validation(curve, bad_start):
    with pytest.raises(TypeError):
        forward_swap_rate(curve, bad_start, [2.0], [1.0])


@pytest.mark.parametrize('bad_start', [-0.1, float('nan'), float('inf'), -float('inf')])
def test_start_time_value_validation(curve, bad_start):
    with pytest.raises(ValueError):
        forward_swap_rate(curve, bad_start, [2.0], [1.0])


@pytest.mark.parametrize('bad_times', [None, '2', 2.0, {}, [], [1.0, 1.0], [3.0, 2.0], [1.0, 2.0]])
def test_payment_schedule_invalid(curve, bad_times):
    with pytest.raises((TypeError, ValueError)):
        forward_swap_rate(curve, 1.0, bad_times, [1.0] * len(bad_times) if isinstance(bad_times, (list, tuple)) else [1.0])


@pytest.mark.parametrize('bad_time', [True, False, '2', None, 0.0, -1.0, float('nan'), float('inf')])
def test_payment_time_elements_invalid(curve, bad_time):
    with pytest.raises((TypeError, ValueError)):
        forward_swap_rate(curve, 1.0, [bad_time], [1.0])


def test_last_payment_cannot_exceed_curve(curve):
    with pytest.raises(ValueError):
        forward_swap_rate(curve, 1.0, [5.0], [1.0])


@pytest.mark.parametrize('bad_alphas', [None, 1.0, '1', {}, [], [1.0, 1.0]])
def test_accrual_schedule_invalid(curve, bad_alphas):
    with pytest.raises((TypeError, ValueError)):
        forward_swap_rate(curve, 1.0, [2.0], bad_alphas)


@pytest.mark.parametrize('bad_alpha', [True, False, '1', None, 0.0, -1.0, float('nan'), float('inf')])
def test_accrual_element_invalid(curve, bad_alpha):
    with pytest.raises((TypeError, ValueError)):
        forward_swap_rate(curve, 1.0, [2.0], [bad_alpha])
