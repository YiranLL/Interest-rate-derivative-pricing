
import math
import pytest

from curves.interpolation import log_linear_df


# 1. Interpolation at the midpoint
def test_midpoint_interpolation():
    result = log_linear_df(
        T=1.5,
        T_left=1.0,
        T_right=2.0,
        P_left=0.96,
        P_right=0.90
    )

    expected = math.sqrt(0.96 * 0.90)

    assert result == pytest.approx(expected)


# 2. Interpolation at an arbitrary point
def test_arbitrary_interpolation():
    result = log_linear_df(
        T=1.25,
        T_left=1.0,
        T_right=2.0,
        P_left=0.96,
        P_right=0.90
    )

    expected = 0.96 ** 0.75 * 0.90 ** 0.25

    assert result == pytest.approx(expected)


# 3. Interpolation at the endpoints
@pytest.mark.parametrize("T, expected", [
    (1.0, 0.96),
    (2.0, 0.90)
])
def test_interpolation_endpoints(T, expected):
    result = log_linear_df(T, 1.0, 2.0, 0.96, 0.90)

    assert result == pytest.approx(expected)


# 4. Invalid interpolation intervals
@pytest.mark.parametrize("T_left, T_right", [
    (2.0, 1.0),
    (1.0, 1.0)
])
def test_invalid_interval(T_left, T_right):
    with pytest.raises(ValueError):
        log_linear_df(1.5, T_left, T_right, 0.96, 0.90)


# 5. Invalid discount factors
@pytest.mark.parametrize("P_left, P_right", [
    (0.0, 0.90),
    (-0.96, 0.90),
    (0.96, 0.0),
    (0.96, -0.90)
])
def test_invalid_discount_factors(P_left, P_right):
    with pytest.raises(ValueError):
        log_linear_df(1.5, 1.0, 2.0, P_left, P_right)


# 6. Extrapolation is not allowed
@pytest.mark.parametrize("T", [
    0.5,
    2.5
])
def test_extrapolation(T):
    with pytest.raises(ValueError):
        log_linear_df(T, 1.0, 2.0, 0.96, 0.90)


# 7. Log of interpolated DF is linear in maturity
def test_log_linearity():
    P_left = 0.96
    P_right = 0.90

    P_25 = log_linear_df(1.25, 1.0, 2.0, P_left, P_right)
    P_50 = log_linear_df(1.50, 1.0, 2.0, P_left, P_right)
    P_75 = log_linear_df(1.75, 1.0, 2.0, P_left, P_right)

    assert math.log(P_50) == pytest.approx(
        (math.log(P_25) + math.log(P_75)) / 2
    )
