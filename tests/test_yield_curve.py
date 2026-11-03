
import math
import pytest

from curves.yield_curve import YieldCurve


@pytest.fixture
def curve():
    return YieldCurve(
        maturities=[0.0, 0.5, 1.0, 2.0],
        discount_factors=[1.0, 0.98, 0.95, 0.90]
    )


# 1. Test all existing curve pillars
@pytest.mark.parametrize("T, expected", [
    (0.0, 1.0),
    (0.5, 0.98),
    (1.0, 0.95),
    (2.0, 0.90)
])
def test_discount_factor_at_pillars(curve, T, expected):
    assert curve.discount_factor(T) == pytest.approx(expected)


# 2. Continuous zero rate
def test_zero_rate(curve):
    result = curve.zero_rate(1.0)
    expected = -math.log(0.95)

    assert result == pytest.approx(expected)


# 3. Simple forward rate
def test_forward_rate(curve):
    result = curve.forward_rate(0.5, 1.0, 0.5)
    expected = (0.98 / 0.95 - 1) / 0.5

    assert result == pytest.approx(expected)


# 4. Invalid curve construction
@pytest.mark.parametrize("maturities, dfs", [
    ([], []),
    ([0.0, 1.0], [1.0]),
    ([0.0, 1.0, 1.0], [1.0, 0.95, 0.90]),
    ([0.0, 2.0, 1.0], [1.0, 0.90, 0.95]),
    ([0.0, -0.5, 1.0], [1.0, 0.98, 0.95]),
    ([0.0, 1.0], [1.0, -0.95]),
    ([0.0, 1.0], [1.0, 0.0]),
    ([0.5, 1.0], [0.98, 0.95]),
    ([0.0, 1.0], [0.99, 0.95])
])
def test_invalid_curve_construction(maturities, dfs):
    with pytest.raises(ValueError):
        YieldCurve(maturities, dfs)


# 5. Input lists should not affect the curve afterwards
def test_input_mutation_does_not_affect_curve():
    maturities = [0.0, 0.5, 1.0]
    dfs = [1.0, 0.98, 0.95]

    curve = YieldCurve(maturities, dfs)

    maturities[1] = 10.0
    dfs[1] = 0.50

    assert curve.maturities == (0.0, 0.5, 1.0)
    assert curve.discount_factors == (1.0, 0.98, 0.95)
    assert curve.discount_factor(0.5) == pytest.approx(0.98)


# 6. A single time-zero pillar is valid
def test_single_pillar():
    curve = YieldCurve([0.0], [1.0])

    assert curve.discount_factor(0.0) == pytest.approx(1.0)

    with pytest.raises(ValueError):
        curve.discount_factor(1.0)


# 7. Zero rate at T=0 is undefined
def test_zero_rate_at_zero(curve):
    with pytest.raises(ValueError):
        curve.zero_rate(0.0)


# 8. Forward rate requires a positive accrual fraction
def test_invalid_forward_alpha(curve):
    with pytest.raises(ValueError):
        curve.forward_rate(0.5, 1.0, 0.0)


# Additional tests for YieldCurve interpolation

@pytest.mark.parametrize("T, expected", [
    (0.25, math.sqrt(1.0 * 0.98)),
    (0.75, math.sqrt(0.98 * 0.95)),
    (1.50, math.sqrt(0.95 * 0.90))
])
def test_interpolated_discount_factor(curve, T, expected):
    assert curve.discount_factor(T) == pytest.approx(expected)


@pytest.mark.parametrize("T", [
    -0.5,
    3.0
])
def test_discount_factor_extrapolation(curve, T):
    with pytest.raises(ValueError):
        curve.discount_factor(T)


# Forward rate should be constant within a log-linear segment
def test_constant_forward_within_segment(curve):
    forward_1 = curve.forward_rate(
        1.0, 1.25, 0.25
    )

    forward_2 = curve.forward_rate(
        1.25, 1.5, 0.25
    )

    # Continuous forwards are constant within the segment.
    # Simple forwards over equal-length intervals are also equal.
    assert forward_1 == pytest.approx(forward_2)


# Zero rate should be consistent with interpolated DF
def test_interpolated_zero_rate(curve):
    T = 1.5
    df = math.sqrt(0.95 * 0.90)

    expected = -math.log(df) / T

    assert curve.zero_rate(T) == pytest.approx(expected)
