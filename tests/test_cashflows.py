
from datetime import date
import pytest

from utils.cashflows import fixed_leg_cashflows


# 1. Single coupon: ACT/360
def test_single_coupon():
    schedule = [
        date(2027, 1, 1),
        date(2027, 6, 30)
    ]

    result = fixed_leg_cashflows(
        1_000_000, 0.05, schedule, "ACT/360"
    )

    assert len(result) == 1
    assert result[0][0] == date(2027, 6, 30)
    assert result[0][1] == pytest.approx(25_000)


# 2. Multiple coupons
def test_multiple_coupons():
    schedule = [
        date(2027, 1, 15),
        date(2027, 7, 15),
        date(2028, 1, 15)
    ]

    result = fixed_leg_cashflows(
        1_000_000, 0.05, schedule, "ACT/360"
    )

    assert len(result) == 2
    assert result[0][0] == date(2027, 7, 15)
    assert result[1][0] == date(2028, 1, 15)

    assert result[0][1] == pytest.approx(
        1_000_000 * 0.05 * 181 / 360
    )
    assert result[1][1] == pytest.approx(
        1_000_000 * 0.05 * 184 / 360
    )


# 3. Negative fixed rate is valid
def test_negative_fixed_rate():
    schedule = [
        date(2027, 1, 1),
        date(2027, 6, 30)
    ]

    result = fixed_leg_cashflows(
        1_000_000, -0.01, schedule, "ACT/360"
    )

    assert result[0][1] == pytest.approx(-5_000)


# 4. Floating-point notional is valid
def test_float_notional():
    schedule = [
        date(2027, 1, 1),
        date(2027, 6, 30)
    ]

    result = fixed_leg_cashflows(
        1_000_000.5, 0.05, schedule, "ACT/360"
    )

    assert result[0][1] == pytest.approx(
        1_000_000.5 * 0.05 * 180 / 360
    )


# 5. Invalid notional
@pytest.mark.parametrize("notional", [
    -100,
    0,
    True,
    False,
    "1000000"
])
def test_invalid_notional(notional):
    schedule = [
        date(2027, 1, 1),
        date(2027, 6, 30)
    ]

    with pytest.raises(ValueError):
        fixed_leg_cashflows(
            notional, 0.05, schedule, "ACT/360"
        )


# 6. Schedule too short
def test_short_schedule():
    schedule = [date(2027, 1, 1)]

    with pytest.raises(ValueError):
        fixed_leg_cashflows(
            1_000_000, 0.05, schedule, "ACT/360"
        )


# 7. Schedule is not strictly increasing
@pytest.mark.parametrize("schedule", [
    [date(2027, 7, 1), date(2027, 1, 1)],
    [date(2027, 1, 1), date(2027, 1, 1)]
])
def test_invalid_schedule(schedule):
    with pytest.raises(ValueError):
        fixed_leg_cashflows(
            1_000_000, 0.05, schedule, "ACT/360"
        )


# 8. Unsupported day-count convention
def test_invalid_day_count():
    schedule = [
        date(2027, 1, 1),
        date(2027, 6, 30)
    ]

    with pytest.raises(ValueError):
        fixed_leg_cashflows(
            1_000_000, 0.05, schedule, "INVALID"
        )
