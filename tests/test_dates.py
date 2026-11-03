from datetime import date
import pytest

from utils.dates import generate_schedule


def test_regular_schedule():
    result = generate_schedule(
        date(2027, 1, 15),
        date(2028, 1, 15),
        6
    )

    expected = [
        date(2027, 1, 15),
        date(2027, 7, 15),
        date(2028, 1, 15)
    ]

    assert result == expected


def test_month_end_schedule():
    result = generate_schedule(
        date(2028, 1, 31),
        date(2028, 4, 30),
        1
    )

    expected = [
        date(2028, 1, 31),
        date(2028, 2, 29),
        date(2028, 3, 31),
        date(2028, 4, 30)
    ]

    assert result == expected


def test_invalid_frequency():
    with pytest.raises(ValueError):
        generate_schedule(
            date(2027, 1, 15),
            date(2028, 1, 15),
            0
        )