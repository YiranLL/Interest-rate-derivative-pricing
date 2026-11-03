
import math
import pytest

from curves.bootstrap import (
    bootstrap_deposit,
    bootstrap_fra,
    bootstrap_swap
)


# ==========================================
# 1. DEPOSIT BOOTSTRAPPING
# ==========================================

def test_deposit_basic():
    result = bootstrap_deposit(0.04, 0.5)
    expected = 1 / 1.02

    assert result == pytest.approx(expected)


def test_deposit_negative_rate():
    result = bootstrap_deposit(-0.01, 0.5)
    expected = 1 / 0.995

    assert result == pytest.approx(expected)
    assert result > 1


@pytest.mark.parametrize("rate, alpha", [
    (0.04, 0),
    (0.04, -0.5),
    (-3.0, 0.5),
])
def test_deposit_invalid_values(rate, alpha):
    with pytest.raises(ValueError):
        bootstrap_deposit(rate, alpha)


# ==========================================
# 2. FRA BOOTSTRAPPING
# ==========================================

def test_fra_basic():
    start_df = 0.98
    forward_rate = 0.04
    alpha = 0.5

    result = bootstrap_fra(start_df, forward_rate, alpha)

    assert result == pytest.approx(0.98 / 1.02)


def test_fra_negative_rate():
    result = bootstrap_fra(0.98, -0.01, 0.5)

    assert result == pytest.approx(0.98 / 0.995)


@pytest.mark.parametrize("start_df, rate, alpha", [
    (0, 0.04, 0.5),
    (-0.95, 0.04, 0.5),
    (0.98, 0.04, 0),
    (0.98, 0.04, -0.5),
    (0.98, -3.0, 0.5),
])
def test_fra_invalid_values(start_df, rate, alpha):
    with pytest.raises(ValueError):
        bootstrap_fra(start_df, rate, alpha)


# ==========================================
# 3. SWAP BOOTSTRAPPING
# ==========================================

def test_swap_basic():
    result = bootstrap_swap(
        par_rate=0.04,
        known_dfs=[0.96, 0.92],
        accrual_fractions=[1.0, 1.0, 1.0]
    )

    expected = (1 - 0.04 * (0.96 + 0.92)) / 1.04

    assert result == pytest.approx(expected)


def test_swap_single_period():
    result = bootstrap_swap(
        par_rate=0.04,
        known_dfs=[],
        accrual_fractions=[0.5]
    )

    assert result == pytest.approx(1 / 1.02)


def test_swap_negative_rate():
    result = bootstrap_swap(
        par_rate=-0.01,
        known_dfs=[1.005],
        accrual_fractions=[0.5, 0.5]
    )

    expected = (1 + 0.01 * 0.5 * 1.005) / 0.995

    assert result == pytest.approx(expected)


@pytest.mark.parametrize("known_dfs, accruals", [
    ([0.96, 0.92], [1.0, 1.0]),
    ([0.96], [1.0, 1.0, 1.0]),
    ([], []),
    ([-0.96], [1.0, 1.0]),
    ([0.96], [0.0, 1.0]),
    ([0.96], [1.0, -1.0]),
])
def test_swap_invalid_inputs(known_dfs, accruals):
    with pytest.raises(ValueError):
        bootstrap_swap(0.04, known_dfs, accruals)


# ==========================================
# 4. REPRICING CONSISTENCY
# ==========================================

def test_deposit_repricing():
    market_rate = 0.04
    alpha = 0.5

    df = bootstrap_deposit(market_rate, alpha)

    implied_rate = (1 / df - 1) / alpha

    assert implied_rate == pytest.approx(market_rate)


def test_fra_repricing():
    market_rate = 0.042
    alpha = 0.5
    start_df = 0.98

    end_df = bootstrap_fra(start_df, market_rate, alpha)

    implied_forward = (start_df / end_df - 1) / alpha

    assert implied_forward == pytest.approx(market_rate)


def test_swap_repricing():
    market_par_rate = 0.04
    known_dfs = [0.96, 0.92]
    accruals = [1.0, 1.0, 1.0]

    final_df = bootstrap_swap(
        market_par_rate, known_dfs, accruals
    )

    all_dfs = known_dfs + [final_df]

    annuity = sum(
        alpha * df
        for alpha, df in zip(accruals, all_dfs)
    )

    implied_par_rate = (1 - final_df) / annuity

    assert implied_par_rate == pytest.approx(market_par_rate)


# ==========================================
# 5. NUMERICAL ROBUSTNESS
# ==========================================

@pytest.mark.parametrize("bad_rate", [
    float("nan"),
    float("inf"),
    float("-inf"),
])
def test_swap_nonfinite_par_rate(bad_rate):
    with pytest.raises(ValueError):
        bootstrap_swap(
            bad_rate,
            [0.96],
            [1.0, 1.0]
        )


@pytest.mark.parametrize("bad_value", [
    float("nan"),
    float("inf"),
    float("-inf"),
])
def test_swap_nonfinite_known_df(bad_value):
    with pytest.raises(ValueError):
        bootstrap_swap(
            0.04,
            [bad_value],
            [1.0, 1.0]
        )


@pytest.mark.parametrize("bad_value", [
    float("nan"),
    float("inf"),
    float("-inf"),
])
def test_swap_nonfinite_accrual(bad_value):
    with pytest.raises(ValueError):
        bootstrap_swap(
            0.04,
            [0.96],
            [bad_value, 1.0]
        )
