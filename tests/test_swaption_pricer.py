"""Black-76 European swaption pricer tests.

Place in tests/test_swaption_pricer.py and run:
    python -m pytest tests/test_swaption_pricer.py -v

Uses the REAL project modules, no mocking. Assumes Swaption is currently
in instruments/swap.py and price_swaption in pricing/swaption_pricer.py.
"""

import copy
import math

import pytest
from scipy.stats import norm

from curves.yield_curve import YieldCurve
from instruments.swap import Swaption
from pricing.swap_pricer import forward_swap_rate
from pricing.swaption_pricer import price_swaption


@pytest.fixture
def curve():
    # Smooth, strictly positive forward swap rates, with exact annual pillars.
    r = 0.04
    maturities = [float(i) for i in range(7)]
    dfs = [math.exp(-r * t) for t in maturities]
    return YieldCurve(maturities, dfs)


@pytest.fixture
def contract_kwargs():
    return dict(
        notional=1_000_000,
        strike=0.04,
        expiry=1.0,
        payment_times=[2.0, 3.0, 4.0, 5.0, 6.0],
        accrual_fractions=[1.0] * 5,
    )


def make_swaption(kwargs, direction="payer", **overrides):
    data = dict(kwargs)
    data.update(overrides)
    return Swaption(direction=direction, **data)


def annuity(curve, swaption):
    return sum(alpha * curve.discount_factor(t) for alpha, t in zip(
        swaption.accrual_fractions, swaption.payment_times
    ))


def forward(curve, swaption):
    # Independent present-value identity (no call to the tested pricer).
    return (curve.discount_factor(swaption.expiry) -
            curve.discount_factor(swaption.payment_times[-1])) / annuity(curve, swaption)


def reference_black76(curve, swaption, volatility):
    # Independent mathematical benchmark of the pricing formula.
    A = annuity(curve, swaption)
    F = forward(curve, swaption)
    K = swaption.strike
    N = swaption.notional
    if volatility == 0:
        intrinsic = max(F - K, 0.0) if swaption.direction == "payer" else max(K - F, 0.0)
        return N * A * intrinsic
    st = volatility * math.sqrt(swaption.expiry)
    d1 = (math.log(F / K) + 0.5 * st * st) / st
    d2 = d1 - st
    if swaption.direction == "payer":
        return N * A * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return N * A * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


@pytest.mark.parametrize("direction", ["payer", "receiver"])
@pytest.mark.parametrize("vol", [0.05, 0.10, 0.20, 0.50, 1.00])
def test_matches_independent_black76_reference(curve, contract_kwargs, direction, vol):
    swpt = make_swaption(contract_kwargs, direction)
    expected = reference_black76(curve, swpt, vol)
    assert price_swaption(swpt, curve, vol) == pytest.approx(expected, rel=1e-11, abs=1e-8)


@pytest.mark.parametrize("direction", ["payer", "receiver"])
@pytest.mark.parametrize("strike", [0.025, 0.04, 0.06])
def test_otm_atm_itm_prices_are_nonnegative(curve, contract_kwargs, direction, strike):
    swpt = make_swaption(contract_kwargs, direction, strike=strike)
    assert price_swaption(swpt, curve, 0.25) >= 0.0


@pytest.mark.parametrize("strike", [0.025, 0.04, 0.06])
@pytest.mark.parametrize("vol", [0.0, 0.10, 0.25, 0.70])
def test_payer_receiver_parity(curve, contract_kwargs, strike, vol):
    payer = make_swaption(contract_kwargs, "payer", strike=strike)
    receiver = make_swaption(contract_kwargs, "receiver", strike=strike)
    expected = payer.notional * annuity(curve, payer) * (forward(curve, payer) - strike)
    diff = price_swaption(payer, curve, vol) - price_swaption(receiver, curve, vol)
    assert diff == pytest.approx(expected, rel=1e-10, abs=1e-7)


def test_at_the_money_payer_receiver_equal_and_positive(curve, contract_kwargs):
    proto = make_swaption(contract_kwargs)
    atm_strike = forward(curve, proto)
    payer = make_swaption(contract_kwargs, "payer", strike=atm_strike)
    receiver = make_swaption(contract_kwargs, "receiver", strike=atm_strike)
    p = price_swaption(payer, curve, 0.20)
    q = price_swaption(receiver, curve, 0.20)
    assert p > 0
    assert p == pytest.approx(q, abs=1e-7)


@pytest.mark.parametrize("direction,strike", [
    ("payer", 0.025), ("payer", 0.06), ("receiver", 0.025), ("receiver", 0.06)
])
def test_zero_volatility_equals_discounted_intrinsic(curve, contract_kwargs, direction, strike):
    swpt = make_swaption(contract_kwargs, direction, strike=strike)
    F = forward(curve, swpt)
    expected = swpt.notional * annuity(curve, swpt) * (
        max(F - strike, 0.0) if direction == "payer" else max(strike - F, 0.0)
    )
    assert price_swaption(swpt, curve, 0.0) == pytest.approx(expected, abs=1e-8)


@pytest.mark.parametrize("direction", ["payer", "receiver"])
def test_tiny_volatility_converges_to_intrinsic(curve, contract_kwargs, direction):
    swpt = make_swaption(contract_kwargs, direction, strike=0.05)
    assert price_swaption(swpt, curve, 1e-8) == pytest.approx(
        price_swaption(swpt, curve, 0.0), abs=1e-5
    )


@pytest.mark.parametrize("direction", ["payer", "receiver"])
def test_price_increases_with_volatility(curve, contract_kwargs, direction):
    swpt = make_swaption(contract_kwargs, direction)
    values = [price_swaption(swpt, curve, vol) for vol in (0.0, 0.1, 0.3, 0.6)]
    assert all(a < b for a, b in zip(values, values[1:]))


@pytest.mark.parametrize("direction", ["payer", "receiver"])
def test_notional_scaling(curve, contract_kwargs, direction):
    swpt = make_swaption(contract_kwargs, direction)
    doubled = make_swaption(contract_kwargs, direction, notional=2_000_000)
    assert price_swaption(doubled, curve, 0.25) == pytest.approx(
        2 * price_swaption(swpt, curve, 0.25), rel=1e-12
    )


def test_payer_decreases_with_strike(curve, contract_kwargs):
    prices = [price_swaption(make_swaption(contract_kwargs, "payer", strike=k), curve, 0.2)
              for k in (0.02, 0.04, 0.06)]
    assert prices[0] > prices[1] > prices[2]


def test_receiver_increases_with_strike(curve, contract_kwargs):
    prices = [price_swaption(make_swaption(contract_kwargs, "receiver", strike=k), curve, 0.2)
              for k in (0.02, 0.04, 0.06)]
    assert prices[0] < prices[1] < prices[2]


def test_uses_forward_swap_rate_consistently(curve, contract_kwargs):
    swpt = make_swaption(contract_kwargs)
    assert forward(curve, swpt) == pytest.approx(
        forward_swap_rate(curve, swpt.expiry, swpt.payment_times, swpt.accrual_fractions)
    )


def test_curve_and_contract_unchanged(curve, contract_kwargs):
    swpt = make_swaption(contract_kwargs)
    before_curve = (tuple(curve.maturities), tuple(curve.discount_factors))
    before_swaption = copy.deepcopy(vars(swpt))
    price_swaption(swpt, curve, 0.25)
    assert (tuple(curve.maturities), tuple(curve.discount_factors)) == before_curve
    assert vars(swpt) == before_swaption


def test_single_payment_swaption():
    curve = YieldCurve([0.0, 1.0, 2.0], [1.0, 0.97, 0.94])
    swpt = Swaption(1_000_000, 0.03, 1.0, [2.0], [1.0], "payer")
    assert price_swaption(swpt, curve, 0.2) == pytest.approx(
        reference_black76(curve, swpt, 0.2), rel=1e-11
    )


def test_interpolated_schedule():
    curve = YieldCurve([0.0, 1.0, 2.0, 3.0, 4.0], [1.0, 0.97, 0.94, 0.90, 0.86])
    swpt = Swaption(1_000_000, 0.04, 1.5, [2.5, 3.5], [1.0, 1.0], "receiver")
    assert price_swaption(swpt, curve, 0.20) == pytest.approx(
        reference_black76(curve, swpt, 0.20), rel=1e-11
    )


@pytest.mark.parametrize("bad", [None, 1, "swaption", [], {}])
def test_swaption_type_validation(curve, bad):
    with pytest.raises(TypeError):
        price_swaption(bad, curve, 0.2)


@pytest.mark.parametrize("bad", [None, 1, "curve", [], {}])
def test_curve_type_validation(contract_kwargs, bad):
    with pytest.raises(TypeError):
        price_swaption(make_swaption(contract_kwargs), bad, 0.2)


@pytest.mark.parametrize("bad", [True, False, "0.2", None, {}, []])
def test_volatility_type_validation(curve, contract_kwargs, bad):
    with pytest.raises(TypeError):
        price_swaption(make_swaption(contract_kwargs), curve, bad)


@pytest.mark.parametrize("bad", [-0.1, float("nan"), float("inf"), -float("inf")])
def test_volatility_value_validation(curve, contract_kwargs, bad):
    with pytest.raises(ValueError):
        price_swaption(make_swaption(contract_kwargs), curve, bad)


def test_swap_maturity_outside_curve(contract_kwargs):
    short_curve = YieldCurve([0.0, 1.0, 2.0, 3.0], [1.0, 0.97, 0.94, 0.90])
    swpt = make_swaption(contract_kwargs)
    with pytest.raises(ValueError):
        price_swaption(swpt, short_curve, 0.2)


def test_black76_rejects_zero_forward_rate(contract_kwargs):
    flat_curve = YieldCurve([float(i) for i in range(7)], [1.0] * 7)
    swpt = make_swaption(contract_kwargs)
    with pytest.raises(ValueError, match="positive"):
        price_swaption(swpt, flat_curve, 0.2)


def test_black76_rejects_negative_forward_rate(contract_kwargs):
    rising_dfs = YieldCurve([float(i) for i in range(7)], [math.exp(0.01 * i) for i in range(7)])
    swpt = make_swaption(contract_kwargs)
    with pytest.raises(ValueError, match="positive"):
        price_swaption(swpt, rising_dfs, 0.2)
