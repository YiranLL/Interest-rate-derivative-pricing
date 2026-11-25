import math
from numbers import Real
from scipy.stats import norm

from curves.yield_curve import YieldCurve
from instruments.swap import Swaption
from pricing.swap_pricer import forward_swap_rate

def price_swaption(
    swaption: Swaption,
    curve: YieldCurve,
    volatility: float,
) -> float:
    ...
    if not isinstance(swaption, Swaption):
        raise TypeError(
            "swaption must be a Swaption instance."
        )

    if not isinstance(curve, YieldCurve):
        raise TypeError(
            "curve must be a YieldCurve instance."
        )

    if isinstance(volatility, bool) or not isinstance(volatility, Real):
        raise TypeError(
            "volatility must be a real number."
        )

    if not math.isfinite(volatility):
        raise ValueError(
            "volatility must be finite."
        )

    if volatility < 0:
        raise ValueError(
            "volatility must be non-negative."
        )

    if swaption.payment_times[-1] > curve.maturities[-1]:
        raise ValueError(
            "Underlying swap maturity exceeds curve maturity."
        )

    S0 = forward_swap_rate(
        curve,
        swaption.expiry,
        swaption.payment_times,
        swaption.accrual_fractions,
    )

    if not math.isfinite(S0) or S0 <= 0:
        raise ValueError(
            "Black-76 requires a strictly positive "
            "and finite forward swap rate."
        )

    
    N = swaption.notional
    
    annuity = 0
    for i, time in enumerate(swaption.payment_times):
        annuity += swaption.accrual_fractions[i] * curve.discount_factor(time)

    if volatility == 0:
        if swaption.direction == "payer":
            return N * annuity * max([S0 - swaption.strike, 0])
        else:
            return N * annuity * max([swaption.strike - S0, 0])
        
    d_1 = (math.log(S0 / swaption.strike) + 0.5* (volatility**2)*swaption.expiry)/(volatility* math.sqrt(swaption.expiry))
    d_2 = d_1 - volatility*math.sqrt(swaption.expiry)

    if swaption.direction == "payer":
        return N * annuity * (S0 * norm.cdf(d_1) - swaption.strike* norm.cdf(d_2))
    else:
        return N * annuity * (swaption.strike * norm.cdf(-d_2) - S0* norm.cdf(-d_1))