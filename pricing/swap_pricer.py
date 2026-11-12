import math
from numbers import Real
from collections.abc import Sequence

from curves.yield_curve import YieldCurve



def price_swap(swap,curve):
    annuity = 0
    for i in range(len(swap.accrual_fractions)):
        annuity += swap.accrual_fractions[i] * curve.discount_factor(swap.payment_times[i])

    df = curve.discount_factor(swap.payment_times[-1])

    fixed_pv = swap.notional * swap.fixed_rate * annuity
    float_pv = swap.notional * (1-df)

    res = fixed_pv - float_pv
    return res if swap.direction == "receiver" else -res

def par_swap_rate(swap, curve):
    annuity = 0
    for i in range(len(swap.accrual_fractions)):
            annuity += swap.accrual_fractions[i] * curve.discount_factor(swap.payment_times[i])

    df = curve.discount_factor(swap.payment_times[-1])
    rate = (1-df) / annuity

    return rate




def forward_swap_rate(
    curve: YieldCurve,
    start_time: float,
    payment_times: list[float],
    accrual_fractions: list[float],
) -> float:
    

    if not isinstance(curve, YieldCurve):
        raise TypeError("curve must be a YieldCurve instance.")

    if (isinstance(start_time, bool)
            or not isinstance(start_time, Real)):
        raise TypeError("start_time must be a real number.")

    if not math.isfinite(start_time) or start_time < 0:
        raise ValueError("start_time must be finite and non-negative.")

    if not isinstance(payment_times, (list, tuple)):
        raise TypeError("payment_times must be a list or tuple.")

    if not payment_times:
        raise ValueError("payment_times must not be empty.")

    if not isinstance(accrual_fractions, (list, tuple)):
        raise TypeError("accrual_fractions must be a list or tuple.")

    if len(payment_times) != len(accrual_fractions):
        raise ValueError(
            "payment_times and accrual_fractions must have equal lengths."
        )

    for i, t in enumerate(payment_times):
        if isinstance(t, bool) or not isinstance(t, Real):
            raise TypeError(
                f"payment_times[{i}] must be a real number."
            )

        if not math.isfinite(t) or t <= 0:
            raise ValueError(
                f"payment_times[{i}] must be finite and positive."
            )

        if i > 0 and t <= payment_times[i - 1]:
            raise ValueError(
                "payment_times must be strictly increasing."
            )

    if payment_times[0] <= start_time:
        raise ValueError(
            "The first payment time must be after start_time."
        )

    for i, alpha in enumerate(accrual_fractions):
        if isinstance(alpha, bool) or not isinstance(alpha, Real):
            raise TypeError(
                f"accrual_fractions[{i}] must be a real number."
            )

        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError(
                f"accrual_fractions[{i}] must be finite and positive."
            )

    if payment_times[-1] > curve.maturities[-1]:
        raise ValueError(
            "Swap maturity exceeds the yield curve's maximum maturity."
        )

    annuity = 0
    for i in range(len(accrual_fractions)):
        annuity += accrual_fractions[i] * curve.discount_factor(payment_times[i])
    
    df_start = curve.discount_factor(start_time)
    df_end = curve.discount_factor(payment_times[-1])

    return (df_start - df_end) / annuity
