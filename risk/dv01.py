import math
from curves.yield_curve import YieldCurve
from instruments.swap import VanillaSwap
from pricing.swap_pricer import price_swap
import copy
from numbers import Real
from curves.bootstrap import build_yield_curve

def _validate_bump(bump: float, allow_negative: bool = False) -> None:
    if isinstance(bump, bool) or not isinstance(bump, (int, float)):
        raise TypeError("bump must be a real number.")

    if not math.isfinite(bump):
        raise ValueError("bump must be finite.")

    if not allow_negative and bump <= 0:
        raise ValueError("bump must be strictly positive.")

def bump_zero_curve(curve: YieldCurve, bump: float) -> YieldCurve:
    if not isinstance(curve, YieldCurve):
        raise TypeError("curve must be a YieldCurve instance.")

    _validate_bump(bump, allow_negative=True)

    maturities = curve.maturities
    df = curve.discount_factors
    new_df = []

    for i, T in enumerate(maturities):
        if T == 0:
            new_df.append(1)
        else:
            new_df.append(df[i]*math.exp(-bump*T))

    new_curve = YieldCurve(maturities, new_df)

    return new_curve

def parallel_zero_dv01(swap: VanillaSwap, curve: YieldCurve,bump: float = 1e-4
) -> float:

    if not isinstance(swap, VanillaSwap):
        raise TypeError("swap must be a VanillaSwap instance.")

    if not isinstance(curve, YieldCurve):
        raise TypeError("curve must be a YieldCurve instance.")

    _validate_bump(bump)
    
    bump_up_curve = bump_zero_curve(curve, bump)
    bump_down_curve = bump_zero_curve(curve, -bump)

    V_up = price_swap(swap,bump_up_curve)
    V_down = price_swap(swap, bump_down_curve)

    return (V_up - V_down) / (2*bump) * 1e-4

def parallel_quote_dv01(swap: VanillaSwap, market_quotes: list[dict], bump: float = 1e-4) -> float:

    if not isinstance(swap, VanillaSwap):
        raise TypeError("swap must be a VanillaSwap instance.")

    if isinstance(bump, bool) or not isinstance(bump, Real):
        raise TypeError("bump must be a real number.")

    if not math.isfinite(bump):
        raise ValueError("bump must be finite.")

    if bump <= 0:
        raise ValueError("bump must be strictly positive.")

    if not isinstance(market_quotes, list):
        raise TypeError("market_quotes must be a list.")

    if not market_quotes:
        raise ValueError("market_quotes must not be empty.")

    for i, quote in enumerate(market_quotes):
        if not isinstance(quote, dict):
            raise TypeError(f"market_quotes[{i}] must be a dictionary.")

        for key in ("type", "maturity", "rate"):
            if key not in quote:
                raise ValueError(
                    f"market_quotes[{i}] is missing required key '{key}'."
                )

        rate = quote["rate"]

        if isinstance(rate, bool) or not isinstance(rate, Real):
            raise TypeError(
                f"market_quotes[{i}]['rate'] must be a real number."
            )

        if not math.isfinite(rate):
            raise ValueError(
                f"market_quotes[{i}]['rate'] must be finite."
            )

    up_market_quotes = copy.deepcopy(market_quotes)
    down_market_quotes = copy.deepcopy(market_quotes)

    for quote in up_market_quotes:
        quote["rate"] += bump

    for quote in down_market_quotes:
        quote["rate"] -= bump

    up_curve = build_yield_curve(up_market_quotes)
    down_curve = build_yield_curve(down_market_quotes)

    V_up = price_swap(swap,up_curve)
    V_down = price_swap(swap, down_curve)

    return (V_up - V_down) / (2*bump) * 1e-4


def bucketed_quote_dv01(
    swap: VanillaSwap,
    market_quotes: list[dict],
    bump: float = 1e-4,
) -> dict[tuple[str, float], float]:
    if not isinstance(swap, VanillaSwap):
        raise TypeError("swap must be a VanillaSwap instance.")

    if isinstance(bump, bool) or not isinstance(bump, Real):
        raise TypeError("bump must be a real number.")

    if not math.isfinite(bump):
        raise ValueError("bump must be finite.")

    if bump <= 0:
        raise ValueError("bump must be strictly positive.")

    if not isinstance(market_quotes, list):
        raise TypeError("market_quotes must be a list.")

    if not market_quotes:
        raise ValueError("market_quotes must not be empty.")

    seen_keys = set()

    for i, quote in enumerate(market_quotes):
        if not isinstance(quote, dict):
            raise TypeError(
                f"market_quotes[{i}] must be a dictionary."
            )

        for key in ("type", "maturity", "rate"):
            if key not in quote:
                raise ValueError(
                    f"market_quotes[{i}] is missing '{key}'."
                )

        if quote["type"] not in ("deposit", "fra", "swap"):
            raise ValueError(
                f"market_quotes[{i}] has an invalid instrument type."
            )

        for key in ("maturity", "rate"):
            value = quote[key]

            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(
                    f"market_quotes[{i}]['{key}'] "
                    "must be a real number."
                )

            if not math.isfinite(value):
                raise ValueError(
                    f"market_quotes[{i}]['{key}'] must be finite."
                )

        if quote["maturity"] <= 0:
            raise ValueError("Quote maturity must be positive.")

        bucket_key = (quote["type"], quote["maturity"])

        if bucket_key in seen_keys:
            raise ValueError(
                f"Duplicate bucket identifier: {bucket_key}"
            )

        seen_keys.add(bucket_key)

    results = {}

    for j, quote in enumerate(market_quotes):
        up_quotes = copy.deepcopy(market_quotes)
        down_quotes = copy.deepcopy(market_quotes)

        up_quotes[j]["rate"] += bump
        down_quotes[j]["rate"] -= bump

        up_curve = build_yield_curve(up_quotes)
        down_curve = build_yield_curve(down_quotes)

        V_up = price_swap(swap, up_curve)
        V_down = price_swap(swap, down_curve)

        dv01_j = (V_up - V_down) / (2 * bump) * 1e-4

        key = (market_quotes[j]["type"],
           market_quotes[j]["maturity"])

        results[key] = dv01_j
    
    return results
