import math
from curves.yield_curve import YieldCurve
from instruments.swap import VanillaSwap
from pricing.swap_pricer import price_swap

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

