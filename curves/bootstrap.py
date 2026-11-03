import math
from curves.yield_curve import YieldCurve
from scipy.optimize import brentq

def bootstrap_deposit(rate, alpha):
    if not isinstance(alpha,(int, float)) or type(alpha) == bool or alpha <= 0:
        raise ValueError("Accrual fraction should be positive.")

    if not isinstance(rate,(int, float)) or type(rate) == bool:
        raise ValueError("Invalid rate.")

    if (1+rate*alpha) <= 0:
        raise ValueError("Invalid value.")
    
    return 1/(1+rate * alpha)

def bootstrap_fra(start_df, forward_rate, alpha):
    if not isinstance(start_df, (float, int)) or type(start_df) == bool or start_df <= 0:
        raise ValueError("Invalid starting discount factor.")

    if not isinstance(alpha,(int, float)) or type(alpha) == bool or alpha <= 0:
        raise ValueError("Accrual fraction should be positive.")

    if not isinstance(forward_rate,(int, float)) or type(forward_rate) == bool:
        raise ValueError("Invalid forward rate.")

    if 1+forward_rate*alpha <= 0:
        raise ValueError("Invalid value.")

    return start_df / (1+forward_rate*alpha)

def bootstrap_swap(par_rate, known_dfs, accrual_fractions):
    #Imput Validation
    if len(known_dfs) + 1 != len(accrual_fractions):
        raise ValueError("Invalid discount factors list and accrual fractions list.")

    if not isinstance(par_rate, (int, float)) or type(par_rate) is bool:
        raise ValueError("Invalid par rate.")

    if not math.isfinite(par_rate):
        raise ValueError("Par rate need to be finite.")

    #Main function
    annuity_known = 0
    for i in range(len(known_dfs)):
        if not math.isfinite(known_dfs[i]) or not math.isfinite(accrual_fractions[i]) or not isinstance(known_dfs[i],(int, float)) or not isinstance(accrual_fractions[i],(int,float)) or type(accrual_fractions[i]) is bool or type(known_dfs[i]) is bool:
            raise ValueError("Invalid discount factor or accrual fraction.")

        if known_dfs[i] <= 0 or accrual_fractions[i] <= 0:
            raise ValueError("Discount factors and accrual fractions need to be positive.")
        annuity_known += known_dfs[i] * accrual_fractions[i]

    if not math.isfinite(accrual_fractions[-1]) or not isinstance(accrual_fractions[-1],(int,float)) or type(accrual_fractions[-1]) is bool or accrual_fractions[-1] <= 0:
        raise ValueError("Discount factors and accrual fractions need to be positive.")

    if (1+par_rate*accrual_fractions[-1]) == 0 or (1-par_rate*annuity_known)/(1+par_rate*accrual_fractions[-1]) <=0:
        raise ValueError("Invalid input.")

    return (1-par_rate*annuity_known)/(1+par_rate*accrual_fractions[-1])

def bootstrap_swap_numerical(
    par_rate,
    curve,
    payment_times,
    accrual_fractions
):
    #Input Validation
        # Input Validation
    if not isinstance(curve, YieldCurve):
        raise TypeError("curve must be a YieldCurve instance.")

    if (isinstance(par_rate, bool)
            or not isinstance(par_rate, (int, float))
            or not math.isfinite(par_rate)):
        raise ValueError("par_rate must be a finite number.")

    if not isinstance(payment_times, (list, tuple)) or not payment_times:
        raise ValueError("payment_times must be a non-empty sequence.")

    if not isinstance(accrual_fractions, (list, tuple)):
        raise ValueError("accrual_fractions must be a sequence.")

    if len(payment_times) != len(accrual_fractions):
        raise ValueError("Payment times and accrual fractions must have equal lengths.")

    for t in payment_times:
        if (isinstance(t, bool)
                or not isinstance(t, (int, float))
                or not math.isfinite(t)
                or t <= 0):
            raise ValueError("Payment times must be finite positive numbers.")

    for alpha in accrual_fractions:
        if (isinstance(alpha, bool)
                or not isinstance(alpha, (int, float))
                or not math.isfinite(alpha)
                or alpha <= 0):
            raise ValueError("Accrual fractions must be finite positive numbers.")

    if any(payment_times[i] <= payment_times[i - 1]
           for i in range(1, len(payment_times))):
        raise ValueError("Payment times must be strictly increasing.")

    if payment_times[-1] <= curve.maturities[-1]:
        raise ValueError("Swap maturity must exceed the last curve maturity.")
    

    def objective(x):
        if x <= 0:
            raise ValueError
        maturities = list(curve.maturities)
        maturities.append(payment_times[-1])
        discount_factors = list(curve.discount_factors)
        discount_factors.append(x)
        temp_curve = YieldCurve(maturities, discount_factors)
        annuity = 0
        for i in range(len(payment_times)):
            annuity += accrual_fractions[i]*temp_curve.discount_factor(payment_times[i])
    
        return par_rate*annuity - (1-x)

    a = 0.00001
    b = 3
    f_a = objective(a)
    f_b = objective(b)
    target = 0

    if f_a == 0:
        return a
    if f_b == 0:
        return b
    if f_a * f_b < 0:
        target = brentq(objective, a, b)
    else:
        raise ValueError("Root is not bracketed")

    return target
        
def build_yield_curve(market_quotes):
    if not market_quotes:
        raise ValueError("Market quotes is empty.")

    sorted_market_quotes = sorted(market_quotes, key=lambda x: x["maturity"])

    maturities = [0.0]
    discount_factors = [1.0]
    for quote in sorted_market_quotes:
        if quote["maturity"] <= maturities[-1]:
            raise ValueError("Invalid maturity.")
        if quote["type"] not in {"fra", "deposit", "swap"}:
            raise ValueError("Invalid instrument type.")
        
        if quote["type"] == "deposit":
            discount_factors.append(bootstrap_deposit(quote["rate"], quote["alpha"]))
            maturities.append(quote["maturity"])
        elif quote["type"] == "fra":
            if quote["start"] >= quote["maturity"] or quote["start"] < 0:
                raise ValueError("Starting time must be non-negative and strictly earlier than maturity.")
            curve = YieldCurve(maturities, discount_factors)
            start_df = curve.discount_factor(quote["start"])
            discount_factors.append(bootstrap_fra(start_df, quote["rate"], quote["alpha"]))
            maturities.append(quote["maturity"])
        elif quote["type"] == "swap":
            if len(quote["payment_times"]) == 0:
                raise ValueError("Empty payment times.")
            if len(quote["payment_times"]) != len(quote["accrual_fractions"]):
                raise ValueError("Invalid payment times or accrual fractions.")
            if quote["payment_times"][-1] != quote["maturity"]:
                raise ValueError("Last payment time not matched with maturity.")
            if quote["payment_times"][0] <= 0:
                raise ValueError("First payment time should be positive.")

            known_dfs = []
            curve = YieldCurve(maturities, discount_factors)
            for i in range(len(quote["payment_times"])-1):
                if quote["payment_times"][i] >= quote["payment_times"][i+1]:
                    raise ValueError("Payment times should be strictly increasing.")
            if len(quote["payment_times"]) > 1 and quote["payment_times"][-2] > curve.maturities[-1]:
                discount_factors.append(bootstrap_swap_numerical(quote["rate"],curve,quote["payment_times"], quote["accrual_fractions"]))
            else:
                for i in range(len(quote["payment_times"])-1):
                    time = quote["payment_times"][i]
                    known_dfs.append(curve.discount_factor(time))
                discount_factors.append(bootstrap_swap(quote["rate"], known_dfs, quote["accrual_fractions"]))   
            maturities.append(quote["maturity"])

    res = YieldCurve(maturities, discount_factors)

    return res


