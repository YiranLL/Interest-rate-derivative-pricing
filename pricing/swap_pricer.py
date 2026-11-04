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