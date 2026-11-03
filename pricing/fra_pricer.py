def price_fra(fra, curve):
    p2 = curve.discount_factor(fra.end_time)
    forward_rate = curve.forward_rate(fra.start_time, fra.end_time, fra.accrual_fraction)

    fra_value = fra.notional * fra.accrual_fraction * p2 * (forward_rate - fra.fixed_rate)
    return fra_value if fra.direction == "receive" else -fra_value

