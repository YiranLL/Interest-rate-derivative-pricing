import math

class FRA:
    def __init__(
        self,
        notional,
        start_time,
        end_time,
        fixed_rate,
        accrual_fraction,
        direction
    ):
        if end_time <= start_time or start_time < 0:
            raise ValueError("Invalid start time or end time.")
        if notional <= 0:
            raise ValueError("Invalid notional.")
        if accrual_fraction <= 0:
            raise ValueError("Invalid accrual fraction.")
        if direction not in {"receive", "pay"}:
            raise ValueError("Invalid direction.")
        if not math.isfinite(fixed_rate):
            raise ValueError("Fixed rate is infinite.")
            
        self.notional = notional
        self.start_time = start_time
        self.end_time = end_time
        self.fixed_rate = fixed_rate
        self.accrual_fraction = accrual_fraction
        self.direction = direction #"receive" for receive float, "pay" for pay float.