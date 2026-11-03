import math

class VanillaSwap:
    def __init__(
        self,
        notional,
        fixed_rate,
        payment_times,
        accrual_fractions,
        direction
    ):
        # Validate notional
        if (isinstance(notional, bool)
                or not isinstance(notional, (int, float))
                or not math.isfinite(notional)
                or notional <= 0):
            raise ValueError("Notional must be a finite positive number.")

        # Validate fixed rate (negative rates are allowed)
        if (isinstance(fixed_rate, bool)
                or not isinstance(fixed_rate, (int, float))
                or not math.isfinite(fixed_rate)):
            raise ValueError("Fixed rate must be a finite number.")

        # Validate payment times
        if not isinstance(payment_times, (list, tuple)) or not payment_times:
            raise ValueError("Payment times must be a non-empty list or tuple.")

        for t in payment_times:
            if (isinstance(t, bool)
                    or not isinstance(t, (int, float))
                    or not math.isfinite(t)
                    or t <= 0):
                raise ValueError("Payment times must be finite positive numbers.")

        if any(payment_times[i] <= payment_times[i - 1]
               for i in range(1, len(payment_times))):
            raise ValueError("Payment times must be strictly increasing.")

        # Validate accrual fractions
        if not isinstance(accrual_fractions, (list, tuple)):
            raise ValueError("Accrual fractions must be a list or tuple.")

        if len(accrual_fractions) != len(payment_times):
            raise ValueError("Payment times and accrual fractions must have equal lengths.")

        for alpha in accrual_fractions:
            if (isinstance(alpha, bool)
                    or not isinstance(alpha, (int, float))
                    or not math.isfinite(alpha)
                    or alpha <= 0):
                raise ValueError("Accrual fractions must be finite positive numbers.")

        # Validate direction
        if direction not in ("payer", "receiver"):
            raise ValueError("Direction must be 'payer' or 'receiver'.")

        # Store contract attributes
        self.notional = notional
        self.fixed_rate = fixed_rate
        self.payment_times = tuple(payment_times)
        self.accrual_fractions = tuple(accrual_fractions)
        self.direction = direction