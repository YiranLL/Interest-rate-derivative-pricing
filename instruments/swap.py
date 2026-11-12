import math
from numbers import Real

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




class Swaption:

    def __init__(
        self,
        notional: float,
        strike: float,
        expiry: float,
        payment_times: list[float],
        accrual_fractions: list[float],
        direction: str,
    ) -> None:

        for name, value in (
            ("notional", notional),
            ("strike", strike),
            ("expiry", expiry),
        ):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real number.")

            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite.")

        if notional <= 0:
            raise ValueError("notional must be strictly positive.")

        if expiry <= 0:
            raise ValueError("expiry must be strictly positive.")

        # Black-76 requires a strictly positive strike.
        if strike <= 0:
            raise ValueError("strike must be strictly positive.")

        if not isinstance(payment_times, (list, tuple)):
            raise TypeError(
                "payment_times must be a list or tuple."
            )

        if not payment_times:
            raise ValueError("payment_times must not be empty.")

        if not isinstance(accrual_fractions, (list, tuple)):
            raise TypeError(
                "accrual_fractions must be a list or tuple."
            )

        if len(payment_times) != len(accrual_fractions):
            raise ValueError(
                "payment_times and accrual_fractions "
                "must have equal lengths."
            )

        for i, t in enumerate(payment_times):
            if isinstance(t, bool) or not isinstance(t, Real):
                raise TypeError(
                    f"payment_times[{i}] must be a real number."
                )

            if not math.isfinite(t) or t <= expiry:
                raise ValueError(
                    f"payment_times[{i}] must be finite "
                    "and strictly after expiry."
                )

            if i > 0 and t <= payment_times[i - 1]:
                raise ValueError(
                    "payment_times must be strictly increasing."
                )

        for i, alpha in enumerate(accrual_fractions):
            if isinstance(alpha, bool) or not isinstance(alpha, Real):
                raise TypeError(
                    f"accrual_fractions[{i}] must be a real number."
                )

            if not math.isfinite(alpha) or alpha <= 0:
                raise ValueError(
                    f"accrual_fractions[{i}] must be "
                    "finite and strictly positive."
                )

        if not isinstance(direction, str):
            raise TypeError("direction must be a string.")

        if direction not in ("payer", "receiver"):
            raise ValueError(
                "direction must be 'payer' or 'receiver'."
            )

        self.notional = notional
        self.strike = strike
        self.expiry = expiry
        self.payment_times = tuple(payment_times)
        self.accrual_fractions = tuple(accrual_fractions)
        self.direction = direction