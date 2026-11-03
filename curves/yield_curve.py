from utils.rates import zero_rate, forward_rate
from curves.interpolation import log_linear_df

class YieldCurve:
    def __init__(self, maturities, discount_factors):
        if len(maturities) != len(discount_factors):
            raise ValueError("Maturities and discount factors must have same length.")

        if len(maturities) == 0:
            raise ValueError("Empty list.")
        
        for i in range(1, len(maturities)):
            if maturities[i] <0 or maturities[i] <= maturities[i-1]:
                raise ValueError("Maturities must be a list of nonnegative strictly increasing floating numbers.")

            if discount_factors[i] <= 0:
                raise ValueError("Discount factors must be strictly positive.")

        if maturities[0] != 0 or discount_factors[0] != 1:
            raise ValueError("By convention, the first maturities must be 0 and the corresponding discount factor is 1.")

        self.maturities = tuple(maturities)
        self.discount_factors = tuple(discount_factors)

    def discount_factor(self, T):
        if T > self.maturities[-1] or T < 0:
            raise ValueError("Extrapolation required")
        l = 0
        r = len(self.maturities) - 1
        while l < r:
            mid = (l+r)//2
            if self.maturities[mid] < T:
                l = mid+1
            elif self.maturities[mid] == T:
                return self.discount_factors[mid]
            else:
                r = mid

        if self.maturities[l] == T:
            return self.discount_factors[l]
        
        return log_linear_df(T, self.maturities[l-1], self.maturities[l], self.discount_factors[l-1], self.discount_factors[l])

    def zero_rate(self, T):
        return zero_rate(T, self.discount_factor(T))

    def forward_rate(self, T1, T2, alpha):
        return forward_rate(self.discount_factor(T1), self.discount_factor(T2), alpha)

