import math

def discount_factor(T, rate, compounding = 'continuous'):
    if T < 0:
        raise ValueError('T should be nonnegative.')
    if compounding == 'annual' and 1+rate <= 0:
        raise ValueError('For annual compounding, 1+rate should be positive.')
    
    if compounding == 'continuous':
        return math.exp(- rate * T)
    elif compounding == 'annual':
        return (1+rate)**(-T)
    else:
        raise ValueError('Unsupported compounding method.')

def zero_rate(T, df, compounding = 'continuous'):
    if T <= 0:
        raise ValueError('T should be positive.')
    if df <= 0:
        raise ValueError('Discount factor should be positive.')
    
    if compounding == 'continuous':
        return -math.log(df) / T
    elif compounding == 'annual':
        return df**(-1/T) - 1
    else:
        raise ValueError('Unsupported compounding method.')

def forward_rate(P1, P2, alpha, compounding = 'simple'):
    if alpha <= 0:
        raise ValueError('Alpha should be positive.')
    if P1 <= 0 or P2 <= 0:
        raise ValueError('P1 and P2 should be positive.')
    
    if compounding == 'simple':
        return (P1/P2 - 1)/alpha
    elif compounding == 'continuous':
        return (math.log(P1)-math.log(P2)) / alpha
    else:
        raise ValueError('Unsupported compounding method.')



