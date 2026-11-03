import math

def log_linear_df(T, T_left, T_right, P_left, P_right):
    if T_right <= T_left:
        raise ValueError("Left time is not strictled earier than right time.")

    if P_left <= 0 or P_right <= 0:
        raise ValueError("Discount factors should be strictly positive.")

    if T < T_left or T > T_right:
        raise ValueError("T out of range.")

    w = (T-T_left) / (T_right - T_left)

    return math.exp((1-w)*math.log(P_left) + w * math.log(P_right)) 

    