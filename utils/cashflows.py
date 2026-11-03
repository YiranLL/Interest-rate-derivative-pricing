from utils.dates import year_fraction

def fixed_leg_cashflows(notional, fixed_rate, schedule, day_count):
    if not isinstance(notional, (int, float)) or notional <= 0 or notional is True:
        raise ValueError("Notional value need to be a positive number.") 

    if len(schedule) <= 1:
        raise ValueError("Schedule should contain more than 1 dates.")

    res = []
    for i in range(1, len(schedule)):
        if schedule[i] <= schedule[i-1]:
            raise ValueError("List of schedule should contain only strictly increasing dates.")
        res.append((schedule[i], notional * fixed_rate * year_fraction(schedule[i-1], schedule[i], day_count)))

    return res