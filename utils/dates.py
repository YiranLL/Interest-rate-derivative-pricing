from datetime import date
import calendar
from dateutil.relativedelta import relativedelta

def year_fraction(start_date, end_date, convention):
    if convention != 'ACT/360' and convention != 'ACT/365' and convention != '30/360US':
        raise ValueError('Unsupported convention.')
    if end_date < start_date:
        raise ValueError('End date earlier than start date.')
    if end_date == start_date:
        return 0
    
    time_difference = end_date - start_date
    if convention == 'ACT/360':
        return (time_difference.days) / 360
    elif convention == 'ACT/365':
        return (time_difference.days) / 365
    elif convention == '30/360US':
        y1 = start_date.year
        m1 = start_date.month
        d1 = start_date.day

        y2 = end_date.year
        m2 = end_date.month
        d2 = end_date.day

        start_is_feb_end = True if m1 == 2 and (calendar.monthrange(y1, m1)[1] == d1) else False
        end_is_feb_end = True if m2 == 2 and (calendar.monthrange(y2, m2)[1] == d2) else False
        if d1 == 31 or start_is_feb_end == True:
            d1 = 30
        if (d2 == 31 and d1 == 30) or (end_is_feb_end == True and start_is_feb_end == True):
            d2 = 30

        return (360*(y2 - y1) + 30*(m2 - m1) + (d2 - d1))/360

def generate_schedule(start_date, end_date, frequency_months):
    if end_date <= start_date:
        raise ValueError("End date before start date.")

    if not isinstance(frequency_months, int) or frequency_months <= 0:
        raise ValueError("Frequency months should be an positive integer.")

    res = [start_date]
    count = 1
    date = start_date + relativedelta(months = frequency_months * count)

    while  date < end_date:
        res.append(date)
        count += 1
        date = start_date + relativedelta(months = frequency_months * count)

    res.append(end_date)

    return res
