import calendar

def find_five_sunday_months(year):
    calendar.setfirstweekday(calendar.SUNDAY)
    five_sunday_months = []
    for month in range(1, 13):
        calendar_month = calendar.monthcalendar(year, month)
        # If you're counting Sunday as the first day of the week, then any month that extends into
        # six weeks, or starts on a Sunday and extends into five weeks, will contain five Sundays.
        if len(calendar_month) == 6 or (len(calendar_month) == 5 and calendar_month[0][0] == 1):
            five_sunday_months.append(calendar.month_abbr[month])

    return five_sunday_months

print (find_five_sunday_months(2018))
