import datetime
import re

'''
script used to create a file containing holiday days in new york from 2010-01-01
to 2014-12-31. Each line of the created file is in the format yyyymmdd.
holidays.txt was manually created from https://www.public-holidays.us/US_PD_2010_New%20York
'''

p = ', (\d* [a-zA-Z]* \d*)'
with open('NY_Holiday.txt', 'w')as f1:
    with open('holidays.txt') as f2:
        newline = ''
        for line in f2:
            str_date = re.search(p, line).group(1)
            ts = datetime.datetime.strptime(str_date, "%d %B %Y")
            print(ts.strftime("%Y%m%d"))
            f1.write(newline + ts.strftime("%Y%m%d"))
            newline = '\n'
            if 'str' in line:
                break
