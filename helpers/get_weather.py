import requests
import h5py
import json
from calendar import monthrange
import numpy as np

from secrets import * # file where api_key is saved

'''
script used to generate weather info of newyork city from 2010-01-01 to 2014-12-31
'''

years = [2010, 2011, 2012, 2013, 2014]
months = [i for i in range(1,13)]
T = 24 # intervals per day
city = 'New+York' # string to insert in the http request
fname = 'NY_Meteorology.h5' # output file name

# dictionary to build one-hot vectore from weather code
# see http://www.worldweatheronline.com/feed/wwoConditionCodes.txt
codes_index_map = {
  '395': 8,
  '392': 8,
  '389': 8,
  '386': 8,
  '377': 9,
  '374': 9,
  '371': 11,
  '368': 11,
  '365': 11,
  '362': 11,
  '359': 7,
  '356': 12,
  '353': 5,
  '350': 9,
  '338': 10,
  '335': 10,
  '332': 12,
  '329': 12,
  '326': 11,
  '323': 11,
  '320': 12,
  '317': 12,
  '314': 5,
  '311': 4,
  '308': 6,
  '305': 6,
  '302': 5,
  '299': 5,
  '296': 5,
  '293': 5,
  '284': 9,
  '281': 9,
  '266': 5,
  '263': 5,
  '260': 3,
  '248': 3,
  '230': 10,
  '227': 10,
  '200': 8,
  '185': 9,
  '182': 11,
  '179': 11,
  '176': 4,
  '143': 3,
  '122': 2,
  '119': 1,
  '116': 1,
  '113': 0
}

timestamps = []
temperatures = []
windspeeds = []
weather = []
# loop on years and months because this API only support startdate and enddate
# in the same month and year
for year in years:
  for month in months:
    # get the number of days of the curren month in the current year
    num_days_in_month = monthrange(year, month)[-1]

    startdate = f'{year}-{month:02d}-01'
    enddate = f'{year}-{month:02d}-{num_days_in_month}'

    # call api and get weather info for current month
    r = requests.get(f'http://api.worldweatheronline.com/premium/v1/past-weather.ashx?q={city}&tp=1&date={startdate}&enddate={enddate}&format=json&key={api_key}')

    # get json (see https://www.worldweatheronline.com/developer/api/docs/historical-weather-api.aspx)
    j = json.loads(r.text)['data']['weather']

    # loop on days
    for day in range(1,num_days_in_month+1):
      # loop on time intervals
      for t in range(T):
        # append timestamp
        timestamp = f'{year}{month:02d}{day:02d}{t+1:02d}'
        timestamps.append(bytes(timestamp, 'utf8'))

        # get info for current timestamp
        info = j[day-1]['hourly'][t]

        temperatures.append(int(info['tempC']))
        windspeeds.append(int(info['windspeedMiles']))
        weather_code = info['weatherCode']
        weather_vector = np.zeros(len(set(codes_index_map.values())))
        weather_vector[codes_index_map[weather_code]] = 1
        weather.append(weather_vector)

# convert lists to np arrays
timestamps = np.asarray(timestamps)
temperatures = np.asarray(temperatures)
windspeeds = np.asarray(windspeeds)
weather = np.asarray(weather)

# write to file
h5 = h5py.File(fname, 'w')
h5.create_dataset('date', data=timestamps)
h5.create_dataset('Temperature', data=temperatures)
h5.create_dataset('WindSpeed', data=windspeeds)
h5.create_dataset('Weather', data=weather)
h5.close()
