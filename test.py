import csv
import time


date_format = '%Y-%m-%d%I:%M %p'
new_date_format = '%Y/%m/%d %H:%M:%S'
new_obs_data = []

with open('data/observation.csv') as f:
    obs_csv = csv.reader(f)
    headers = next(obs_csv)
    new_headers = ['temp', 'dewPt', 'rh', 'wdir_cardinal', 'wspd', 'pressure', 'wx_phrase',
                                           'valid_time', 'feels_like']
    new_obs_data.append(new_headers)
    for row in obs_csv:
        new_row = row[2:6]
        new_row.append(row[8])
        new_row.append(row[7])
        date_time = row[0] + row[1]
        date_tuple = time.strptime(date_time, date_format)
        new_date_time = time.strftime(new_date_format, date_tuple)
        new_row.append(new_date_time)
        new_row.append(row[2])
        new_obs_data.append(new_row)

with open('data/weather_test.csv', 'w+', newline='') as f:
    weather_csv = csv.writer(f)
    weather_csv.writerows(new_obs_data)



