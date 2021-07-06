import csv
import time
import re
import os
import pandas as pd


def gen_weather():
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


def stat_coords():

    lon_Min = 100
    lon_Max = -100

    lat_Min = 100
    lat_Max = -100

    with open('data/speed_data/2018-8/DOT_Traffic_Speeds_NBE_API_2018_10.csv') as f:
        speed_csv = csv.reader(f)
        header = next(speed_csv)
        for row in speed_csv:
            link_points = row[6]
            lon_lats = link_points.strip().split(' ')
            for lon_lat in lon_lats:
                if len(lon_lat) < 16:
                    print(lon_lat)
                    continue
                elif 16 <= len(lon_lat) <= 24:
                    lon, lat = lon_lat.split(',')

                    if float(lon) < lon_Min:
                        lon_Min = float(lon)
                    if float(lon) > lon_Max:
                        lon_Max = float(lon)
                    if float(lat) < lat_Min:
                        lat_Min = float(lat)
                    if float(lat) > lat_Max:
                        lat_Max = float(lat)
                else:
                    print(lon_lat)
                    idx = lon_lat.index('40.', 3)
                    lon_lat1 = lon_lat[:idx]
                    lon_lat2 = lon_lat[idx:]
                    lon, lat = lon_lat1.split(',')
                    if float(lon) < lon_Min:
                        lon_Min = float(lon)
                    if float(lon) > lon_Max:
                        lon_Max = float(lon)
                    if float(lat) < lat_Min:
                        lat_Min = float(lat)
                    if float(lat) > lat_Max:
                        lat_Max = float(lat)
                    lon, lat = lon_lat2.split(',')
                    if float(lon) < lon_Min:
                        lon_Min = float(lon)
                    if float(lon) > lon_Max:
                        lon_Max = float(lon)
                    if float(lat) < lat_Min:
                        lat_Min = float(lat)
                    if float(lat) > lat_Max:
                        lat_Max = float(lat)

        print("lon:", lon_Min, lon_Max)
        print("lat:", lat_Min, lat_Max)


def extract_speed_colums_daily(baseFilePath, monthFileList):

    # origin_date_format = '%m/%d/%Y %I:%M:%S %p'
    origin_date_format = '%Y-%m-%dT%H:%M:%S.000'
    new_date_format = "%Y-%m-%d %H:%M:%S"

    for monthFile in monthFileList:

        new_speed_dict = {}

        month_file_path = os.path.join(baseFilePath, monthFile)
        print(month_file_path)

        month_speed_csv = csv.reader(open(month_file_path))
        headers = next(month_speed_csv)

        for row in month_speed_csv:
            speed = row[1]
            date_time = row[4]
            date_tuple = time.strptime(date_time, origin_date_format)
            new_date_time = time.strftime(new_date_format, date_tuple)
            new_date = new_date_time[:10]
            daily_file_folder = os.path.join(baseFilePath, new_date)
            if not os.path.exists(daily_file_folder):
                os.makedirs(daily_file_folder)
            if new_date not in new_speed_dict:
                new_speed_dict[new_date] = []
            lat_lons = row[6]
            sub_lat_idx = [sub_lat.start() for sub_lat in re.finditer('40\.', lat_lons)]
            for i in range(len(sub_lat_idx)):
                new_speed_row = []
                if i == len(sub_lat_idx) - 1:
                    lat_lon_last = lat_lons[sub_lat_idx[i]:].strip()
                    if lat_lon_last.find(' ') > -1:
                        lat_lon = lat_lon_last.rsplit(' ', 1)[0].split(',')
                    else:
                        lat_lon = lat_lon_last.split(',')
                    if len(lat_lon) != 2:
                        continue
                    elif len(lat_lon[-1]) < 3:
                        continue
                    else:
                        lat = lat_lon[0].strip()
                        lon = lat_lon[1].strip()
                else:
                    lat, lon = lat_lons[sub_lat_idx[i]: sub_lat_idx[i+1]].strip().split(',')
                if lon.find(' ') > -1:
                    print(lat_lons)
                    print(sub_lat_idx)
                    print(lat)
                    print(lon)
                new_speed_row.append(lon)
                new_speed_row.append(lat)
                new_speed_row.append(speed)
                new_speed_row.append(new_date_time)
                new_speed_dict[new_date].append(new_speed_row)

        for _date in new_speed_dict:
            daily_speed_data = new_speed_dict[_date]
            daily_speed_csv = csv.writer(open(os.path.join(baseFilePath, _date, 'speed_data.csv'), 'w+', newline=''))
            daily_speed_csv.writerows(daily_speed_data)


if __name__ == '__main__':

    timeRange = pd.date_range('2018-10-01', periods=24, freq="1H")

    # with open('data/speed_data/all_grids_speed.csv') as f:
    #     speed = csv.reader(f)
    #     headers = next(speed)
    #     row1 = next(speed)
    #     print(headers)
    #     print(row1)

    for eathTime in timeRange:
        print('===')

    baseFilePath = "E:/Nicole_bak/Nicole_data/Real-Time Traffic Speed Data"

    # stat_coords()

    # extract_speed_colums_daily(baseFilePath, os.listdir(baseFilePath))
    extract_speed_colums_daily(baseFilePath, ['DOT_Traffic_Speeds_NBE_API_2018_10.csv',
                                              'DOT_Traffic_Speeds_NBE_API_2018_11.csv',
                                              'DOT_Traffic_Speeds_NBE_API_2018_12.csv'])


