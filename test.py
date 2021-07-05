import csv
import time


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


if __name__ == '__main__':
    stat_coords()



