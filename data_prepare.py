import csv
import time
import re
import os
import pandas as pd
import math
import multiprocessing
import copy
from tqdm import tqdm
from transform_coord.coord_converter import utm_to_latlng
from math import radians, cos, sin, asin, sqrt
from datetime import datetime


def gen_weather():

    date_format = '%Y-%m-%d%I:%M %p'
    new_date_format = '%Y/%m/%d %H:%M:%S'
    new_obs_data = []

    tr_set = set()

    with open('data/observation.csv') as f:
        obs_csv = csv.reader(f)
        headers = next(obs_csv)
        new_headers = ['temp', 'dewPt', 'rh']
        # new_headers = ['temp', 'dewPt', 'rh', 'wdir_cardinal', 'wspd', 'pressure', 'wx_phrase', 'feels_like']
        for i in range(18):
            new_headers.append('wdir_cardinal' + str(i))
        new_headers.append('wspd')
        new_headers.append('pressure')
        for i in range(14):
            new_headers.append('wx_phrase' + str(i))
        new_headers.append('feels_like')
        new_obs_data.append(new_headers)
        for row in tqdm(obs_csv):
            new_row = []
            # temp
            try:
                new_row.append(float(row[2][:2]))
            except:
                new_row.append(float(row[2][:1]))
            # dew_point
            try:
                new_row.append(float(row[3][:2]))
            except:
                new_row.append(float(row[3][:1]))
            # humidity
            new_row.append(float(row[4].split(' ')[0]))
            # wind direction
            for i in range(18):
                new_row.append(0.0)
            # wind_speed
            new_row.append(row[6].split(' ')[0])
            # pressure
            new_row.append(float(row[8].split(' ')[0]))
            # wind type
            for i in range(14):
                new_row.append(0.0)
            # feels like
            try:
                new_row.append(float(row[2][:2]))
            except :
                new_row.append(float(row[2][:1]))

            date_time = row[0] + row[1]
            date_tuple = time.strptime(date_time, date_format)
            year, month, day, hour = date_tuple[:4]
            if year != 2018:
                continue
            if month < 9:
                continue
            # new_date_time = time.strftime(new_date_format, date_tuple)
            _dt = datetime(year, month, day, hour)
            new_date_time = _dt.strftime(new_date_format)
            if new_date_time in tr_set:
                continue
            tr_set.add(new_date_time)
            # new_row.append(new_date_time)
            new_obs_data.append(new_row)

    time_range = pd.date_range('2016-01-01 00:00:00', '2016-12-31 23:59:59', freq="1H")

    for _tr in time_range:
        _dt = datetime.strptime(str(_tr), '%Y-%m-%d %H:%M:%S')
        new_dt = _dt.strftime(new_date_format)
        if new_dt not in tr_set:
            print(new_dt)

    with open('data/weather_no_index.csv', 'w+', newline='') as f:
        weather_csv = csv.writer(f)
        weather_csv.writerows(new_obs_data)

    df_weather = pd.read_csv('data/weather_no_index.csv', header=0)
    time_range = pd.date_range('2016-01-01 00:00:00', '2016-12-31 23:59:59', freq="1H")
    df_weather.insert(0, 'index', time_range)
    df_weather.set_index('index', inplace=True)
    df_weather.to_csv('data/weather.csv')


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


def extract_speed_colums_multi_kernal(baseFilePath, kernel=8):

    pool = []
    dailyFilePathList = []
    file_paths = os.listdir(baseFilePath)
    for file_path in file_paths:
        if file_path.find('\.csv') > -1:
            dailyFilePathList.append(file_path)

    eachKernelFileCount = int(math.ceil(len(dailyFilePathList) / kernel))

    for i in range(0, len(dailyFilePathList), eachKernelFileCount):
        endIndex = i + eachKernelFileCount
        if (endIndex > len(dailyFilePathList)):
            endIndex = len(dailyFilePathList)
        process = multiprocessing.Process(target=extract_speed_colums_daily, args=(baseFilePath, dailyFilePathList[i: endIndex]))
        # pool.apply_async(getGridTaxiSpeed, (baseFilePath, dailyFilePathList[i: endIndex]))
        process.start()
        pool.append(process)

    for process in pool:
        process.join()


def convert_edgelist_from_utm_to_latlon(input, output):

    lonlat_data = []

    with open(input) as f:
        utm_csv = csv.reader(f)
        headers = next(utm_csv)
        print(headers)
        lonlat_data.append(headers)
        for row in utm_csv:
            (lat, lon) = utm_to_latlng(18, float(row[0]), float(row[1]))
            row[0] = lon
            row[1] = lat
            lonlat_data.append(row)

    with open(output, 'w+', newline='') as f:
        lonlat_csv = csv.writer(f)
        lonlat_csv.writerows(lonlat_data)


def get_index_from_list(array, start, end, lat):
    mid_index = (end - start)//2 + start
    if start <= end:
        if array[mid_index][1][1] < lat:
            return get_index_from_list(array, mid_index+1, end, lat)
        elif array[mid_index][1][1] > lat:
            return get_index_from_list(array, start, mid_index-1, lat)
        else:
            return mid_index
    else:
        return end if end > 0 else start


def get_nodes_from_list(array, lat):
    acc = 0.002
    start = get_index_from_list(array, 0, len(array) - 1, lat - acc)
    end = get_index_from_list(array, 0, len(array) - 1, lat + acc)
    return array[start:end]


def get_distance(lng1, lat1, lng2, lat2):
    lng1, lat1, lng2, lat2 = map(radians, [lng1, lat1, lng2, lat2])
    dlon = lng2-lng1
    dlat = lat2-lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    dis = 2*asin(sqrt(a))*6371*1000
    return dis


def gen_edge_h5(input_edgelist, input_pois, output):

    all_edges_pois = []

    poi_coordinate = []
    poi_data = pd.read_csv(input_pois, header=0)
    for index, row in poi_data.iterrows():
        lon = row['longitude']
        lat = row['latitude']
        poi_type = row['poi_type']
        coordinate = [float(lon), float(lat)]
        poi_coordinate.append((poi_type, coordinate))

    poi_coordinate_lat = copy.copy(poi_coordinate)
    poi_coordinate_lat.sort(key=lambda ele: ele[1][1])

    

    new_headers = ['XCoord', 'YCoord', 'LENGTH', 'NUM_NODE', 'spatial_features']

    edges_data = pd.read_csv(input_edgelist, header=0)
    edges_as_nodes = edges_data.groupby('EDGE').agg({'XCoord': 'mean',
                                               'YCoord': 'mean',
                                               'START_NODE': 'nunique',
                                               'END_NODE': 'nunique',
                                               'LENGTH': 'mean'})
    edges_as_nodes['NUM_NODE'] = edges_as_nodes['START_NODE']
    edges_as_nodes.drop(['START_NODE', 'END_NODE'], axis=1, inplace=True)


    edges_coordinate = []
    for index, row in tqdm(edges_as_nodes.iterrows(), 'Processing edges coords'):
        lon = row['XCoord']
        lat = row['YCoord']
        _coordinate = (float(lon), float(lat))
        edges_coordinate.append(_coordinate)

    print("processing pois:")
    for _coords in tqdm(edges_coordinate, 'Processing edges pois'):
        _edge_pois = [0 for i in range(22)]
        (_lon, _lat) = _coords
        _pois = get_nodes_from_list(poi_coordinate_lat, _lat)
        for _poi in _pois:
            (poi_lon, poi_lat) = _poi[1]
            if abs(poi_lon - _lon) <= 0.002:
                poi_type = int(_poi[0])
                _edge_pois[poi_type-1] += 1
        all_edges_pois.append(_edge_pois)

    edges_as_nodes['spatial_features'] = all_edges_pois
    edges_as_nodes.to_csv(output, index=False, columns=new_headers)


if __name__ == '__main__':

    gen_weather()

    # timeRange = pd.date_range('2018-10-01', periods=24, freq="1H")
    #
    # for tr in timeRange:
    #     print(tr)

    # with open('data/speed_data/all_grids_speed.csv') as f:
    #     speed = csv.reader(f)
    #     headers = next(speed)
    #     row1 = next(speed)
    #     print(headers)
    #     print(row1)

    baseFilePath = "E:/Nicole_data/Real-Time Traffic Speed Data"

    # convert_edgelist_from_utm_to_latlon('data/NewYork_Edgelist_utm.csv',
    #                                     'data/NewYork_Edgelist_test.csv')

    # stat_coords()

    # extract_speed_colums_daily(baseFilePath, os.listdir(baseFilePath))
    # extract_speed_colums_daily(baseFilePath, ['DOT_Traffic_Speeds_NBE_API_2018_9.csv',
    #                                           ])

    # extract_speed_colums_multi_kernal(baseFilePath, 4)


    # gen_edge_h5('data/NewYork_Edgelist_latlon.csv',
    #             'data/poi.csv',
    #             'data/edges_data.h5')