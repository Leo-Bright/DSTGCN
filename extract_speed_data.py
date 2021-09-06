import json
import math
import numpy as np
from tqdm import tqdm

divideBound = 5
file_count = 0
floatBitNumber = 8

longitudeMin = -74.891
longitudeMax = -73.72294
latitudeMin = 40.52419
latitudeMax = 40.90706

# 网格的划分
widthSingle = 0.01 / math.cos(latitudeMin / 180 * math.pi) / divideBound
width = math.floor((longitudeMax - longitudeMin) / widthSingle)
heightSingle = 0.01 / divideBound
height = math.floor((latitudeMax - latitudeMin) / heightSingle)
print("height = ", height)
print("heightSingle = ", heightSingle)
print("width = ", width)
print("widthSingle = ", widthSingle)


def extract_speed_from_segments(segments_file_path, nodes_file_path, speed_file_path):

    all_nodes_dict = json.load(open(nodes_file_path))

    all_road_segments = json.load(open(segments_file_path))

    location_index_speeds = {}
    all_location_index = set()

    print('all road segments: ', len(all_road_segments))
    for segment in tqdm(all_road_segments, total=len(all_road_segments)):
        speed = segment[-1]
        if speed == 'null' or speed is None or speed == 0 or speed == '0':
            continue
        source = str(segment[2])
        target = str(segment[3])

        if source not in all_nodes_dict and target not in all_nodes_dict:
            continue

        if source not in all_nodes_dict:
            source_lon = 0
            source_lat = 0
        else:
            source_lon_lat_info = all_nodes_dict[source]
            source_lon = source_lon_lat_info[-1][0]
            source_lat = source_lon_lat_info[-1][1]

        if target not in all_nodes_dict:
            target_lon = 0
            target_lat = 0

        else:
            target_lon_lat_info = all_nodes_dict[target]
            target_lon = target_lon_lat_info[-1][0]
            target_lat = target_lon_lat_info[-1][1]

        segment_lon = float(source_lon) + float(target_lon) / 2
        segment_lat = float(source_lat) + float(target_lat) / 2

        column_index = math.floor((segment_lon - longitudeMin) / widthSingle)
        row_index = math.floor((segment_lat - latitudeMin) / heightSingle)

        location_index = str(row_index) + ',' + str(column_index)
        all_location_index.add(location_index)
        if location_index not in location_index_speeds:
            location_index_speeds[location_index] = list()
        location_index_speeds[location_index].append(speed)

    all_grid_speed_dict = {}
    for row in range(height + 1):
        for column in range(width + 1):
            location_index = str(row) + ',' + str(column)
            if location_index not in location_index_speeds:
                all_grid_speed_dict[location_index] = 0
            else:
                all_grid_speed_dict[location_index] = np.mean(location_index_speeds[location_index])
                print(np.mean(location_index_speeds[location_index]))

    with open(speed_file_path, 'w+') as speed_file:
        speed_file.write(json.dumps(all_grid_speed_dict))
    print('Done....')


if __name__ == '__main__':

    segments_file_path = 'data/newyork_road_segment.json'
    nodes_file_path = 'data/newyork_nodes.json'
    speed_file_path = 'data/all_grids_speed_data.json'

    extract_speed_from_segments(segments_file_path, nodes_file_path, speed_file_path)