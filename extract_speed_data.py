import json
import math
from tqdm import tqdm

divideBound = 5
file_count = 0
floatBitNumber = 8

# longitudeMin = 116.09608
# longitudeMax = 116.71040
# latitudeMin = 39.69086
# latitudeMax = 40.17647
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


def extract_speed_from_segments(segments_file_path, nodes_file_path):

    all_nodes_dict = json.load(open(nodes_file_path))

    all_road_segments = json.load(open(segments_file_path))

    have_speed_segments = []

    for segment in tqdm(all_road_segments, total=len(all_road_segments)):
        speed = segment[-1]
        if speed == 'null' or speed is None:
            continue
        source = segment[2]
        target = segment[3]
        source_lon_lat = all_nodes_dict.get(source, default=None)
        target_lon_lat = all_nodes_dict.get(target, default=None)
        if source_lon_lat is None and target_lon_lat is None:
            continue
        elif:


widthIndex = math.floor((row["longitude"] - longitudeMin) / widthSingle)
heightIndex = math.floor((row["latitude"] - latitudeMin) / heightSingle)

if __name__ == '__main__':

    segments_file_path = 'data/newyork_road_segment.json'
    nodes_file_path = 'data/newyork_nodes.json'

    extract_speed_from_segments(segments_file_path, nodes_file_path)