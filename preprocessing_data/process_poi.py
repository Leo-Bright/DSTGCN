import pandas as pd
from transform_coord.coord_converter import convert_by_type
from preprocessing_data.generate_data import point_is_in_girds


def clean_pois(POIFilePath, outRoadsPOIPath):
    origin_pois = pd.read_csv(POIFilePath, header=0)
    index_list = []
    for index, row in origin_pois.iterrows():
        longitude = row["LON"]
        latitude = row["LAT"]
        if not point_is_in_girds(longitude=longitude, latitude=latitude):
            print(f"ignore poi {index}")
            continue
        index_list.append(index)
    pois = origin_pois.iloc[index_list][["LON", "LAT", "TYPE_NUMBER"]].reset_index(drop=True)
    pois.columns = ["longitude", "latitude", "poi_type"]
    # convert longitude and latitude
    coords = list(zip(pois["longitude"].values.tolist(), pois["latitude"].values.tolist()))
    convert_corrds = []
    for coord in coords:
        convert_lng, convert_lat = convert_by_type(lng=coord[0], lat=coord[1], type="g2w")
        convert_corrds.append([convert_lng, convert_lat])
    pois["longitude"] = pd.Series(list(zip(*convert_corrds))[0])
    pois["latitude"] = pd.Series(list(zip(*convert_corrds))[1])
    pois.to_csv(outRoadsPOIPath, index=False)
    print(outRoadsPOIPath, " writes successfully.")


def process_pois(input, output):

    origin_pois = pd.read_csv(input, header=0)

    lons = []
    lats = []

    index_list = []
    for index, row in origin_pois.iterrows():
        geom = row['the_geom']
        lonlat = geom[7:-1]
        longitude, latitude = lonlat.strip().split(' ')
        lons.append(longitude)
        lats.append(latitude)
        # longitude = row["LON"]
        # latitude = row["LAT"]
        if not point_is_in_girds(longitude=float(longitude), latitude=float(latitude)):
            print(f"ignore poi {index}")
            continue
        index_list.append(index)
    origin_pois['longitude'] = lons
    origin_pois['latitude'] = lats
    # pois = origin_pois.iloc[index_list][["LON", "LAT", "TYPE_NUMBER"]].reset_index(drop=True)
    pois = origin_pois.iloc[index_list][["longitude", "latitude", "FACI_DOM"]].reset_index(drop=True)
    pois.columns = ["longitude", "latitude", "poi_type"]
    # convert longitude and latitude
    # coords = list(zip(pois["longitude"].values.tolist(), pois["latitude"].values.tolist()))
    # convert_corrds = []
    # for coord in coords:
    #     convert_lng, convert_lat = convert_by_type(lng=coord[0], lat=coord[1], type="g2w")
    #     convert_corrds.append([convert_lng, convert_lat])
    # pois["longitude"] = pd.Series(list(zip(*convert_corrds))[0])
    # pois["latitude"] = pd.Series(list(zip(*convert_corrds))[1])
    pois.to_csv(output, index=False)
    print(output, " writes successfully.")


if __name__ == "__main__":
    # POIFilePath = "/home/yule/文档/POI data/poi_analyse.csv"
    # outRoadsPOIPath = "/home/yule/桌面/traffic_accident_data/poi.csv"
    # clean_pois(POIFilePath, outRoadsPOIPath)

    poi_file_path = 'E:/Nicole_bak/Nicole_data/POI NYC/Point_Of_Interest.csv'

    cleaned_poi_file_path = '../data/poi_test.csv'

    process_pois(poi_file_path, cleaned_poi_file_path)