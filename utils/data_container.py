import datetime
import math
import json
from typing import Tuple

import dgl
import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from transform_coord.coord_converter import convert_by_type
from utils.load_config import get_attribute

# longitudeMin = 116.09608
# longitudeMax = 116.71040
# latitudeMin = 39.69086
# latitudeMax = 40.17647

longitudeMin = -74.891
longitudeMax = -73.72294
latitudeMin = 40.52419
latitudeMax = 40.90706

# 坐标转换
longitudeMin, latitudeMin = convert_by_type(lng=longitudeMin, lat=latitudeMin, type="g2w")
longitudeMax, latitudeMax = convert_by_type(lng=longitudeMax, lat=latitudeMax, type="g2w")

# 1110m分为多少份
divideBound = 5

# 网格的划分
widthSingle = 0.01 / math.cos(latitudeMin / 180 * math.pi) / divideBound
width = math.floor((longitudeMax - longitudeMin) / widthSingle)
heightSingle = 0.01 / divideBound
height = math.floor((latitudeMax - latitudeMin) / heightSingle)


# 得到传入的node_list的邻居节点
def get_neighbors(network: nx.Graph, nodes):
    nodes_set = set()
    for node in nodes:
        nodes_set = nodes_set.union({n for n in network.neighbors(node)})
    return nodes_set


def collate_fn(batch):
    """
    Args:
        batch: list, shape (batch_size, XXX)

    Returns:
        batched data
    """
    ret = list()
    for idx, item in enumerate(zip(*batch)):
        if isinstance(item[0], torch.Tensor):
            if idx < 3:  # spatial and temporal features
                ret.append(torch.cat(item))
            else:  # overall features and y
                ret.append(torch.stack(item))
        elif isinstance(item[0], dgl.DGLGraph):
            ret.append(dgl.batch(item))
        else:
            raise ValueError(f'batch must contain tensors or graphs; found {type(item[0])}')
    return tuple(ret)


# 填充速度文件
def fill_speed(speed_data):
    date_range = pd.date_range(start="2018-10-01", end="2018-12-31", freq="1H")[:-1]
    speed_data = speed_data.resample(rule="1H").mean()
    assert date_range[0] in speed_data.index
    assert date_range[-1] in speed_data.index
    one_week, two_week = datetime.timedelta(days=7), datetime.timedelta(days=14)
    for date in tqdm(date_range, 'Fill speed'):
        if any(speed_data.loc[date].isna()):
            for idx in [date - one_week, date + one_week, date - two_week, date + two_week]:
                if idx in speed_data.index and all(speed_data.loc[idx].notna()):
                    speed_data.loc[date] = speed_data.loc[idx]
                    break
            else:
                raise ValueError(f"not find time slot for {date}")
    return speed_data


class AccidentDataset(Dataset):
    def __init__(self,
                 k_order: int,
                 network: nx.Graph,
                 node_attr: pd.DataFrame,
                 accident: pd.DataFrame,
                 weather: pd.DataFrame,
                 speed: pd.DataFrame,
                 sf_scaler: Tuple[np.ndarray, np.ndarray] = None,
                 tf_scaler: Tuple[np.ndarray, np.ndarray] = None,
                 ef_scaler: Tuple[np.ndarray, np.ndarray] = None):
        self.k_order = k_order
        self.network = network
        self.nodes = node_attr
        self.accident = accident
        self.weather = weather
        self.speed = speed

        self.sf_scaler = sf_scaler
        self.tf_scaler = tf_scaler
        self.ef_scaler = ef_scaler

    def __getitem__(self, sample_id: int):
        """
        :param sample_id:
        :return: a 5-tuple:
                g: a subgraph around `node_id` with `k_order`
                spatial_features: spatial and static features of each node, shape [N, F_1] (pois + node_num + road_len)
                temporal_features: dynamic features of each node, shape [N, F_2, T] (speed)
                external_features: static features of subgraph, shape [F_3] (weather + calendar)
                y: whether a accident happened at the specific time
        """
        # get `node_id` from `sample_id`
        _, _, accident_time, node_id, target = self.accident.iloc[sample_id]

        # 得到邻居节点
        neighbors = nx.single_source_shortest_path_length(self.network, node_id, cutoff=self.k_order)

        # neighbors -> list
        neighbors.pop(node_id, None)
        neighbors = [node_id] + sorted(neighbors.keys())

        # get subgraph
        sub_graph = nx.subgraph(self.network, neighbors)
        sub_graph = nx.relabel_nodes(sub_graph, dict(zip(neighbors, range(len(neighbors)))))
        sub_graph.add_edges_from([(v, v) for v in sub_graph.nodes])
        g = dgl.DGLGraph(sub_graph)

        # get temporal_features (speed)
        date_range = pd.date_range(end=accident_time.strftime("%Y%m%d %H"), freq="1H", periods=24)
        selected_time = self.speed.loc[date_range]

        selected_nodes = self.nodes.loc[neighbors]

        spatial_features = selected_nodes['spatial_features'].tolist()

        x_ids = np.floor((selected_nodes['XCoord'].values - longitudeMin) / widthSingle).astype(np.int)
        y_ids = np.floor((selected_nodes['YCoord'].values - latitudeMin) / heightSingle).astype(np.int)

        temporal_features = selected_time[list(map(lambda ids: f'{ids[0]},{ids[1]}', zip(y_ids, x_ids)))].values.transpose()

        # get external_features (weather + calendar)
        # 天气使用预测的前一时刻近似, 时间点信息有月,日,周几,时间点,是否为周末
        _dr = date_range[-1]
        weather = self.weather.loc[_dr].tolist()
        external_features = weather + [accident_time.month, accident_time.day, accident_time.dayofweek,
                                       accident_time.hour, int(accident_time.dayofweek >= 5)]

        if self.sf_scaler is not None:
            mean, std = self.sf_scaler
            _sf = np.array(spatial_features)
            spatial_features = (_sf - mean) / std
        if self.tf_scaler is not None:
            mean, std = self.tf_scaler
            temporal_features = (np.array(temporal_features) - mean) / std
        if self.ef_scaler is not None:
            mean, std = self.ef_scaler
            _ef = np.array(external_features)
            external_features = (_ef - mean) / std

        # [N, F_1]
        spatial_features = torch.tensor(spatial_features).float()
        # [N, F_2, T]
        temporal_features = torch.tensor(temporal_features).unsqueeze(1).float()
        # [F_3]
        external_features = torch.tensor(external_features).float()

        target = torch.tensor(target).float()

        return g, spatial_features, temporal_features, external_features, target

    def __len__(self):
        return len(self.accident) - 1


def get_data_loaders(k_order, batch_size):
    """
    Args:
        k_order: int
        batch_size: int

    Returns:
        data_loader: DataLoader
    """
    network_path = r'../data/newyork_roadnet.gpickle'
    node_attr_path = r'../data/edges_data.h5'
    accident_path = r'../data/accident_10_201810_201812.h5'
    # weather_path = "../data/weather.h5"
    weather_path = "../data/weather.csv"
    # speed_path = "../data/all_grids_speed.h5"
    speed_path = "../data/speed_data/all_grids_speed_201810_201812.csv"

    sf_mean, sf_std = np.array(get_attribute('spatial_features_mean')), np.array(get_attribute('spatial_features_std'))
    tf_mean, tf_std = np.array(get_attribute('temporal_features_mean')), np.array(
        get_attribute('temporal_features_std'))
    ef_mean, ef_std = np.array(get_attribute('external_features_mean')), np.array(
        get_attribute('external_features_std'))

    network = nx.read_gpickle(network_path)
    # XCoord  YCoord LENGTH  NUM_NODE
    # nodes = pd.read_hdf(node_attr_path)
    nodes = pd.read_csv(node_attr_path)
    nodes['spatial_features'] = nodes.apply(lambda x: json.loads(x['spatial_features']), axis=1)
    # nodes.drop('spatial_features')
    # nodes['spatial_features'] = nodes['spatial_features_tmp']
    # nodes.drop('spatial_features_tmp')
    # 'valid_time', 'temp', 'dewPt', 'rh', 'pressure', 'wspd', 'feels_like',  ......
    # weather = pd.read_hdf(weather_path)
    weather = pd.read_csv(weather_path, index_col=0, parse_dates=True)

    # speed = fill_speed(pd.read_hdf(speed_path))
    speed_data = pd.read_csv(speed_path, index_col=0, parse_dates=True)
    speed = fill_speed(speed_data)

    dls = dict()
    for key in ['train', 'validate', 'test']:
        # longitude   latitude  time  node_id  accident
        accident = pd.read_hdf(accident_path, key=key)
        dataset = AccidentDataset(k_order, network, nodes, accident, weather, speed,
                                  sf_scaler=(sf_mean, sf_std),
                                  tf_scaler=(tf_mean, tf_std),
                                  ef_scaler=(ef_mean, ef_std))
        dls[key] = DataLoader(dataset=dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              drop_last=False,
                              collate_fn=collate_fn,
                              num_workers=0)
    return dls


if __name__ == "__main__":
    dls = get_data_loaders(get_attribute("K_hop"), get_attribute('batch_size'))
    for key in ["train", "validate", "test"]:
        first = True
        exception = False
        sf = []
        tf = []
        ef = []
        for step, (g, spatial_features, temporal_features, external_features, y) in tqdm(enumerate(dls[key])):
            _sf = list(spatial_features.size())
            _tf = list(temporal_features.size())
            _ef = list(external_features.size())
            try:
                assert _sf[0] == 26617
                assert _sf[1] == 22
                assert _tf[0] == 26617
                assert _tf[1] == 1
                assert _tf[2] == 24
                assert _ef[0] == 64
                assert _ef[1] == 43
            except:
                print('==========')
                print(_sf)
                print(spatial_features)
                print('==========')
                print(_tf)
                print(temporal_features)
                print('==========')
                print(_ef)
                print(external_features)
                print('==========')

            # if first:
            #     sf = _sf
            #     tf = _tf
            #     ef = _ef
            #     first = False
            # for idx, e in sf:
            #     if _sf[idx] != e:
            #         exception = True
            #         print('sf:', sf)
            #         print('_sf:', _sf)
            #
            # for idx, e in tf:
            #     if _tf[idx] != e:
            #         exception = True
            #         print('tf:', tf)
            #         print('_tf:', _tf)
            #
            # for idx, e in ef:
            #     if _ef[idx] != e:
            #         exception = True
            #         print('ef:', ef)
            #         print('_ef:', _ef)

            # input_data, truth_data
            # if step == 0:
            #     print(g, spatial_features.shape, temporal_features.shape, external_features.shape, y.shape)
            pass
