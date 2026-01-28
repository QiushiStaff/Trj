import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import pickle
from geopy.distance import geodesic
import torch.nn.utils.rnn as rnn_utils
from datetime import datetime


def calculate_azimuth(lat1, lon1, lat2, lon2):
    """
    Calculate the azimuth between two points.
    Returns value in radians, normalized to [-1, 1] by dividing by PI.
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    azimuth = np.arctan2(y, x)
    return azimuth / np.pi


def calculate_dist(lat1, lon1, lat2, lon2):
    """
    Calculate distance in km.
    """
    # Simple haversine or approximate for speed if many points
    # Using a fast approximate version
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


class InductiveDataset(Dataset):
    def __init__(self, traj_data, road_features_df, route_max_len=100, gps_max_len=256):
        self.traj_data = traj_data
        self.route_max_len = route_max_len
        self.gps_max_len = gps_max_len

        # 1. Preprocess Road Features
        self.road_features = self._preprocess_road_features(road_features_df)

        # 2. Extract Trajectory Lengths
        self.gps_length = self._get_gps_lengths(traj_data, route_max_len)

    # 获得道路特征，把道路类型（one-hot 向量表示） 拼接 车道数量 拼接 道路长度 ，作为路段的特征
    def _preprocess_road_features(self, df):
        # highway one-hot
        highway_list = [
            "living_street",
            "motorway",
            "motorway_link",
            "primary",
            "primary_link",
            "residential",
            "secondary",
            "secondary_link",
            "tertiary",
            "tertiary_link",
            "trunk",
            "trunk_link",
            "unclassified",
        ]
        highway_map = {h: i for i, h in enumerate(highway_list)}

        # Fill lanes
        df["lanes"] = pd.to_numeric(df["lanes"], errors="coerce").fillna(2)
        # Normalize length
        max_len = df["length"].max()
        df["norm_length"] = df["length"] / max_len
        # Normalize lanes (assume max 10)
        df["norm_lanes"] = df["lanes"] / 10.0

        # 相当于把道路类型（one-hot 向量表示）+ 车道数量 + 道路长度 作为路段的特征
        features = []
        for _, row in df.iterrows():
            # 这是一个 one-hot 向量表示道路的类型
            h_feat = [0] * len(highway_list)
            if row["highway"] in highway_map:
                h_feat[highway_map[row["highway"]]] = 1

            # 这是一个拼接的操作，把后面两个属性拼到 h_feat 的后面。
            feat = h_feat + [row["norm_lanes"], row["norm_length"]]
            features.append(feat)

        return torch.tensor(features, dtype=torch.float32)

    # 获得每个轨迹下，每个路段有多少个轨迹点，[taj_num,route_num],填充到 max_len 的长度 [taj_num,max_len]
    def _get_gps_lengths(self, data, max_len):
        def split_duplicate(opath_list):
            length_list = []
            if len(opath_list) == 0:
                return [0] * max_len

            subsequence = [opath_list[0]]
            for i in range(0, len(opath_list) - 1):
                if opath_list[i] == opath_list[i + 1]:
                    subsequence.append(opath_list[i + 1])
                else:
                    length_list.append(len(subsequence))
                    subsequence = [opath_list[i + 1]]
            length_list.append(len(subsequence))

            # Use slicing to handle potential negative padding lengths and ensure length is exactly max_len
            return (length_list + [0] * max_len)[:max_len]

        lengths = data["opath_list"].apply(lambda x: split_duplicate(x)).tolist()
        return torch.tensor(lengths, dtype=torch.int)

    # 获得每个轨迹点的特征
    def _process_gps(self, lngs, lats):
        # 1-2. Relative Pos
        rel_lngs = lngs - lngs[0]
        rel_lats = lats - lats[0]

        # 3-4. F/B Dist
        f_dist = np.zeros_like(lngs)
        b_dist = np.zeros_like(lngs)
        for i in range(len(lngs)):
            if i < len(lngs) - 1:
                f_dist[i] = calculate_dist(lats[i], lngs[i], lats[i + 1], lngs[i + 1])
            if i > 0:
                b_dist[i] = calculate_dist(lats[i - 1], lngs[i - 1], lats[i], lngs[i])

        # 5-6. F/B Azimuth
        f_azi = np.zeros_like(lngs)
        b_azi = np.zeros_like(lngs)
        for i in range(len(lngs)):
            if i < len(lngs) - 1:
                f_azi[i] = calculate_azimuth(lats[i], lngs[i], lats[i + 1], lngs[i + 1])
            if i > 0:
                b_azi[i] = calculate_azimuth(lats[i - 1], lngs[i - 1], lats[i], lngs[i])

        gps_feat = np.stack([rel_lngs, rel_lats, f_dist, b_dist, f_azi, b_azi], axis=1)
        return torch.tensor(gps_feat, dtype=torch.float32)

    def __len__(self):
        return len(self.traj_data)

    def __getitem__(self, idx):
        row = self.traj_data.iloc[idx]

        # Route Data (Road features)
        cpath = row["cpath_list"]
        route_feats = self.road_features[cpath]

        # GPS Data
        lngs = np.array(row["lng_list"])
        lats = np.array(row["lat_list"])
        gps_feats = self._process_gps(lngs, lats)

        # Padding masks/assign mats
        # In inductive mode, we still need to know which GPS point belongs to which road in the sequence
        # or we just use the gps_length to handle the mapping.

        return {
            # 每个路段的特征
            "route_feats": route_feats,
            # 每个轨迹点的特征
            "gps_feats": gps_feats,
            # 当前轨迹下，每个路段有多少个轨迹点
            "gps_length": self.gps_length[idx],
            # 每个轨迹的路段序列
            "cpath": torch.tensor(
                cpath, dtype=torch.long
            ),  # for internal reference/matching loss if needed
        }


def collate_fn(batch):
    route_feats = [item["route_feats"] for item in batch]
    gps_feats = [item["gps_feats"] for item in batch]
    gps_length = torch.stack([item["gps_length"] for item in batch])
    cpath = [item["cpath"] for item in batch]

    # Pad sequences
    route_feats_pad = rnn_utils.pad_sequence(
        route_feats, batch_first=True, padding_value=0.0
    )
    gps_feats_pad = rnn_utils.pad_sequence(
        gps_feats, batch_first=True, padding_value=0.0
    )

    # Create mask for route_feats (since it's padded)
    route_mask = []
    for rf in route_feats:
        # 这个轨迹下，路段集合的 size。
        mask = torch.ones(rf.shape[0])
        route_mask.append(mask)
    # route_mask_pad 有多少个路段就有多少个 1，padding 的部分是 0
    route_mask_pad = rnn_utils.pad_sequence(
        route_mask, batch_first=True, padding_value=0.0
    )

    return {
        "route_feats": route_feats_pad,
        "gps_feats": gps_feats_pad,
        "gps_length": gps_length,
        "route_mask": route_mask_pad,
        "cpath": cpath,  # List of Variable length tensors
    }


def get_inductive_loader(traj_path, feature_path, batch_size=64, num_samples=1000):
    traj_df = pd.read_parquet(traj_path)
    # Sample if needed
    if num_samples < len(traj_df):
        traj_df = traj_df.sample(num_samples).reset_index(drop=True)

    feature_df = pd.read_csv(feature_path)

    dataset = InductiveDataset(traj_df, feature_df)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    return loader
