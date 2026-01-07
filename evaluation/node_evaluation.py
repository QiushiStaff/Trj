import sys

sys.path.append(".")
import numpy as np
import pandas as pd
import time
import pickle
import copy

from task import road_cls, speed_inf, time_est, sim_srh, road_cls_prompt
from evluation_utils import (
    get_road,
    fair_sampling,
    get_road_emb_from_traj,
    prepare_data,
    get_seq_emb_from_node,
    get_road_emb_from_traj_before,
)
import torch
import os

torch.set_num_threads(5)

dev_id = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(dev_id)
torch.cuda.set_device(dev_id)


def evaluation(city, exp_path, model_name, start_time):
    route_min_len, route_max_len, gps_min_len, gps_max_len = 10, 100, 10, 256
    model_path = os.path.join(exp_path, "model", model_name)
    embedding_name = model_name.split(".")[0]

    # load task 1 & task2 label
    feature_df = pd.read_csv(
        "/home/harddisk/jxh/trajectory/JGRM/dataset/{}/edge_features.csv".format(city)
    )
    print(feature_df.head())
    num_nodes = len(feature_df)
    print("num_nodes:", num_nodes)

    # load adj，2*16398，起点路段 id、终点路段 id，描述路段之间的连接关系
    edge_index = np.load(
        "/home/harddisk/jxh/trajectory/JGRM/dataset/{}/line_graph_edge_idx.npy".format(
            city
        )
    )
    print("edge_index shape:", edge_index.shape)

    # load origin train data， this is trajectory data including route and gps
    # form of file, parquet loading is faster than pkl
    test_node_data = pd.read_parquet(
        "/home/harddisk/jxh/trajectory/JGRM/dataset/{}/{}_1101_1115_data_sample10w.parquet".format(
            city, city
        )
    )
    road_list = get_road(test_node_data)
    print("number of road obervased in test data: {}".format(len(road_list)))

    # sample train data
    num_samples = 5000  # 'all' or 50000
    if num_samples == "all":
        pass
    elif isinstance(num_samples, int):
        test_node_data = fair_sampling(test_node_data, num_samples)
    road_list = get_road(test_node_data)
    print("number of road obervased after sampling: {}".format(len(road_list)))

    # load model
    seq_model = torch.load(model_path, map_location="cuda:{}".format(dev_id))["model"]
    seq_model.eval()

    print(
        "start time : {}".format(
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
        )
    )
    print("\n=== Evaluation ===")

    # prepare road task dataset
    (
        route_data,
        masked_route_assign_mat,
        gps_data,
        masked_gps_assign_mat,
        route_assign_mat,
        gps_length,
        dataset,
    ) = prepare_data(
        test_node_data, route_min_len, route_max_len, gps_min_len, gps_max_len
    )
    test_node_data = (
        route_data,
        masked_route_assign_mat,
        gps_data,
        masked_gps_assign_mat,
        route_assign_mat,
        gps_length,
        dataset,
    )

    test_node_data_backup = copy.deepcopy(test_node_data)

    update_road = "route"
    emb_path = "/home/harddisk/jxh/trajectory/JGRM/dataset/{}/{}_1101_1115_road_embedding_{}_{}_{}.pkl".format(
        city, city, embedding_name, num_samples, update_road
    )

    gps_road_path = "/home/harddisk/jxh/trajectory/JGRM/dataset/{}/{}_1101_1115_gps_road_embedding_{}_{}_{}.pkl".format(
        city, city, embedding_name, num_samples, update_road
    )

    route_road_path = "/home/harddisk/jxh/trajectory/JGRM/dataset/{}/{}_1101_1115_route_road_embedding_{}_{}_{}.pkl".format(
        city, city, embedding_name, num_samples, update_road
    )

    if (
        os.path.exists(emb_path)
        & os.path.exists(gps_road_path)
        & os.path.exists(route_road_path)
    ):
        # load road embedding from inference result
        road_embedding = torch.load(emb_path, map_location="cuda:{}".format(dev_id))[
            "road_embedding"
        ]
        gps_road_embedding = torch.load(
            gps_road_path, map_location="cuda:{}".format(dev_id)
        )["road_embedding"]
        route_road_embedding = torch.load(
            route_road_path, map_location="cuda:{}".format(dev_id)
        )["road_embedding"]
    else:
        # infer road embedding
        gps_road_embedding, route_road_embedding, road_embedding = (
            get_road_emb_from_traj(
                seq_model,
                test_node_data,
                without_gps=False,
                batch_size=10,
                update_road=update_road,
                city=city,
            )
        )
        road_embedding_before = get_road_emb_from_traj_before(
            seq_model,
            test_node_data_backup,
            without_gps=False,
            batch_size=10,
            update_road=update_road,
            city=city,
        )

        # torch.save({"road_embedding": road_embedding}, emb_path)
        # torch.save({"road_embedding": gps_road_embedding}, gps_road_path)
        # torch.save({"road_embedding": route_road_embedding}, route_road_path)

    print("self task promt")
    # task self
    road_cls_prompt.evaluation(gps_road_embedding, route_road_embedding, feature_df)

    print("task 1 before")

    road_cls.evaluation(road_embedding, feature_df)

    print("task 1")
    # task 1
    road_cls.evaluation(road_embedding_before, feature_df)

    # task 2
    speed_inf.evaluation(road_embedding, feature_df)

    end_time = time.time()
    print("cost time : {:.2f} s".format(end_time - start_time))


if __name__ == "__main__":

    city = "chengdu"

    exp_path = "/home/harddisk/jxh/trajectory/JGRM/exp/JTMR_chengdu_250820183124"
    model_name = "JTMR_chengdu_v1_20_100000_250820183124_19.pt"

    start_time = time.time()
    log_path = os.path.join(exp_path, "evaluation")
    # sys.stdout = Logger(log_path, start_time, stream=sys.stdout)  # record log
    # sys.stderr = Logger(log_path, start_time, stream=sys.stderr)  # record error

    evaluation(city, exp_path, model_name, start_time)
