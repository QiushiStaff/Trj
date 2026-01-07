#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
路段数据缺失对轨迹相似度比较影响的评估脚本
基于JGRM项目的轨迹相似度评估框架，模拟不同程度的Route数据缺失情况
"""

import sys
sys.path.append("/home/harddisk/jxh/trajectory/JGRM")

import numpy as np
import pandas as pd
import time
import pickle
from utils import Logger
import argparse
from JGRM import JGRMModel
from task import road_cls, speed_inf, time_est, sim_srh
from evluation_utils import get_road, fair_sampling, get_seq_emb_from_traj_withRouteOnly, prepare_data
import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

torch.set_num_threads(5)
dev_id = 0
os.environ['CUDA_VISIBLE_DEVICES'] = str(dev_id)
torch.cuda.set_device(dev_id)

def simulate_route_missing(route_assign_mat, route_data, missing_rate, missing_pattern='random', vocab_size=None):
    """
    模拟路段数据缺失
    将选定位置的路段ID设置为padding值 vocab_size
    将选定位置的时间特征设置为0
    更新缺失位置掩码 missing_mask
    
    Args:
        route_assign_mat: 路段分配矩阵 (batch_size, max_route_len)
        route_data: 路段时间特征 (batch_size, max_route_len, 3)
        missing_rate: 缺失比例 (0.0-1.0)
        missing_pattern: 缺失模式 ('random', 'consecutive', 'head', 'tail', 'middle')
        vocab_size: 词汇表大小，用于padding
        
    Returns:
        masked_route_assign_mat: 模拟缺失后的路段分配矩阵
        masked_route_data: 模拟缺失后的路段时间特征
        missing_mask: 缺失位置掩码
    """
    batch_size, max_route_len = route_assign_mat.shape
    masked_route_assign_mat = route_assign_mat.clone()
    masked_route_data = route_data.clone()
    missing_mask = torch.zeros_like(route_assign_mat, dtype=torch.bool)
    
    if vocab_size is None:
        vocab_size = route_assign_mat.max().item()
    
    for i in range(batch_size):
        # 找到有效路段位置（非padding）
        valid_positions = (route_assign_mat[i] != vocab_size).nonzero(as_tuple=True)[0]
        if len(valid_positions) == 0:
            continue
            
        valid_length = len(valid_positions)
        missing_count = int(valid_length * missing_rate)
        
        if missing_count == 0:
            continue
            
        if missing_pattern == 'random':
            # 随机缺失
            missing_indices = torch.randperm(valid_length)[:missing_count]
            missing_positions = valid_positions[missing_indices]
            
        elif missing_pattern == 'consecutive':
            # 连续缺失
            start_pos = torch.randint(0, max(1, valid_length - missing_count + 1), (1,)).item()
            missing_positions = valid_positions[start_pos:start_pos + missing_count]
            
        elif missing_pattern == 'head':
            # 头部缺失
            missing_positions = valid_positions[:missing_count]
            
        elif missing_pattern == 'tail':
            # 尾部缺失
            missing_positions = valid_positions[-missing_count:]
            
        elif missing_pattern == 'middle':
            # 中间缺失
            start_pos = (valid_length - missing_count) // 2
            missing_positions = valid_positions[start_pos:start_pos + missing_count]
        
        # 应用缺失
        masked_route_assign_mat[i, missing_positions] = vocab_size  # 设置为padding值
        masked_route_data[i, missing_positions, :] = 0  # 时间特征设置为0
        missing_mask[i, missing_positions] = True
    
    return masked_route_assign_mat, masked_route_data, missing_mask

def evaluate_missing_impact(city, exp_path, model_name, missing_rates, missing_patterns, start_time):
    """
    评估路段数据缺失对轨迹相似度的影响
    """
    route_min_len, route_max_len, gps_min_len, gps_max_len = 10, 100, 10, 256
    model_path = os.path.join(exp_path, 'model', model_name)
    
    # 加载数据和模型
    feature_df = pd.read_csv("/home/harddisk/jxh/trajectory/JGRM/dataset/{}/edge_features.csv".format(city))
    num_nodes = len(feature_df)
    print("num_nodes:", num_nodes)
    
    geometry_df = pd.read_csv("/home/harddisk/jxh/trajectory/JGRM/dataset/{}/edge_geometry.csv".format(city))
    trans_mat = np.load('/home/harddisk/jxh/trajectory/JGRM/dataset/{}/transition_prob_mat.npy'.format(city))
    trans_mat = torch.tensor(trans_mat)
    
    seq_model = torch.load(model_path, map_location="cuda:{}".format(dev_id))['model']
    seq_model.eval()
    
    # 准备测试数据
    test_seq_data = pickle.load(
        open('/home/harddisk/jxh/trajectory/JGRM/dataset/{}/{}_1101_1115_data_seq_evaluation.pkl'.format(city, city), 'rb'))
    test_seq_data = test_seq_data.sample(10000, random_state=0)  # 使用较小的样本以加快测试
    
    # 准备基础数据
    route_data, masked_route_assign_mat, gps_data, masked_gps_assign_mat, route_assign_mat, gps_length, dataset = prepare_data(
        test_seq_data, route_min_len, route_max_len, gps_min_len, gps_max_len)
    
    # 标准化时间特征
    route_data[:, :, 1] = torch.zeros_like(route_data[:, :, 1]).long()
    route_data[:, :, 2] = torch.full_like(route_data[:, :, 2], fill_value=-1)
    
    # 获取完整数据的轨迹表示作为基准
    print("生成完整数据的轨迹表示...")
    complete_test_data = (route_data, masked_route_assign_mat, gps_data, masked_gps_assign_mat, route_assign_mat, gps_length, dataset)
    complete_seq_embedding = get_seq_emb_from_traj_withRouteOnly(seq_model, complete_test_data, batch_size=1024)
    
    results = []
    
    for missing_pattern in missing_patterns:
        print(f"\n=== 评估缺失模式: {missing_pattern} ===")
        
        for missing_rate in tqdm(missing_rates, desc=f"Testing {missing_pattern}"):
            print(f"\n--- 缺失率: {missing_rate:.2f} ---")
            
            # 模拟缺失
            masked_route_assign_mat_missing, masked_route_data_missing, missing_mask = simulate_route_missing(
                route_assign_mat, route_data, missing_rate, missing_pattern, vocab_size=num_nodes
            )
            
            # 生成缺失数据的轨迹表示
            missing_test_data = (masked_route_data_missing, masked_route_assign_mat_missing, 
                               gps_data, masked_gps_assign_mat, masked_route_assign_mat_missing, gps_length, dataset)
            missing_seq_embedding = get_seq_emb_from_traj_withRouteOnly(seq_model, missing_test_data, batch_size=1024)
            
            # 计算轨迹表示的相似度变化
            embedding_similarity = torch.cosine_similarity(complete_seq_embedding, missing_seq_embedding, dim=1)
            avg_similarity = embedding_similarity.mean().item()
            similarity_std = embedding_similarity.std().item()
            
            # 使用sim_srh.evaluation3评估相似度搜索性能
            print("评估相似度搜索性能...")
            
            # 创建临时的evaluation3函数来捕获结果
            try:
                # 这里需要修改evaluation3函数或创建一个简化版本来返回指标
                # 暂时使用简化的评估方法
                search_results = evaluate_similarity_search_simple(
                    missing_seq_embedding, complete_seq_embedding, fold=3
                )
                
                mean_rank = search_results['mean_rank']
                hr_10 = search_results['hr_10']
                no_hit = search_results['no_hit']
                
            except Exception as e:
                print(f"相似度搜索评估出错: {e}")
                mean_rank, hr_10, no_hit = float('inf'), 0.0, len(missing_seq_embedding)
            
            # 记录结果
            result = {
                'missing_pattern': missing_pattern,
                'missing_rate': missing_rate,
                'avg_embedding_similarity': avg_similarity,
                'embedding_similarity_std': similarity_std,
                'mean_rank': mean_rank,
                'hr_10': hr_10,
                'no_hit': no_hit,
                'total_trajectories': len(missing_seq_embedding)
            }
            results.append(result)
            
            print(f"平均embedding相似度: {avg_similarity:.4f} ± {similarity_std:.4f}")
            print(f"Mean Rank: {mean_rank:.2f}, HR@10: {hr_10:.4f}, No Hit: {no_hit}")
    
    return results

def evaluate_similarity_search_simple(query_embeddings, database_embeddings, fold=3):
    """
    简化的相似度搜索评估
    """
    import faiss
    import torch.nn.functional as F
    
    # 归一化embedding
    query_embeddings = F.normalize(query_embeddings, dim=1)
    database_embeddings = F.normalize(database_embeddings, dim=1)
    
    # 创建FAISS索引
    x = database_embeddings.cpu().numpy()
    index = faiss.IndexFlatL2(x.shape[1])
    index.add(x)
    
    # 搜索
    q = query_embeddings.cpu().numpy()
    k = min(1000, len(database_embeddings))
    D, I = index.search(q, k)
    
    # 计算指标
    hit_count = 0
    rank_sum = 0
    no_hit_count = 0
    
    for i, search_results in enumerate(I):
        if i in search_results:
            rank = np.where(search_results == i)[0][0] + 1  # 排名从1开始
            rank_sum += rank
            if rank <= 10:
                hit_count += 1
        else:
            no_hit_count += 1
    
    total_queries = len(query_embeddings)
    valid_queries = total_queries - no_hit_count
    
    mean_rank = rank_sum / total_queries if total_queries > 0 else float('inf')
    hr_10 = hit_count / valid_queries if valid_queries > 0 else 0.0
    
    return {
        'mean_rank': mean_rank,
        'hr_10': hr_10,
        'no_hit': no_hit_count
    }

def plot_results(results, save_path=None):
    """
    绘制实验结果
    """
    df = pd.DataFrame(results)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('路段数据缺失对轨迹相似度影响分析', fontsize=16)
    
    # 1. Embedding相似度变化
    ax1 = axes[0, 0]
    for pattern in df['missing_pattern'].unique():
        pattern_data = df[df['missing_pattern'] == pattern]
        ax1.plot(pattern_data['missing_rate'], pattern_data['avg_embedding_similarity'], 
                marker='o', label=pattern, linewidth=2)
        ax1.fill_between(pattern_data['missing_rate'], 
                        pattern_data['avg_embedding_similarity'] - pattern_data['embedding_similarity_std'],
                        pattern_data['avg_embedding_similarity'] + pattern_data['embedding_similarity_std'],
                        alpha=0.2)
    ax1.set_xlabel('缺失率')
    ax1.set_ylabel('平均Embedding相似度')
    ax1.set_title('Embedding相似度 vs 缺失率')
    ax1.legend()
    ax1.grid(True)
    
    # 2. Mean Rank变化
    ax2 = axes[0, 1]
    for pattern in df['missing_pattern'].unique():
        pattern_data = df[df['missing_pattern'] == pattern]
        ax2.plot(pattern_data['missing_rate'], pattern_data['mean_rank'], 
                marker='s', label=pattern, linewidth=2)
    ax2.set_xlabel('缺失率')
    ax2.set_ylabel('Mean Rank')
    ax2.set_title('Mean Rank vs 缺失率')
    ax2.legend()
    ax2.grid(True)
    
    # 3. HR@10变化
    ax3 = axes[1, 0]
    for pattern in df['missing_pattern'].unique():
        pattern_data = df[df['missing_pattern'] == pattern]
        ax3.plot(pattern_data['missing_rate'], pattern_data['hr_10'], 
                marker='^', label=pattern, linewidth=2)
    ax3.set_xlabel('缺失率')
    ax3.set_ylabel('HR@10')
    ax3.set_title('HR@10 vs 缺失率')
    ax3.legend()
    ax3.grid(True)
    
    # 4. 热力图显示不同模式的整体表现
    ax4 = axes[1, 1]
    pivot_data = df.pivot_table(values='avg_embedding_similarity', 
                               index='missing_pattern', 
                               columns='missing_rate')
    sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='viridis', ax=ax4)
    ax4.set_title('不同缺失模式的Embedding相似度热力图')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"结果图保存至: {save_path}")
    
    plt.show()

def main():
    """
    主函数
    """
    # 实验配置
    city = 'chengdu'
    exp_path = '/home/harddisk/jxh/trajectory/JGRM/exp/JTMR_chengdu_250820183124'
    model_name = 'JTMR_chengdu_v1_20_100000_250820183124_19.pt'
    
    # 缺失率范围
    missing_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    
    # 缺失模式
    missing_patterns = ['random', 'consecutive', 'head', 'tail', 'middle']
    
    start_time = time.time()
    
    print("开始路段数据缺失影响评估实验...")
    print(f"城市: {city}")
    print(f"缺失率范围: {missing_rates}")
    print(f"缺失模式: {missing_patterns}")
    
    # 运行实验
    results = evaluate_missing_impact(
        city, exp_path, model_name, missing_rates, missing_patterns, start_time
    )
    
    # 保存结果
    results_df = pd.DataFrame(results)
    results_path = f'/home/harddisk/jxh/trajectory/JGRM/results/route_missing_impact_results_{city}.csv'
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    results_df.to_csv(results_path, index=False)
    print(f"实验结果保存至: {results_path}")
    
    # 绘制结果图
    plot_path = f'/home/harddisk/jxh/trajectory/JGRM/results/route_missing_impact_plot_{city}.png'
    plot_results(results, save_path=plot_path)
    
    # 打印总结
    print("\n" + "="*60)
    print("实验总结:")
    print("="*60)
    
    for pattern in missing_patterns:
        pattern_results = results_df[results_df['missing_pattern'] == pattern]
        print(f"\n{pattern}模式:")
        print(f"  - 最大影响(缺失率0.8): Embedding相似度 {pattern_results.iloc[-1]['avg_embedding_similarity']:.3f}")
        print(f"  - 最大影响(缺失率0.8): HR@10 {pattern_results.iloc[-1]['hr_10']:.3f}")
        print(f"  - 最大影响(缺失率0.8): Mean Rank {pattern_results.iloc[-1]['mean_rank']:.2f}")
    
    end_time = time.time()
    print(f"\n总耗时: {end_time - start_time:.2f}秒")

if __name__ == '__main__':
    main()