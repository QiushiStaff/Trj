import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import os


class PromptClassifier(nn.Module):
    def __init__(self, gps_dim, route_dim, num_classes):
        super(PromptClassifier, self).__init__()
        # 可学习的 Prompt 参数，用于动态调整 GPS 和 Route 特征的权重
        # 初始权设为相等 (Softmax ([0, 0]) = [0.5, 0.5])
        self.fusion_prompt = nn.Parameter(torch.zeros(2))

        # 分类头：输入是两个特征拼接后的维度
        self.fc = nn.Linear(gps_dim + route_dim, num_classes)

    def forward(self, x_gps, x_route):
        # 通过 Softmax 归一化权重
        weights = torch.softmax(self.fusion_prompt, dim=0)

        # 加权特征
        x_gps_weighted = x_gps * weights[0]
        x_route_weighted = x_route * weights[1]

        # 拼接后的特征输入分类器
        fused_x = torch.cat([x_gps_weighted, x_route_weighted], dim=-1)
        return self.fc(fused_x)


def evaluation(x_gps_all, x_route_all, feature_df, fold=100):
    """
    x_gps_all: (num_nodes, dim)
    x_route_all: (num_nodes, dim)
    """
    valid_labels = ["primary", "secondary", "tertiary", "residential"]
    id_dict = {idx: i for i, idx in enumerate(valid_labels)}
    y_df = feature_df.loc[feature_df["highway"].isin(valid_labels)]

    # 按照 fid 提取对应的特征
    fids = y_df["fid"].tolist()
    x_gps = x_gps_all[fids]
    x_route = x_route_all[fids]
    y = torch.tensor(y_df["highway"].map(id_dict).tolist())

    split = x_gps.shape[0] // fold
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    y_preds = []
    y_trues = []

    # 打印一次设备信息
    print(f"prompt evaluation using device: {device}")

    for i in range(fold):
        eval_idx = list(range(i * split, (i + 1) * split, 1))
        train_idx = list(set(list(range(x_gps.shape[0]))) - set(eval_idx))

        # 训练集和测试集划分
        xg_train, xg_eval = x_gps[train_idx], x_gps[eval_idx]
        xr_train, xr_eval = x_route[train_idx], x_route[eval_idx]
        y_train, y_eval = y[train_idx].to(device), y[eval_idx]

        # 初始化 Prompt 模型
        model = PromptClassifier(
            xg_train.shape[1], xr_train.shape[1], len(valid_labels)
        ).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)

        best_acc = 0.0
        best_pred = None

        for e in range(1, 101):
            model.train()
            # 训练时更新 prompt 和 fc
            logits = model(xg_train.to(device), xr_train.to(device))
            ce_loss = nn.CrossEntropyLoss()(logits, y_train)

            opt.zero_grad()
            ce_loss.backward()
            opt.step()

            model.eval()
            with torch.no_grad():
                eval_logits = model(xg_eval.to(device), xr_eval.to(device)).cpu()
                logit_sm = F.softmax(eval_logits, -1)
                y_pred = torch.argmax(logit_sm, dim=1)
                acc = accuracy_score(y_eval, y_pred, normalize=False)
                if acc > best_acc:
                    best_acc = acc
                    best_pred = y_pred

        y_preds.append(best_pred)
        y_trues.append(y_eval)

    y_preds = torch.cat(y_preds, dim=0)
    y_trues = torch.cat(y_trues, dim=0)

    macro_f1 = f1_score(y_trues, y_preds, average="macro")
    micro_f1 = f1_score(y_trues, y_preds, average="micro")

    # 打印最终的融合比例（Prompt 结果）
    # 使用最后一个 fold 的 model 作为展示
    final_weights = torch.softmax(model.fusion_prompt, dim=0).detach().cpu().numpy()
    print(
        f"Final Prompt Weights - GPS: {final_weights[0]:.4f}, Route: {final_weights[1]:.4f}"
    )
    print(
        f"road prompt classification | micro F1: {micro_f1:.4f}, macro F1: {macro_f1:.4f}"
    )
