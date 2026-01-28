import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_inductive_loader
from model import JGRMInductiveModel
import os


def contrastive_loss(gps_rep, road_rep, temperature=0.07):
    """
    SimCLR-like contrastive loss.
    gps_rep: [B, H]
    road_rep: [B, H]
    """
    gps_rep = r_norm = torch.nn.functional.normalize(gps_rep, dim=1)
    road_rep = t_norm = torch.nn.functional.normalize(road_rep, dim=1)

    # 生成矩阵维度是 [B, B]，其中每个元素表示一个gps_rep与road_rep的相似度
    logits = torch.matmul(r_norm, t_norm.T) / temperature
    # 这是标签，0到B-1，表示每个样本的正确匹配，0表示 0 行的 gps_rep 应该匹配 0 列的 road_rep，依此类推
    labels = torch.arange(gps_rep.size(0)).to(gps_rep.device)

    loss_r = torch.nn.functional.cross_entropy(logits, labels)
    loss_t = torch.nn.functional.cross_entropy(logits.T, labels)

    return (loss_r + loss_t) / 2


def train():
    # Configuration
    traj_path = "/home/harddisk/jxh/trajectory/JGRM/dataset/chengdu/chengdu_1101_1115_data_sample10w.parquet"
    feature_path = (
        "/home/harddisk/jxh/trajectory/JGRM/dataset/chengdu/edge_features.csv"
    )
    batch_size = 32
    num_samples = 5000  # Small sample for testing
    road_feat_dim = 15  # 13 highway + 1 lanes + 1 length
    gps_feat_dim = 6  # rel_lng, rel_lat, f_dist, b_dist, f_azi, b_azi
    embed_size = 128
    hidden_size = 256
    learning_rate = 1e-3
    epochs = 10
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # 1. Load Data
    loader = get_inductive_loader(traj_path, feature_path, batch_size, num_samples)

    # 2. Build Model
    model = JGRMInductiveModel(road_feat_dim, gps_feat_dim, embed_size, hidden_size).to(
        device
    )
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    mse_loss = nn.MSELoss()

    # 3. Training Loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for i, batch in enumerate(loader):
            # 填充之后的值
            route_feats = batch["route_feats"].to(device)
            gps_feats = batch["gps_feats"].to(device)
            route_mask = batch["route_mask"].to(device)  # [B, R]

            # Forward
            road_rep, gps_rep_seq, recon_feats = model(route_feats, gps_feats)

            # Reconstruction Loss (only for non-padded roads)
            # Flatten to [B*R, D] and filter by mask
            recon_feats_flat = recon_feats.view(-1, road_feat_dim)
            route_feats_flat = route_feats.view(-1, road_feat_dim)
            mask_flat = route_mask.view(-1) > 0

            recon_loss = mse_loss(
                recon_feats_flat[mask_flat], route_feats_flat[mask_flat]
            )

            # Matching Loss
            # Pool route_rep and gps_rep_seq
            route_rep_pooled = (road_rep * route_mask.unsqueeze(-1)).sum(
                1
            ) / route_mask.sum(1).unsqueeze(-1)
            # Pool GPS seq (just take mean for now, or use max/last)
            gps_rep_pooled = gps_rep_seq.mean(1)  # Simple mean pooling

            match_loss = contrastive_loss(gps_rep_pooled, route_rep_pooled)

            loss = recon_loss + match_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if i % 10 == 0:
                print(
                    f"Epoch {epoch}, Step {i}, Loss: {loss.item():.4f} (Recon: {recon_loss.item():.4f}, Match: {match_loss.item():.4f})"
                )

        print(f"Epoch {epoch} Average Loss: {total_loss / len(loader):.4f}")

    # Save model
    torch.save(model.state_dict(), "JGRM_Inductive_model.pt")


if __name__ == "__main__":
    train()
