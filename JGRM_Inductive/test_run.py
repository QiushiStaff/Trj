import torch
from dataset import get_inductive_loader
from model import JGRMInductiveModel

def test():
    traj_path = "/Users/lucky-j/Project/pycode/trajectory/JGRM/dataset/chengdu/chengdu_1101_1115_data_sample10w.pkl"
    feature_path = "/Users/lucky-j/Project/pycode/trajectory/JGRM/dataset/chengdu/edge_features.csv"
    
    loader = get_inductive_loader(traj_path, feature_path, batch_size=4, num_samples=20)
    
    road_feat_dim = 15
    gps_feat_dim = 6
    embed_size = 128
    hidden_size = 256
    
    device = torch.device("cpu") # Test on CPU
    model = JGRMInductiveModel(road_feat_dim, gps_feat_dim, embed_size, hidden_size).to(device)
    
    for batch in loader:
        route_feats = batch['route_feats'].to(device)
        gps_feats = batch['gps_feats'].to(device)
        
        print(f"Input Route Shape: {route_feats.shape}")
        print(f"Input GPS Shape: {gps_feats.shape}")
        
        road_rep, gps_rep, recon_feats = model(route_feats, gps_feats)
        
        print(f"Road Rep Shape: {road_rep.shape}")
        print(f"GPS Rep Shape: {gps_rep.shape}")
        print(f"Recon Feats Shape: {recon_feats.shape}")
        break

if __name__ == "__main__":
    test()
