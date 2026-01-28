import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

class DiffusionBlock(MessagePassing):
    """
    Inductive Graph Diffusion Block.
    Computes x = alpha * x + (1-alpha) * S * x
    where S is the transition matrix derived from edge_index.
    """
    def __init__(self, in_channels, out_channels, K=2, alpha=0.5):
        super(DiffusionBlock, self).__init__(aggr='add')
        self.K = K
        self.alpha = alpha
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x: [num_nodes, in_channels]
        # edge_index: [2, num_edges]
        
        # Calculate degree for normalization
        row, col = edge_index
        deg = torch.zeros(x.size(0), device=x.device)
        deg.scatter_add_(0, row, torch.ones_like(row, dtype=torch.float32))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        res = x
        for _ in range(self.K):
            res = self.propagate(edge_index, x=res, norm=norm)
            res = self.alpha * x + (1 - self.alpha) * res
            
        return self.lin(res)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

class RoadEncoder(nn.Module):
    def __init__(self, road_feat_dim, embed_size):
        super(RoadEncoder, self).__init__()
        self.fc = nn.Linear(road_feat_dim, embed_size)

    def forward(self, x):
        return F.relu(self.fc(x))

class GPSEncoder(nn.Module):
    def __init__(self, gps_feat_dim, embed_size, hidden_size):
        super(GPSEncoder, self).__init__()
        self.gps_linear = nn.Linear(gps_feat_dim, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True, bidirectional=True)

    def forward(self, x):
        # x: [batch, seq_len, gps_feat_dim]
        x = F.relu(self.gps_linear(x))
        output, _ = self.gru(x) # [batch, seq_len, 2*hidden_size]
        return output

class AttributeDecoder(nn.Module):
    def __init__(self, hidden_size, road_feat_dim):
        super(AttributeDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, road_feat_dim)
        )

    def forward(self, x):
        return self.decoder(x)

class JGRMInductiveModel(nn.Module):
    def __init__(self, road_feat_dim, gps_feat_dim, embed_size, hidden_size):
        super(JGRMInductiveModel, self).__init__()
        
        self.road_encoder = RoadEncoder(road_feat_dim, embed_size)
        self.gps_encoder = GPSEncoder(gps_feat_dim, embed_size, hidden_size)
        
        # Graph Diffusion instead of GAT
        self.diffusion = DiffusionBlock(embed_size, embed_size)
        
        # Shared Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=4, dim_feedforward=hidden_size, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Proj heads for matching
        self.road_proj = nn.Linear(embed_size, hidden_size)
        self.gps_proj = nn.Linear(2*hidden_size, hidden_size)
        
        # Reconstruction Head
        self.attr_decoder = AttributeDecoder(hidden_size, road_feat_dim)

    def forward(self, route_feats, gps_feats, edge_index=None):
        """
        route_feats: [batch, route_len, road_feat_dim]
        gps_feats: [batch, gps_len, gps_feat_dim]
        edge_index: [2, num_edges] (Global or local edge index)
        """
        batch_size, route_len, _ = route_feats.shape
        device = route_feats.device
        
        # 1. Road Encoding
        road_emb = self.road_encoder(route_feats) # [B, R, E]
        
        # 2. Graph Diffusion
        # If no edge_index, we assume sequential connection (Path Graph)
        if edge_index is None:
            # Build local edge_index for each sequence in batch
            # Actually, to be efficient, we can process them as a batch if possible
            # But let's simplify: treat each route as a line graph
            # Construct a batch-wise edge_index
            edges = []
            for b in range(batch_size):
                for i in range(route_len - 1):
                    # Each node in batch is b * route_len + i
                    u = b * route_len + i
                    v = b * route_len + i + 1
                    edges.append([u, v])
                    edges.append([v, u]) # undirected for diffusion
            
            if edges:
                edge_index = torch.tensor(edges, dtype=torch.long, device=device).t()
                road_emb_flat = road_emb.view(-1, road_emb.shape[-1])
                road_emb_diffused = self.diffusion(road_emb_flat, edge_index)
                road_emb = road_emb_diffused.view(batch_size, route_len, -1)
        else:
            # Use provided edge_index (needs mapping if IDs are not 0..N)
            pass

        # 3. GPS Encoding
        gps_emb = self.gps_encoder(gps_feats) # [B, G, 2*H]
        
        # 4. Projection
        road_rep = self.road_proj(road_emb) # [B, R, H]
        gps_traj_rep = self.gps_proj(gps_emb) # [B, G, H]
        
        # 5. Attribute Reconstruction
        recon_road_feats = self.attr_decoder(road_rep)
        
        return road_rep, gps_traj_rep, recon_road_feats
