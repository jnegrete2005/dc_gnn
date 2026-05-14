import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch_geometric.data import HeteroData

FEATURE_DIM = 512


class GNN(nn.Module):
    def __init__(self, hidden_channels, out_channels, metadata, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.preamble = nn.Sequential(
            nn.Linear(512, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.gnn_layers = pyg_nn.Sequential(
            "x, edge_index",
            [
                (pyg_nn.SAGEConv(hidden_channels, hidden_channels), "x, edge_index -> x"),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                (pyg_nn.SAGEConv(hidden_channels, out_channels), "x, edge_index -> x"),
            ],
        )

        # Convert to hetero
        self.preamble = pyg_nn.to_hetero(self.preamble, metadata)
        self.gnn_layers = pyg_nn.to_hetero(self.gnn_layers, metadata, aggr="sum")

    def forward(self, x_dict: dict, edge_index_dict: dict) -> dict:
        x_dict = self.preamble(x_dict)
        x_dict = self.gnn_layers(x_dict, edge_index_dict)
        return x_dict


class MLPEdgeDecoder(nn.Module):
    def __init__(self, hidden_channels, dropout=0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, x_dc: torch.Tensor, x_ds: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        edge_feat_dc = x_dc[edge_index[0]]
        edge_feat_ds = x_ds[edge_index[1]]

        edge_feat = torch.cat([edge_feat_dc, edge_feat_ds], dim=-1)
        return self.mlp(edge_feat).squeeze(-1)


class EdgeDecoder(nn.Module):
    def forward(self, x_dc: torch.Tensor, x_ds: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        edge_feat_dc = x_dc[edge_index[0]]
        edge_feat_ds = x_ds[edge_index[1]]

        return (edge_feat_dc * edge_feat_ds).sum(dim=-1)


class Model(nn.Module):
    def __init__(self, hidden_channels, out_channels, data: HeteroData, dropout=0.2):
        super().__init__()

        self.encoder = GNN(hidden_channels, out_channels, data.metadata(), dropout=dropout)

        self.decoder = MLPEdgeDecoder(out_channels, dropout=dropout)

    def forward(self, data: HeteroData):
        x_dict = data.x_dict
        x_dict = self.encoder(x_dict, data.edge_index_dict)

        pred = self.decoder(x_dict["dc"], x_dict["disease"], data["dc", "treats", "disease"].edge_label_index)

        return pred
