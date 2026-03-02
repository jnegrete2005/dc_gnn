import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.data import HeteroData

FEATURE_DIM = 512


class GNN(nn.Module):
    def __init__(self, hidden_channels, out_channels, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.preamble = nn.Sequential(
            nn.Linear(512, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
        )

        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

        self.postamble = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.preamble(x)

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = self.postamble(x)
        return x


class EdgeDecoder(nn.Module):
    def forward(self, x_dc: torch.Tensor, x_ds: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        edge_feat_dc = x_dc[edge_index[0]]
        edge_feat_ds = x_ds[edge_index[1]]

        return (edge_feat_dc * edge_feat_ds).sum(dim=-1)


class Model(nn.Module):
    def __init__(self, hidden_channels, out_channels, data: HeteroData, dropout=0.2):
        super().__init__()

        self.encoder = GNN(hidden_channels, out_channels, dropout=dropout)
        self.encoder = to_hetero(self.encoder, metadata=data.metadata(), aggr='sum')

        self.decoder = EdgeDecoder()

    def forward(self, data: HeteroData):
        x_dict = data.x_dict
        x_dict = self.encoder(x_dict, data.edge_index_dict)

        pred = self.decoder(
            x_dict["dc"],
            x_dict["disease"],
            data["dc", "treats", "disease"].edge_label_index
        )

        return pred
