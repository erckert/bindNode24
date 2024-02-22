import torch.nn.functional as F
import torch
from torch_geometric.nn import GCNConv, GATv2Conv, SAGEConv


class GCNConvModel(torch.nn.Module):
    """
    Two layers of GCNConv. Matches SOTA performance with 20% parameters.
    """
    def __init__(self, in_channels=1024, feature_channels=128, out_channels=3, dropout=0.7):
        super(GCNConvModel, self).__init__()
        self.gcn1 = GCNConv(in_channels=in_channels, out_channels=feature_channels)
        self.gcn2 = GCNConv(in_channels=feature_channels, out_channels=out_channels)
        self.dropout = dropout

    def forward(self, features, edges, edges2, edge_features):
        """Second pair of edge tensor allows different shells of neighbors to be evaluated.
        Not utilized in this model, but data pipeline must work for all models."""
        edge_features = None
        x = self.gcn1(features, edges, edge_features)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        return self.gcn2(x, edges, edge_features)


class SAGEConvModel:
    def __init__(self):
        print("implement me!")


class SAGEConvMLPModel:
    def __init__(self):
        print("implement me!")


class SAGEConvGATMLPModel:
    def __init__(self):
        print("implement me!")
