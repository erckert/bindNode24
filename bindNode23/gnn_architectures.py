import torch
from torch_geometric.nn import GCNConv, GATv2Conv, SAGEConv
import pandas as pd
from torch.nn import Linear, ReLU
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.profile import count_parameters
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)


def print_model_summary(model, feat_input=(70000, 1044), edge_input=(2, 2000000)):
    print(f"Number of trainable params: {count_parameters(model)}")


class BindNode23GCN(torch.nn.Module):
    """
    Two layers of GCNConv. Matches SOTA performance with 20% parameters.
    """

    def __init__(
        self, in_channels, feature_channels, out_channels, dropout, heads=4, **kwargs
    ):
        super(BindNode23GCN, self).__init__()
        self.gcn1 = GCNConv(in_channels=in_channels, out_channels=feature_channels)
        self.gcn2 = GCNConv(in_channels=feature_channels, out_channels=out_channels)
        self.dropout = dropout
        print_model_summary(self)

    def forward(self, features, edges, edges2, edge_features):
        """Second pair of edge tensor allows different shells of neighbors to be evaluated.
        Not utilized in this model, but data pipeline must work for all models."""
        edge_features = None
        x = self.gcn1(features, edges, edge_features)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        return self.gcn2(x, edges, edge_features)


class BindNode23SAGEConv(torch.nn.Module):
    """
    bindNode23: Multi-layer SAGEConv model.
    """

    def __init__(
        self,
        in_channels,
        feature_channels,
        out_channels,
        dropout,
        activation,
        heads=4,
        **kwargs,
    ):
        super(BindNode23SAGEConv, self).__init__()
        self.sage1 = SAGEConv(
            in_channels=in_channels, out_channels=feature_channels, aggr="mean"
        )
        self.sage2 = SAGEConv(
            in_channels=feature_channels, out_channels=out_channels, aggr="mean"
        )
        self.dropout = dropout
        self.activation = activation
        print_model_summary(self)
        print(f"In-channels: {in_channels}")

    def forward(self, features, edges, edges2, edge_features):
        """Second pair of edge tensor allows different shells of neighbors to be evaluated.
        Not utilized in this model, but data pipeline must work for all models."""

        x = self.sage1(features, edges)
        x = self.activation(x)
        x = F.dropout(x, self.dropout, training=self.training)
        return self.sage2(x, edges)



class BindNode23SAGEConvMLP(torch.nn.Module):
    """
    SAGEConv followed by MLP. Numerically strongest model.
    """

    def __init__(
        self,
        in_channels,
        feature_channels,
        out_channels,
        dropout,
        activation,
        heads=4,
        dropout_fcn=0.5,
        **kwargs,
    ):
        super(BindNode23SAGEConvMLP, self).__init__()
        self.sage1 = SAGEConv(
            in_channels=1024, out_channels=feature_channels, aggr="mean"
        )
        self.fc1 = nn.Linear(feature_channels + 20, (feature_channels + 20) // heads)
        self.fc2 = nn.Linear((feature_channels + 20) // heads, 3)
        self.norm = nn.BatchNorm1d((feature_channels + 20) // heads)
        self.dropout = dropout
        self.dropout_fcn = dropout_fcn
        self.activation = activation
        print_model_summary(self)

    def forward(self, features, edges, edges2, edge_features):
        """Second pair of edge tensor allows different shells of neighbors to be evaluated.
        Not utilized in this model, but data pipeline must work for all models."""

        x = self.sage1(features[:, :1024], edges)
        x = self.activation(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fc1(torch.concat([x, features[:, 1024:]], dim=1))  #numerically, it performed better to feed in DSSP feats here
        x = F.relu(x)
        x = self.norm(x)
        x = F.dropout(x, self.dropout_fcn, training=self.training)

        return self.fc2(x)


class BindNode23SAGEConvGATMLP(torch.nn.Module):
    """
    Mix of SAGEConv, GAT and MLP model with different neighbor shells evaluated.
    """

    def __init__(
        self,
        in_channels,
        feature_channels,
        out_channels,
        dropout,
        activation,
        heads=4,
        dropout_fcn=0.5,
        **kwargs,
    ):
        super(BindNode23SAGEConvGATMLP, self).__init__()
        self.sage1 = SAGEConv(
            in_channels=1024, out_channels=feature_channels, aggr="mean"
        )
        self.gat1 = GATv2Conv(
            in_channels=20, out_channels=10, heads=heads, edge_dim=1, concat=True
        )
        self.gat2 = GATv2Conv(
            in_channels=heads * 10, out_channels=5, heads=heads, edge_dim=1, concat=True
        )
        self.fc1 = nn.Linear(
            feature_channels + 20 + 5 * heads, (feature_channels + 20 + 5 * heads)
        )
        self.fc2 = nn.Linear((feature_channels + 20 + 5 * heads), 3)
        self.normx = nn.BatchNorm1d((feature_channels))
        self.normy = nn.BatchNorm1d((5 * heads))
        self.normfcn = nn.BatchNorm1d((feature_channels + 20 + 5 * heads))
        self.dropout = dropout
        self.dropout_fcn = dropout_fcn
        self.activation = activation
        print_model_summary(self)

    def forward(self, features, edges, edges2, edge_features):
        """Second pair of edge tensor allows different shells of neighbors to be evaluated."""

        x = self.sage1(features[:, :1024], edges)
        y = self.gat1(features[:, 1024:], edges2, edge_features)
        y = self.activation(y)
        y = F.dropout(y, self.dropout, training=self.training)
        y = self.gat2(y, edges2, edge_features)
        x = self.normx(x)
        y = self.normy(y)
        x = torch.concat([x, y], dim=1)
        x = self.activation(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fc1(torch.concat([x, features[:, 1024:]], dim=1))
        x = self.normfcn(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout_fcn, training=self.training)
        return self.fc2(x)


class ZeroRateClassifier(torch.nn.Module):
    """
    Predicts majority class non-binding for each residue. Used as baseline for comparison.
    """

    def __init__(
        self,
        in_channels,
        feature_channels,
        out_channels,
        dropout,
        activation,
        heads=4,
        dropout_fcn=0.5,
        **kwargs,
    ):
        super(ZeroRateClassifier, self).__init__()
        print("I am ZeroRateClassifier.")
        self.fc2 = nn.Linear((2 * feature_channels + 20) // heads, 3)

    def forward(self, features, edges, edges2, edge_features):
        zero = torch.zeros(
            features.shape[0], 3, requires_grad=True, device=device, dtype=float
        )
        one = torch.ones(
            features.shape[0], 3, requires_grad=True, device=device, dtype=float
        )
        return zero - one


class RandomRateClassifier(torch.nn.Module):
    """
    Predicts by sampling from class distribution. Used as baseline for comparison.
    """

    def __init__(
        self,
        in_channels,
        feature_channels,
        out_channels,
        dropout,
        activation,
        heads=4,
        dropout_fcn=0.5,
        **kwargs,
    ):
        super(RandomRateClassifier, self).__init__()
        print("I am RandomRateClassifier.")
        self.fc2 = nn.Linear((2 * feature_channels + 20) // heads, 3)
        metals = [0] * 2370
        nucleics = [1] * 2689
        smalls = [2] * 9236
        total_num = 170181 - 2370 - 2689 - 9236
        nb = [3] * total_num
        nb.extend(metals)
        nb.extend(nucleics)
        nb.extend(smalls)
        self.samplingpool = np.array(nb)

    def forward(self, features, edges, edges2, edge_features):
        """Prediction is independent of inputs to fwd."""
        random_draws = np.random.choice(self.samplingpool, size=features.shape[0])
        zero = np.zeros((features.shape[0], 3))
        zero[:, 0] = np.where(random_draws == 0, 1, 0)
        zero[:, 1] = np.where(random_draws == 1, 1, 0)
        zero[:, 2] = np.where(random_draws == 2, 1, 0)
        zero = torch.tensor(zero, requires_grad=True, device=device, dtype=float)
        one = torch.ones(
            features.shape[0], 3, requires_grad=True, device=device, dtype=float
        )
        return zero - one
