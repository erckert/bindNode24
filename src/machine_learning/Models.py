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

    def forward(self, features, edges, edges2, edge_features, additional_feature):
        """Second pair of edge tensor allows different shells of neighbors to be evaluated.
        Not utilized in this model, but data pipeline must work for all models."""
        edge_features = None
        x = self.gcn1(features, edges, edge_features)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        return self.gcn2(x, edges, edge_features)


class SAGEConvModel(torch.nn.Module):
    """
    Two layers of SAGEConv.
    """
    def __init__(self, in_channels=1024, feature_channels=128, out_channels=3, dropout=0.7, activation=F.leaky_relu):
        super(SAGEConvModel, self).__init__()
        self.sage1 = SAGEConv(in_channels=in_channels, out_channels=feature_channels, aggr="mean")
        self.sage2 = SAGEConv(in_channels=feature_channels, out_channels=out_channels, aggr="mean")
        self.dropout = dropout
        self.activation = activation

    def forward(self, features, edges, edges2, edge_features, additional_feature):
        x = self.sage1(features, edges)
        x = self.activation(x)
        x = F.dropout(x, self.dropout, training=self.training)
        return self.sage2(x, edges)


class SAGEConvMLPModel(torch.nn.Module):
    """
    SAGEConv followed by MLP. This model is numerically strongest.
    """

    def __init__(self, in_channels=1024, feature_channels=128, additional_channels=20, out_channels=3, dropout=0.7,
                 activation=F.leaky_relu, heads=4, dropout_fcn=0.5):
        super(SAGEConvMLPModel, self).__init__()
        intermediate_layer_size = feature_channels + additional_channels

        self.sage1 = SAGEConv(in_channels=in_channels, out_channels=feature_channels, aggr="mean")
        # Why // head? What is heads?
        self.fc1 = torch.nn.Linear(intermediate_layer_size, intermediate_layer_size // heads)
        self.fc2 = torch.nn.Linear(intermediate_layer_size // heads, out_channels)

        self.norm = torch.nn.BatchNorm1d(intermediate_layer_size // heads)
        self.dropout = dropout
        self.dropout_fcn = dropout_fcn
        self.activation = activation

    def forward(self, features, edges, edges2, edge_features, additional_feature):
        x = self.sage1(features, edges)
        x = self.activation(x)
        x = F.dropout(x, self.dropout, training=self.training)

        # numerically, it performed better to feed in DSSP feats here
        x = self.fc1(torch.concat([x, additional_feature], dim=1))

        x = F.relu(x)
        x = self.norm(x)
        x = F.dropout(x, self.dropout_fcn, training=self.training)

        return self.fc2(x)


class SAGEConvGATMLPModel(torch.nn.Module):
    """
    Mix of SAGEConv, GAT and MLP model with different neighbor shells evaluated.
    """
    def __init__(self, in_channels=1024, feature_channels=128, additional_channels=20, out_channels=3, dropout=0.7,
                 activation=F.leaky_relu, heads=4, dropout_fcn=0.5):
        super(SAGEConvGATMLPModel, self).__init__()
        self.sage1 = SAGEConv(in_channels=in_channels, out_channels=feature_channels, aggr="mean")
        self.gat1 = GATv2Conv(in_channels=additional_channels, out_channels=int(additional_channels / 2), heads=heads,
                              edge_dim=1, concat=True)
        self.gat2 = GATv2Conv(in_channels=heads * int(additional_channels / 2),
                              out_channels=int(additional_channels / 4), heads=heads, edge_dim=1, concat=True)
        self.fc1 = torch.nn.Linear(
            feature_channels + additional_channels + int(additional_channels / 4) * heads,
            feature_channels + additional_channels + int(additional_channels / 4) * heads
        )
        self.fc2 = torch.nn.Linear(
            feature_channels + additional_channels + int(additional_channels / 4) * heads,
            out_channels
        )

        self.normx = torch.nn.BatchNorm1d(feature_channels)
        self.normy = torch.nn.BatchNorm1d(int(additional_channels / 4) * heads)
        self.normfcn = torch.nn.BatchNorm1d(
            feature_channels + additional_channels + int(additional_channels / 4) * heads
        )
        self.dropout = dropout
        self.dropout_fcn = dropout_fcn
        self.activation = activation

    def forward(self, features, edges, edges2, edge_features, additional_features):
        x = self.sage1(features, edges)
        x = self.normx(x)

        y = self.gat1(additional_features, edges2, edge_features)
        y = self.activation(y)
        y = F.dropout(y, self.dropout, training=self.training)
        y = self.gat2(y, edges2, edge_features)
        y = self.normy(y)

        z = torch.concat([x, y], dim=1)
        z = self.activation(z)
        z = F.dropout(z, self.dropout, training=self.training)
        z = self.fc1(torch.concat([z, additional_features], dim=1))
        z = self.normfcn(z)
        z = F.relu(z)
        z = F.dropout(z, self.dropout_fcn, training=self.training)

        return self.fc2(z)
