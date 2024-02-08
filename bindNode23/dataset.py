# https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_dataset.html

import os.path as osp

import torch
from torch_geometric.data import Dataset, Data
from tqdm import tqdm
import time
from pathlib import Path


import torch
import numpy as np
import sys
from collections import defaultdict
from sklearn.preprocessing import normalize
from config import FileManager, FileSetter
from assess_performance import PerformanceAssessment


class GraphDataset(Dataset):
    def __init__(
        self,
        samples,
        embeddings,
        seqs,
        labels,
        distance_cutoff_embd,
        distance_cutoff_struc,
        structures,
        protein_prediction=False,
        root=".",
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        super().__init__(root, transform, pre_transform, pre_filter)

        self.protein_prediction = protein_prediction
        self.samples = samples
        self.embeddings = embeddings
        self.seqs = seqs
        self.structures = structures
        self.labels = labels
        self.distance_cutoff_embd = distance_cutoff_embd
        self.distance_cutoff_struc = distance_cutoff_struc
        self.eps = 0.000001

    def len(self):
        return len(self.samples)

    def get(self, item):
        prot_id = self.samples[item]
        edges, edges2, attributes = self.get_edge_index_and_attr_tensor(
            self.structures[prot_id]
        )

        graphdata = Data(
            x=torch.Tensor(self.embeddings[prot_id]),
            edge_index=torch.LongTensor(edges),
            edge_attr=torch.Tensor(attributes),
            y=torch.Tensor(self.labels[prot_id]),
            edge_index2=torch.LongTensor(edges2),
        )

        """
        Data necessary for forward call (from: PYG):
        Node feature matrix with shape [num_nodes, num_node_features]. (default: None)
        edge_index (LongTensor, optional) Graph connectivity in COO format with shape [2, num_edges]. (default: None)
        Added: second set of edge_index to allow different shells to be evaluated.
        edge_attr (torch.Tensor, optional) Edge feature matrix with shape [num_edges, num_edge_features]. (default: None)
        y (torch.Tensor, optional) Graph-level or node-level ground-truth labels with arbitrary shape. (default: None)"""

        if self.protein_prediction:
            return graphdata, prot_id
        else:
            return graphdata

    def get_input_dimensions(self):
        return list(self.embeddings.values())[0].shape[1]

    def convert_to_coo(self, arr):
        """Insert a distogram array and return a torch.sparse_coo_tensor"""
        adj = (arr < self.distance_cutoff).astype(int)
        i = np.array(adj.nonzero())
        v = np.ones(i[0].shape[0])
        return torch.sparse_coo_tensor(i, v, (arr.shape[0], arr.shape[1]))

    def convert_to_edge_index_tensor(self, arr):
        """Insert a distogram array and return a torch.Tensor"""
        adj = (arr < self.distance_cutoff).astype(int)
        return np.array(adj.nonzero())

    def edge_connectivity_tensor(self, arr):
        """Insert a distogram array and return a torch.Tensor for connectivity (number of neighbors for each node)"""

        connectivities = np.ndarray((6, arr.shape[0]))
        for k, distance_cutoff in enumerate([5, 7, 9, 14, 18, 30]):
            adj = (arr < distance_cutoff).astype(int)
            edge_indices = np.array(adj.nonzero())
            a = edge_indices[0]
            unique, counts = np.unique(a, return_counts=True)
            connectivities[k] = counts
        return normalize(connectivities, axis=0)

    def get_edge_index_and_attr_tensor(self, arr):
        """Insert a distogram array and return two torch.Tensor: edge_indices and edge_attributes. vectorized version with attr normalization"""
        adj = torch.eye(arr.shape[0]).numpy().astype(int)
        edge_indices = np.array(adj.nonzero())
        adj2 = (arr < self.distance_cutoff_struc).astype(int)
        edge_indices2 = np.array(adj2.nonzero())
        edge_attr = arr[edge_indices2[0], edge_indices2[1]]
        edge_attr = (
            edge_attr / self.distance_cutoff_struc
        )  # divide by distance cutoff to avoid div by zero error for linalg.norm
        edge_attr = (
            1 - edge_attr
        )  # invert distance to achieve weights: distance zero has highest weight
        return edge_indices, edge_indices2, edge_attr
