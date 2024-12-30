from setup.configProcessor import get_id_list_path, get_embeddings_path, get_sequence_path, get_label_path, use_cache
from data_processing.structure_processing import get_connectivity, get_dssp_features
from misc.enums import LabelType

import numpy as np
import h5py
import fastapy
import torch
import copy

from torch_geometric.data import Dataset, Data


class BindingResidueDataset(Dataset):
    def __init__(self, structure_cutoff):
        super().__init__('.', None, None, None)
        self.structure_cutoff = structure_cutoff
        self.protein_ids = self.get_id_list()
        self.embeddings = self.get_embeddings()
        self.sequences = self.get_sequences()
        self.connectivity_matrices = get_connectivity(self.structure_cutoff, self.protein_ids, self.sequences)
        self.dssp_features = get_dssp_features(self.protein_ids)

    def len(self):
        return len(self.protein_ids)

    def get(self, item):
        protein_id = self.protein_ids[item]
        protein_graph_edges = np.array(self.get_connectivity_matrix(item)["backbone"].nonzero())
        protein_graph_cutoff_edges = np.array(self.get_connectivity_matrix(item)["cutoff"].nonzero())

        # retrieve distances of edges below structure cutoff
        edge_attributes = \
            self.get_connectivity_matrix(item)["distance"][protein_graph_cutoff_edges[0], protein_graph_cutoff_edges[1]]
        # divide by distance cutoff to avoid div by zero error for linalg.norm
        edge_attributes = (edge_attributes / self.structure_cutoff)
        # invert distance to achieve weights: distance zero has highest weight
        edge_features = (1 - edge_attributes)
        protein_graph = Data(
            x=torch.Tensor(self.embeddings[protein_id]),
            edge_index=torch.LongTensor(protein_graph_edges),
            edge_index_cutoff=torch.LongTensor(protein_graph_cutoff_edges),
            edge_features=torch.Tensor(edge_features),
            dssp_features=torch.tensor(self.dssp_features[protein_id]).float()
        )
        return protein_graph, protein_id

    def get_embedding(self, item):
        protein_id = self.protein_ids[item]
        return self.embeddings[protein_id]

    def get_sequence(self, item):
        protein_id = self.protein_ids[item]
        return self.sequences[protein_id]

    def get_connectivity_matrix(self, item):
        protein_id = self.protein_ids[item]
        return self.connectivity_matrices[protein_id]

    def train_val_split(self, train_ids, val_ids):
        train_subset = self.create_subset(train_ids)
        val_subset = self.create_subset(val_ids)
        return train_subset, val_subset

    def create_subset(self, subset_ids):
        subset = copy.deepcopy(self)
        self.collect_subset_data(subset_ids, subset)
        return subset

    def collect_subset_data(self, subset_ids, subset):
        subset.protein_ids = [protein_id for protein_id in subset_ids]
        subset.embeddings = dict(
            zip(
                subset.protein_ids,
                [self.embeddings[protein_id] for protein_id in subset.protein_ids]
            )
        )
        subset.sequences = dict(
            zip(
                subset.protein_ids,
                [self.sequences[protein_id] for protein_id in subset.protein_ids]
            )
        )
        subset.connectivity_matrices = dict(
            zip(
                subset.protein_ids,
                [self.connectivity_matrices[protein_id] for protein_id in subset.protein_ids]
            )
        )

    @staticmethod
    def get_id_list():
        id_list = []
        with open(get_id_list_path(), 'r') as fh:
            lines = fh.readlines()
            for line in lines:
                protein_id = line.strip()
                if protein_id != "":
                    id_list.append(protein_id)
        return id_list

    def get_embeddings(self):
        embeddings = {}
        embedding_h5 = h5py.File(get_embeddings_path(), 'r')
        for key, value in embedding_h5.items():
            if key in self.protein_ids:
                embeddings[key] = np.array(value[:, :], dtype=np.float32)
        return embeddings

    def get_sequences(self):
        protein_sequences = {}
        for record in fastapy.parse(get_sequence_path()):
            if record.id in self.protein_ids:
                protein_sequences[record.id] = record.seq
        return protein_sequences


class BindingResidueDatasetWithLabels(BindingResidueDataset):
    def __init__(self, structure_cutoff):
        super(BindingResidueDatasetWithLabels, self).__init__(structure_cutoff)
        self.labels = self.get_labels(self.sequences)

    def get(self, item):
        protein_id = self.protein_ids[item]
        protein_graph_edges = np.array(self.get_connectivity_matrix(item)["backbone"].nonzero())
        protein_graph_cutoff_edges = np.array(self.get_connectivity_matrix(item)["cutoff"].nonzero())

        edge_attributes = \
            self.get_connectivity_matrix(item)["distance"][protein_graph_cutoff_edges[0], protein_graph_cutoff_edges[1]]
        # divide by distance cutoff to avoid div by zero error for linalg.norm
        edge_attributes = (edge_attributes / self.structure_cutoff)
        # invert distance to achieve weights: distance zero has highest weight
        edge_features = (1 - edge_attributes)

        protein_graph = Data(
            x=torch.Tensor(self.get_embedding(item)),
            edge_index=torch.LongTensor(protein_graph_edges),
            edge_index_cutoff=torch.LongTensor(protein_graph_cutoff_edges),
            edge_features=torch.Tensor(edge_features),
            y=torch.Tensor(self.labels[protein_id]),
            dssp_features=torch.tensor(self.dssp_features[protein_id]).float()
        )

        return protein_graph, protein_id

    def collect_subset_data(self, subset_ids, subset):
        super().collect_subset_data(subset_ids, subset)
        subset.labels = dict(
            zip(
                subset.protein_ids,
                [self.labels[protein_id] for protein_id in subset.protein_ids]
            )
        )

    def get_labels(self, protein_sequences):
        labels = {}
        metal_labels = self.generate_label_dict(get_label_path(LabelType.METAL))
        small_labels = self.generate_label_dict(get_label_path(LabelType.SMALL))
        nuclear_labels = self.generate_label_dict(get_label_path(LabelType.NUCLEAR))

        for protein_id in protein_sequences.keys():
            length = len(protein_sequences[protein_id])
            label_array = np.zeros((length, 3))
            if protein_id in metal_labels:
                for metal_label in metal_labels[protein_id]:
                    label_array[metal_label, LabelType.METAL.value] = 1
            if protein_id in small_labels:
                for small_label in small_labels[protein_id]:
                    label_array[small_label, LabelType.SMALL.value] = 1
            if protein_id in nuclear_labels:
                for nuclear_label in nuclear_labels[protein_id]:
                    label_array[nuclear_label, LabelType.NUCLEAR.value] = 1
            labels[protein_id] = label_array

        return labels

    @staticmethod
    def generate_label_dict(file_path):
        labels = {}
        with open(file_path, 'r') as fh:
            lines = fh.readlines()
            for line in lines:
                line = line.rstrip()
                parts = line.split('\t')
                if len(parts) == 2:
                    protein_id = parts[0]
                    # label file is 1 indexed so we need to shift all entries by 1
                    binding_residues = [int(entry) - 1 for entry in parts[1].split(',')]
                    labels[protein_id] = binding_residues
        return labels
