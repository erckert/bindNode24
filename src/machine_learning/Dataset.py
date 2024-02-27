from setup.configProcessor import get_id_list_path, get_embeddings_path, get_sequence_path, get_label_path
from misc.enums import LabelType

import numpy as np
import h5py
import fastapy
import torch
import copy

from torch_geometric.data import Dataset, Data


class BindingResidueDataset(Dataset):
    def __init__(self):
        super().__init__('.', None, None, None)
        self.protein_ids = self.get_id_list()
        self.embeddings = self.get_embeddings(self.protein_ids)
        self.sequences = self.get_sequences(self.protein_ids)
        self.connectivity_matrices = self.get_connectivity(self.protein_ids, self.sequences)

    def len(self):
        return len(self.protein_ids)

    def get(self, item):
        protein_id = self.protein_ids[item]
        protein_graph_edges = np.array(self.get_connectivity_matrix(item).nonzero())
        protein_graph = Data(
            x=torch.Tensor(self.embeddings[protein_id]),
            edge_index=torch.LongTensor(protein_graph_edges),
        )
        return protein_graph

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

    @staticmethod
    def get_embeddings(protein_ids):
        embeddings = {}
        embedding_h5 = h5py.File(get_embeddings_path(), 'r')
        for key, value in embedding_h5.items():
            if key in protein_ids:
                embeddings[key] = np.array(value[:, :], dtype=np.float32)
        return embeddings

    @staticmethod
    def get_connectivity(protein_ids, protein_sequences):
        connectivity_matrices = {}

        # Dummy method to add all protein ids to structure dict and create a matrix that represents the backbone
        for protein_id in protein_ids:
            seq_length = len(protein_sequences[protein_id])
            connectivity_matrices[protein_id] = np.eye(seq_length) + np.eye(seq_length, k=1) + np.eye(seq_length, k=-1)

        # TODO: Fix me!
        return connectivity_matrices

    @staticmethod
    def get_sequences(protein_ids):
        protein_sequences = {}
        for record in fastapy.parse(get_sequence_path()):
            if record.id in protein_ids:
                protein_sequences[record.id] = record.seq
        return protein_sequences


class BindingResidueDatasetWithLabels(BindingResidueDataset):
    def __init__(self):
        super(BindingResidueDatasetWithLabels, self).__init__()
        self.labels = self.get_labels(self.sequences)

    def get(self, item):
        protein_id = self.protein_ids[item]
        protein_graph_edges = np.array(self.get_connectivity_matrix(item).nonzero())
        protein_graph = Data(
            x=torch.Tensor(self.get_embedding(item)),
            edge_index=torch.LongTensor(protein_graph_edges),
            y=torch.Tensor(self.labels[protein_id])
        )
        return protein_graph

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
                for metal_label in metal_labels:
                    label_array[metal_label, 0] = 1
            if protein_id in small_labels:
                for small_label in small_labels:
                    label_array[small_label, 1] = 1
            if protein_id in nuclear_labels:
                for nuclear_label in nuclear_labels:
                    label_array[nuclear_label, 2] = 1
            labels[protein_id] = label_array

        return labels

    @staticmethod
    def generate_label_dict(file_path):
        labels = {}
        with open(file_path, 'r') as fh:
            lines = fh.readlines()
            for line in lines:
                parts = line.split(' ')
                if len(parts) == 2:
                    protein_id = parts[0]
                    # label file is 1 indexed so we need to shift all entries by 1
                    binding_residues = [int(entry)-1 for entry in parts[1].split(',')]
                    labels[protein_id] = binding_residues
        return labels
