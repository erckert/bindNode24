import os

from setup.configProcessor import get_id_list_path, get_embeddings_path, get_sequence_path, get_label_path, \
    get_3d_structure_dir, use_cache, get_DSSP_dir
from misc.Cache import load_distance_matrix, cache_distance_matrix
from misc.enums import LabelType, DSSPStructure
from Bio.PDB import PDBParser
from scipy.spatial import distance
from sklearn import preprocessing

import numpy as np
import h5py
import fastapy
import torch
import copy
import pandas as pd

from torch_geometric.data import Dataset, Data


class BindingResidueDataset(Dataset):
    def __init__(self, structure_cutoff):
        super().__init__('.', None, None, None)
        self.structure_cutoff = structure_cutoff
        self.protein_ids = self.get_id_list()
        self.embeddings = self.get_embeddings()
        self.sequences = self.get_sequences()
        self.connectivity_matrices = self.get_connectivity()
        self.dssp_features = self.get_dssp_features()

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

    def get_connectivity(self):
        connectivity_matrices = {}
        cutoff = self.structure_cutoff

        # add all protein ids to structure dict, create a matrix that represents the backbone, one for the actual
        # distances and ond for the cutoffs
        for protein_id in self.protein_ids:
            seq_length = len(self.sequences[protein_id])
            distance_matrix = None
            if use_cache():
                distance_matrix = load_distance_matrix(protein_id)
            if distance_matrix is None:
                distance_matrix = self.generate_distance_matrix(protein_id)
            connectivity_matrices[protein_id] = \
                {
                    "backbone": np.eye(seq_length),
                    "distance": distance_matrix,
                    "cutoff": (distance_matrix < cutoff).astype(int)
                }

        return connectivity_matrices

    def generate_distance_matrix(self, protein_id):
        structure_dir = get_3d_structure_dir()
        protein_length = len(self.sequences[protein_id])
        files = os.listdir(structure_dir)

        pdb_parser = PDBParser()
        pdb_file = f"{protein_id}.pdb"

        coordinates = []
        if pdb_file in files:
            structure = pdb_parser.get_structure(protein_id, os.path.join(structure_dir, pdb_file))
            residues = structure.get_residues()
            for residue in residues:
                c_alpha_coordinates = residue["CA"].coord
                # models may have missing residues, so we need to keep track of which position the coordinates belong to
                # index needs to be shifted by one (PDB files are 1-based indexed)
                residue_position = residue.id[1]-1
                coordinates.append((residue_position, c_alpha_coordinates))

        else:
            print(f"No structure available for {protein_id}")
            # if this is the case, the distance matrix will only consist of zeros

        distance_matrix = np.zeros((protein_length, protein_length))
        for i, first_coordinate in coordinates:
            for j, second_coordinate in coordinates:
                pairwise_distance = distance.euclidean(first_coordinate, second_coordinate)
                distance_matrix[i, j] = pairwise_distance
        if use_cache():
            cache_distance_matrix(protein_id, distance_matrix)

        return distance_matrix

    @staticmethod
    def one_hot_encode_dssp_structure(dssp_structure):
        one_hot_encoding = np.zeros(8)
        match dssp_structure:
            case "H":
                position = DSSPStructure.HELIXALPHA.value
            case "B":
                position = DSSPStructure.BETABRIDGE.value
            case "E":
                position = DSSPStructure.EXTENDEDSRTRAND.value
            case "G":
                position = DSSPStructure.HELIX3_10.value
            case "I":
                position = DSSPStructure.HELIXPI.value
            case "P":
                position = DSSPStructure.HELIXK.value
            case "T":
                position = DSSPStructure.TURN.value
            case _:
                position = DSSPStructure.BEND.value
        one_hot_encoding[position] = 1
        return one_hot_encoding

    def get_dssp_features(self):
        dssp_dir = get_DSSP_dir()
        files = os.listdir(dssp_dir)

        min_max_scaler = preprocessing.MinMaxScaler()
        normalized_dssp_features = {}

        for protein_id in self.protein_ids:
            dssp_file = f"{protein_id}.csv"
            if dssp_file in files:
                dssp_features = pd.read_csv(os.path.join(dssp_dir, dssp_file), header=None, sep=";")
                sequence_length = len(dssp_features.index)

                # convert absolute position in sequence to relative position
                relative_index = dssp_features.iloc[:, 0].div(sequence_length)

                # one-hot encode secondary structures
                secondary_structure_one_hot_encoding = \
                    pd.DataFrame([self.one_hot_encode_dssp_structure(entry) for entry in dssp_features.iloc[:, 2]])

                # keep solvant accesibility as it is (already normalized)
                solvent_accessibility = dssp_features.iloc[:, 3]

                # phi and psi angles are degrees => division by 360
                normalized_phi = dssp_features.iloc[:, 4].div(360)
                normalized_psi = dssp_features.iloc[:, 5].div(360)

                # use normalization from sci-kit to normalize relative index and energies per protein
                NH_O_1_relative_index = dssp_features.iloc[:, 6]
                normalized_NH_O_1_relative_index = pd.DataFrame(
                    min_max_scaler.fit_transform(pd.DataFrame(NH_O_1_relative_index))
                )

                NH_O_1_relative_energy = dssp_features.iloc[:, 7]
                normalized_NH_O_1_relative_energy = pd.DataFrame(
                    min_max_scaler.fit_transform(pd.DataFrame(NH_O_1_relative_energy))
                )

                O_NH_1_relative_index = dssp_features.iloc[:, 8]
                normalized_O_NH_1_relative_index = pd.DataFrame(
                    min_max_scaler.fit_transform(pd.DataFrame(O_NH_1_relative_index))
                )

                O_NH_1_relative_energy = dssp_features.iloc[:, 9]
                normalized_O_NH_1_relative_energy = pd.DataFrame(
                    min_max_scaler.fit_transform(pd.DataFrame(O_NH_1_relative_energy))
                )

                NH_O_2_relative_index = dssp_features.iloc[:, 10]
                normalized_NH_O_2_relative_index = pd.DataFrame(
                    min_max_scaler.fit_transform(pd.DataFrame(NH_O_2_relative_index))
                )

                NH_O_2_relative_energy = dssp_features.iloc[:, 11]
                normalized_NH_O_2_relative_energy = pd.DataFrame(
                    min_max_scaler.fit_transform(pd.DataFrame(NH_O_2_relative_energy))
                )

                O_NH_2_relative_index = dssp_features.iloc[:, 12]
                normalized_O_NH_2_relative_index = pd.DataFrame(
                    min_max_scaler.fit_transform(pd.DataFrame(O_NH_2_relative_index))
                )

                O_NH_2_relative_energy = dssp_features.iloc[:, 13]
                normalized_O_NH_2_relative_energy = pd.DataFrame(
                    min_max_scaler.fit_transform(pd.DataFrame(O_NH_2_relative_energy))
                )

                frames = [
                    relative_index,
                    secondary_structure_one_hot_encoding,
                    solvent_accessibility,
                    normalized_phi,
                    normalized_psi,
                    normalized_NH_O_1_relative_index,
                    normalized_NH_O_1_relative_energy,
                    normalized_O_NH_1_relative_index,
                    normalized_O_NH_1_relative_energy,
                    normalized_NH_O_2_relative_index,
                    normalized_NH_O_2_relative_energy,
                    normalized_O_NH_2_relative_index,
                    normalized_O_NH_2_relative_energy
                ]

                normalized_dssp_features[protein_id] = pd.concat(frames, axis=1).to_numpy()

            else:
                print(f"No DSSP features available for {protein_id}")
                # dummy DSSP features, in the form of a matrix consisting only of 0 if there are none available
                normalized_dssp_features[protein_id] = np.zeros((len(self.sequences[protein_id]),20))

        return normalized_dssp_features

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
