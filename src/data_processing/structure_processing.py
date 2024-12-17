import os

import numpy as np
import pandas as pd

from Bio.PDB import PDBParser
from scipy.spatial import distance
from sklearn import preprocessing

from setup.configProcessor import get_3d_structure_dir, use_cache, get_DSSP_dir
from misc.Cache import cache_distance_matrix, load_distance_matrix
from misc.enums import DSSPStructure


def get_connectivity(structure_cutoff, protein_ids, sequences):
    connectivity_matrices = {}
    cutoff = structure_cutoff

    # add all protein ids to structure dict, create a matrix that represents the backbone, one for the actual
    # distances and ond for the cutoffs
    for protein_id in protein_ids:
        seq_length = len(sequences[protein_id])
        distance_matrix = None
        if use_cache():
            distance_matrix = load_distance_matrix(protein_id)
        if distance_matrix is None:
            protein_sequence = sequences[protein_id]
            distance_matrix = generate_distance_matrix(protein_sequence, protein_id)
        connectivity_matrices[protein_id] = \
            {
                "backbone": np.eye(seq_length),
                "distance": distance_matrix,
                "cutoff": (distance_matrix < cutoff).astype(int)
            }

    return connectivity_matrices


def generate_distance_matrix(protein_sequence, protein_id):
    structure_dir = get_3d_structure_dir()
    protein_length = len(protein_sequence)
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
            residue_position = residue.id[1] - 1
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


def get_dssp_features(protein_ids):
    dssp_dir = get_DSSP_dir()
    files = os.listdir(dssp_dir)

    min_max_scaler = preprocessing.MinMaxScaler()
    normalized_dssp_features = {}

    for protein_id in protein_ids:
        dssp_file = f"{protein_id}.csv"
        if dssp_file in files:
            dssp_features = pd.read_csv(os.path.join(dssp_dir, dssp_file), header=None, sep=";")
            sequence_length = len(dssp_features.index)

            # convert absolute position in sequence to relative position
            relative_index = dssp_features.iloc[:, 0].div(sequence_length)

            # one-hot encode secondary structures
            secondary_structure_one_hot_encoding = \
                pd.DataFrame([one_hot_encode_dssp_structure(entry) for entry in dssp_features.iloc[:, 2]])

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
            normalized_dssp_features[protein_id] = np.zeros((len(self.sequences[protein_id]), 20))

    return normalized_dssp_features
