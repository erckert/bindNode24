from data_processing.structure_processing import get_connectivity, get_dssp_features
from setup.configProcessor import get_cutoff, get_rASA_cutoff, get_site_cutoff

import pickle
import copy

import numpy as np


class StructurePostprocessor:
    def __init__(self, predictions, structure_cutoff):
        #with open('predictions.pkl', 'wb') as fh:
        #    pickle.dump(predictions, fh)

        with open('predictions.pkl', 'rb') as fh:
            predictions = pickle.load(fh)
        self.predictions = get_averaged_predictions(predictions)
        self.post_processed_predictions = copy.deepcopy(self.predictions)
        self.protein_ids = predictions.keys()
        self.structure_cutoff = structure_cutoff
        self.rASA_cutoff = get_rASA_cutoff()
        self.binding_site_cutoff = get_site_cutoff()

        dummy_sequences = {}
        for protein_id in self.protein_ids:
            dummy_sequences[protein_id] = "A" * self.predictions[protein_id].shape[0]

        # because get_connectivity uses the protein sequence only to determine the size of the connectivity matrix,
        # we can use dummy sequences of the right length instead of the original sequences.
        self.connectivity_matrices = get_connectivity(structure_cutoff, self.protein_ids, dummy_sequences)
        self.dssp_features = get_dssp_features(self.protein_ids)

    def post_process_predictions(self):
        self.remove_buried_binding_residues()
        self.remove_isolated_binding_residues()

        return self.post_processed_predictions

    def remove_isolated_binding_residues(self):
        for protein_id in self.post_processed_predictions:
            protein_distance = self.connectivity_matrices[protein_id]['distance']
            protein_connectivity = (protein_distance < self.binding_site_cutoff).astype(int)
            residue_information = zip(self.post_processed_predictions[protein_id], protein_connectivity)
            for i, residue_info in enumerate(residue_information):
                prediction, residue_connections = residue_info

                if np.any(prediction[prediction > get_cutoff()]):
                    indexes = np.where(residue_connections == 1)[0]
                    # Remove index of current residue i
                    index_self = np.where(indexes == i)[0]
                    indexes = np.delete(indexes, index_self)

                    # get predictions of connected_residues
                    connected_predictions = self.predictions[protein_id][indexes]
                    # check if there are any other residues predicted to be part of the binding site nearby
                    any_other_predictions_near = np.any(connected_predictions[connected_predictions > get_cutoff()])
                    # remove prediction if there are no others nearby
                    if not any_other_predictions_near:
                        prediction[prediction > get_cutoff()] = 0
                        self.post_processed_predictions[protein_id][i] = prediction

    def remove_buried_binding_residues(self):
        for protein_id in self.post_processed_predictions:
            protein_dssp_features = self.dssp_features[protein_id]
            protein_solvent_accessibility = protein_dssp_features[:, 9]
            residue_information = zip(self.post_processed_predictions[protein_id], protein_solvent_accessibility)
            for i, residue_info in enumerate(residue_information):
                prediction, rasa = residue_info
                if rasa < self.rASA_cutoff:
                    prediction[prediction > get_cutoff()] = 0
                    self.post_processed_predictions[protein_id][i] = prediction


def get_averaged_predictions(predictions):
    averaged_probabilities = {}
    for protein_id in predictions.keys():
        prediction_list = predictions[protein_id]
        averaged_probability = sum(prediction_list)/len(prediction_list)
        averaged_probabilities[protein_id] = averaged_probability

    return averaged_probabilities
