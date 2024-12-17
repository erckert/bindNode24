from setup.configProcessor import get_3d_structure_dir, get_DSSP_dir
from data_processing.structure_processing import get_connectivity, get_dssp_features
import pickle


class StructurePostprocessor:
    def __init__(self, predictions, structure_cutoff):
        #with open('predictions.pkl', 'wb') as fh:
        #    pickle.dump(predictions, fh)

        with open('predictions.pkl', 'rb') as fh:
            predictions = pickle.load(fh)
        self.predictions = get_averaged_predictions(predictions)
        self.protein_ids = predictions.keys()
        self.structure_cutoff = structure_cutoff

        dummy_sequences = {}
        for protein_id in self.protein_ids:
            dummy_sequences[protein_id] = "A" * self.predictions[protein_id].shape[0]

        # because get_connectivity uses the protein sequence only to determine the size of the connectivity matrix,
        # we can use dummy sequences of the right length instead of the original sequences.
        self.connectivity_matrices = get_connectivity(structure_cutoff, self.protein_ids, dummy_sequences)
        self.dssp_features = get_dssp_features(self.protein_ids)

    def remove_isolated_binding_residues(self):
        pass


def get_averaged_predictions(predictions):
    averaged_probabilities = {}
    for protein_id in predictions.keys():
        prediction_list = predictions[protein_id]
        averaged_probability = sum(prediction_list)/len(prediction_list)
        averaged_probabilities[protein_id] = averaged_probability

    return averaged_probabilities
