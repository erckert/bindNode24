from machine_learning.Dataset import BindingResidueDataset
from machine_learning.ModelManager import load_classifier_torch
from setup.configProcessor import get_weight_dir, get_structure_cutoff
from setup.generalSetup import select_device
from output.outputFileWriting import write_predictions_to_file

from torch_geometric.loader import DataLoader
import os
import torch
import numpy as np


def run_prediction():
    structure_cutoff = get_structure_cutoff(only_first_value=True)
    dataset = BindingResidueDataset(structure_cutoff)
    predictions = {}

    sigmoid = torch.nn.Sigmoid()

    models = os.listdir(get_weight_dir())
    for model in models:
        model_path = os.path.join(get_weight_dir(), model)
        pretrained_model = load_classifier_torch(model_path)
        pretrained_model.to(select_device())

        data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

        pretrained_model.eval()
        for protein_graph, protein_id in data_loader:
            with torch.no_grad():
                protein_id = protein_id[0]
                protein_graph = protein_graph.to(select_device())

                prediction = pretrained_model.forward(
                    protein_graph.x,
                    protein_graph.edge_index,
                    protein_graph.edge_index_cutoff,
                    protein_graph.edge_features,
                    protein_graph.dssp_features
                )
                prediction = sigmoid(prediction)
                prediction = prediction.detach().cpu()
                prediction = np.array(prediction)

                if protein_id not in predictions:
                    predictions[protein_id] = [prediction]
                else:
                    predictions[protein_id] += [prediction]
    write_predictions_to_file(predictions)
