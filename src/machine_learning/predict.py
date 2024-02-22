from machine_learning.Dataset import BindingResidueDataset
from machine_learning.ModelManager import load_classifier_torch
from setup.configProcessor import get_weight_dir

import os


def run_prediction():
    dataset = BindingResidueDataset()
    predictions = {}

    models = os.listdir(get_weight_dir())
    for model in models:
        model_path = os.path.join(get_weight_dir(), model)
        pretrained_model = load_classifier_torch(model_path)

        print(pretrained_model)
    return None

