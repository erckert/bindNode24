"""Implementation is heavily based on the works of Littmann, et al. with bindEmbed21DL
https://github.com/Rostlab/bindPredict/blob/master/bindEmbed21DL.py"""
from bindNode23.bindNode23 import BindNode23
from assess_performance import PerformanceAssessment
from config import FileSetter, FileManager, GeneralInformation
from data_preparation import ProteinInformation

import torch.nn.functional as F


import sys
from pathlib import Path


def main():
    GeneralInformation.seed_all(42)

    keyword = "optimize-architecture"
    keyword = "best-training"
    # keyword = "testing"
    path = "/home/frank/bind/bindPredict"  # TODO set path to working directory
    Path(path).mkdir(parents=True, exist_ok=True)
    LOGGING = False

    cutoff = 0.5
    distance_cutoff_singleruns = 1  # shell 1 distance cutoff. sufficient for most models except SAGEConvGATMLP
    struct_cutoff = 15  # shell 2 distance cutoff. Ignored for most models
    ri = False  # Should RI or raw probabilities be written?

    sequences = FileManager.read_fasta(FileSetter.fasta_file())

    

    if keyword == "optimize-architecture":
        params = {
            "lr": [0.01],
            "betas": [(0.9, 0.999)],
            "eps": [1e-8],
            "weight_decay": [0.0],
            "features": [128, 192, 256, 320],
            "kernel": [5],  # leftover from BindEmbed21DL. unused.
            "stride": [1],  # leftover from BindEmbed21DL. unused.
            "dropout": [0.8, 0.7, 0.6],
            "epochs": [100],
            "early_stopping": [True],
            "weights": [[8.9, 7.7, 4.4]],
            "cutoff_embd": [10, 14, 18],    # shell 1 distance cutoff. sufficient for most models except SAGEConvGATMLP
            "cutoff_struc": [15, 30, 45],   # shell 2 distance cutoff. Ignored for most models
            "heads": [2, 4],    # leftover from GATv2 experiments. unused.
            "architecture": ["BindNode23SAGEConvMLP"], # ["BindNode23GCN", "BindNode23SAGEConv", "BindNode23SAGEConvMLP"],
            "activation": [F.leaky_relu],
            "batchsize": [406],
            "dropout_fcn": [0.8],
        }
 
        result_file = "{}/cross_validation_results.txt".format(path)
        BindNode23.hyperparameter_optimization_pipeline(params, 5, result_file)

    elif keyword == "best-training":
        params = {
            "lr": 0.01,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "weight_decay": 0.0,
            "features": 128,
            "kernel": 5,    # leftover from BindEmbed21DL. unused.
            "stride": 1,    # leftover from BindEmbed21DL. unused.
            "dropout": 0.7,
            "epochs": 100,
            "early_stopping": True,
            "weights": [8.9, 7.7, 4.4],
            "cutoff_embd": distance_cutoff_singleruns,  # shell 1 distance cutoff. sufficient for most models except SAGEConvGATMLP
            "cutoff_struc": struct_cutoff,  # shell 2 distance cutoff. Ignored for most models
            "heads": 1, # leftover from GATv2 experiments. unused.
            "architecture": "BindNode23SAGEConvMLP",
            "activation": F.leaky_relu,
            "batchsize": 406,
            "dropout_fcn": 0.8,
        }

        model_prefix = "{}/trained_model".format(path)
        prediction_folder = "{}/predictions".format(path)
        Path(prediction_folder).mkdir(parents=True, exist_ok=True)

        proteins = BindNode23.cross_train_pipeline(
            params, model_prefix, prediction_folder, ri
        )

        # assess performance
        labels = ProteinInformation.get_labels(proteins.keys(), sequences)
        model_performances = PerformanceAssessment.combine_protein_performance(
            proteins, cutoff, labels
        )
        PerformanceAssessment.print_performance_results(model_performances, params)

    elif keyword == "testing":
        model_prefix = "{}/trained_model".format(path)
        prediction_folder = "{}/predictions_testset/".format(path)
        Path(prediction_folder).mkdir(parents=True, exist_ok=True)

        ids_in = FileSetter.test_ids_in()
        with open(ids_in, "r") as labels_in:
            ids_in = labels_in.readlines()
            ids_in = [id.rstrip("\n") for id in ids_in]
        fasta_file = FileSetter.fasta_file()
        proteins = BindNode23.prediction_pipeline(
            model_prefix,
            cutoff,
            prediction_folder,
            ids_in,
            fasta_file,
            ri,
            distance_cutoff_embd=distance_cutoff_singleruns,
            distance_cutoff_struc=struct_cutoff
        )

        # assess performance
        labels = ProteinInformation.get_labels(proteins.keys(), sequences)
        model_performances = PerformanceAssessment.combine_protein_performance(
            proteins, cutoff, labels
        )
        PerformanceAssessment.print_performance_results(model_performances, params={})


main()
