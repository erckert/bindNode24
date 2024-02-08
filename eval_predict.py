from bindNode23.bindNode23_eval import BindNode23
from assess_performance_eval import PerformanceAssessment
from config import FileSetter, FileManager, GeneralInformation
from data_preparation import ProteinInformation

import torch.nn.functional as F


import sys
import os
from pathlib import Path

"""In this file, we switch to bindNode23 throughout all steps.
We need it to enable calling an architecture by name and pass all additional params, like heads."""


def main():
    GeneralInformation.seed_all(42)

    keyword = "devset"
    # keyword = "testing"
    path = "/home/frank/bind/bindPredict"  # TODO set path to working directory
    Path(path).mkdir(parents=True, exist_ok=True)
    LOGGING = False

    cutoff = 0.5
    distance_cutoff_singleruns = 18
    struct_cutoff = 45
    ri = False  # Should RI or raw probabilities be written?

    sequences = FileManager.read_fasta(FileSetter.fasta_file())

    if keyword == "devset":
        params = {
            "lr": 0.01,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "weight_decay": 0.0,
            "features": 320,
            "kernel": 5,
            "stride": 1,
            "dropout": 0.7,
            "epochs": 100,
            "early_stopping": True,
            "weights": [8.9, 7.7, 4.4],
            "cutoff_embd": distance_cutoff_singleruns,
            "cutoff_struc": struct_cutoff,
            "heads": 1,
            "architecture": "BindNode23SAGEConvMLP",
            "activation": F.leaky_relu,
            "batchsize": 406,
            "dropout_fcn": 0.8,
        }
        model_name = f"{params['architecture']}_{params['features']}_{params['heads']}"
        model_name = (
            f"BindNode23SAGEConvMLP_NODSSP_{params['features']}_{params['heads']}"
        )
        prediction_folder = os.path.join(
            "/home/frank/bind/bindPredict/bindNode23/final_eval",
            model_name,
            "predictions_devset",
        )
        Path(prediction_folder).mkdir(parents=True, exist_ok=True)

        proteins = BindNode23.eval_pipeline(params, model_name, prediction_folder, ri)

        # assess performance
        labels = ProteinInformation.get_labels(proteins.keys(), sequences)
        model_performances = PerformanceAssessment.combine_protein_performance(
            proteins, cutoff, labels
        )
        PerformanceAssessment.store_performance_results(
            model_performances,
            params,
            os.path.join(
                "/home/frank/bind/bindPredict/bindNode23/final_eval", model_name
            ),
        )

    elif keyword == "testing":
        model_prefix = "{}/trained_model".format(path)
        prediction_folder = os.path.join(
            "/home/frank/bind/bindPredict/bindNode23/final_eval",
            model_name,
            "predictions_test",
        )
        Path(prediction_folder).mkdir(parents=True, exist_ok=True)

        ids_in = FileSetter.test_ids_in()
        with open(ids_in, "r") as labels_in:
            ids_in = labels_in.readlines()
            ids_in = [id.rstrip("\n") for id in ids_in]
        fasta_file = FileSetter.fasta_file()
        proteins = BindNode23.prediction_pipeline(  # TODO: pred pipeline does not work yet. fix it once final model is chosen!
            model_prefix,
            cutoff,
            prediction_folder,
            ids_in,
            fasta_file,
            ri,
            distance_cutoff=distance_cutoff_singleruns,
        )

        # assess performance
        labels = ProteinInformation.get_labels(proteins.keys(), sequences)
        model_performances = PerformanceAssessment.combine_protein_performance(
            proteins, cutoff, labels
        )
        PerformanceAssessment.print_performance_results(
            model_performances, params=params
        )


main()
