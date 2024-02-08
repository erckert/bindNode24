from data_preparation import ProteinInformation
from ml_trainer import MLTrainer
from ml_predictor import MLPredictor
from config import FileSetter, FileManager
import torch
import os

import numpy as np
from sklearn.model_selection import PredefinedSplit
from bindNode23.logging_utils import init_log_run_to_wandb, finish_run_in_wandb, LOGGING


class BindNode23(object):
    @staticmethod
    def cross_train_pipeline(params, model_output, predictions_output, ri):
        """
        Run cross-training pipeline for a specific set of parameters
        :param params:
        :param model_output: If None, trained model is not written
        :param predictions_output: If None, predictions are not written
        :param ri: Should RI or raw probabilities be written?
        :return:
        """

        print("Prepare data")
        ids = []
        fold_array = []
        for s in range(1, 6):
            ids_in = "{}{}.txt".format(FileSetter.split_ids_in(), s)
            split_ids = FileManager.read_ids(ids_in)

            ids += split_ids
            fold_array += [s] * len(split_ids)

        ps = PredefinedSplit(fold_array)

        # get sequences + maximum length + labels
        sequences, max_length, labels = ProteinInformation.get_data(ids)
        embeddings, structures = FileManager.read_embeddings_and_structures(
            FileSetter.embeddings_and_structures_input()
        )

        proteins = dict()
        trainer = MLTrainer(
            pos_weights=params["weights"], batch_size=params["batchsize"]
        )

        for train_index, test_index in ps.split():
            split_counter = fold_array[test_index[0]]
            train_ids = [ids[train_idx] for train_idx in train_index]
            validation_ids = [ids[test_idx] for test_idx in test_index]

            print("Train model")
            if LOGGING:
                init_log_run_to_wandb(params=params)
            model_split = trainer.train_validate(
                params,
                train_ids,
                validation_ids,
                sequences,
                embeddings,
                labels,
                max_length,
                structures,
                verbose=True,
            )
            if LOGGING:
                finish_run_in_wandb()

            if model_output is not None:
                model_path = "{}{}.pt".format(model_output, split_counter)
                FileManager.save_classifier_torch(model_split, model_path)

            print("Calculate predictions per protein")
            ml_predictor = MLPredictor(model_split)
            curr_proteins = ml_predictor.predict_per_protein(
                validation_ids,
                sequences,
                embeddings,
                labels,
                max_length,
                structures=structures,
                distance_cutoff_embd=params["cutoff_embd"],
                distance_cutoff_struc=params["cutoff_struc"],
            )

            proteins = {**proteins, **curr_proteins}

            if predictions_output is not None:
                pass
                # FileManager.write_predictions(proteins, predictions_output, 0.5, ri)

        return proteins

    @staticmethod
    def eval_pipeline(params, model_output, predictions_output, ri):
        """
        Run prediction pipeline for evaluation purposes
        :param params:
        :param model_output: If None, trained model is not written
        :param predictions_output: If None, predictions are not written
        :param ri: Should RI or raw probabilities be written?
        :return:
        """
        print("Prepare data")
        ids = []
        fold_array = []
        for s in range(1, 6):
            ids_in = "{}{}.txt".format(FileSetter.split_ids_in(), s)
            split_ids = FileManager.read_ids(ids_in)

            ids += split_ids
            fold_array += [s] * len(split_ids)

        ps = PredefinedSplit(fold_array)

        # get sequences + maximum length + labels
        sequences, max_length, labels = ProteinInformation.get_data(ids)
        embeddings, structures = FileManager.read_embeddings_and_structures(
            FileSetter.embeddings_and_structures_input()
        )

        proteins = dict()

        for train_index, test_index in ps.split():
            split_counter = fold_array[test_index[0]]
            print("split is ", split_counter)

            train_ids = [ids[train_idx] for train_idx in train_index]
            validation_ids = [ids[test_idx] for test_idx in test_index]

            model_split = torch.load(
                f"/home/frank/bind/bindPredict/trained_model{split_counter}.pt"
            )
            torch.cuda.empty_cache()

            print("Calculate predictions per protein")
            ml_predictor = MLPredictor(model_split)
            curr_proteins = ml_predictor.predict_per_protein(
                validation_ids,
                sequences,
                embeddings,
                labels,
                max_length,
                structures=structures,
                distance_cutoff_embd=params["cutoff_embd"],
                distance_cutoff_struc=params["cutoff_struc"],
            )
            torch.save(
                model_split,
                os.path.join(
                    "/home/frank/bind/bindPredict/bindNode23/final_eval",
                    model_output,
                    f"{model_output}_{split_counter}.pt",
                ),
            )

            proteins = {**proteins, **curr_proteins}

            if predictions_output is not None:
                FileManager.write_predictions(proteins, predictions_output, 0.5, ri)

        return proteins

    @staticmethod
    def hyperparameter_optimization_pipeline(params, num_splits, result_file):
        """
        Development pipeline used to optimize hyperparameters
        :param params:
        :param num_splits:
        :param result_file:
        :return:
        """

        print("Prepare data")
        ids = []
        fold_array = []
        for s in range(1, num_splits + 1):
            ids_in = "{}{}.txt".format(FileSetter.split_ids_in(), s)
            split_ids = FileManager.read_ids(ids_in)

            ids += split_ids
            fold_array += [s] * len(split_ids)

        ids = np.array(ids)

        # get sequences + maximum length + labels
        sequences, max_length, labels = ProteinInformation.get_data(ids)
        embeddings, structures = FileManager.read_embeddings_and_structures(
            FileSetter.embeddings_and_structures_input()
        )

        print("Perform hyperparameter optimization")
        trainer = MLTrainer(
            pos_weights=params["weights"], batch_size=params["batchsize"]
        )
        del params[
            "weights"
        ]  # remove weights to not consider as parameter for optimization

        model = trainer.cross_validate(
            params,
            ids,
            fold_array,
            sequences,
            embeddings,
            labels,
            max_length,
            structures,
            result_file,
        )

        return model

    @staticmethod
    def prediction_pipeline(
        model_prefix,
        cutoff,
        result_folder,
        ids,
        fasta_file,
        ri,
        distance_cutoff_embd,
        distance_cutoff_struc,
    ):
        """
        Run predictions with bindEmbed21DL for a given list of proteins
        :param model_prefix:
        :param cutoff: Cutoff to use to define prediction as binding (default: 0.5)
        :param result_folder:
        :param ids:
        :param fasta_file:
        :param ri: Should RI or raw probabilities be written?
        :return:
        """

        print("Prepare data")
        sequences, max_length, labels = ProteinInformation.get_data_predictions(
            ids, fasta_file
        )
        embeddings, structures = FileManager.read_embeddings_and_structures(
            FileSetter.embeddings_and_structures_input()
        )

        proteins = dict()
        for i in range(0, 5):
            print("Load model")
            model_path = "{}{}.pt".format(model_prefix, i + 1)
            model = FileManager.load_classifier_torch(model_path)

            print("Calculate predictions")
            ml_predictor = MLPredictor(model)

            curr_proteins = ml_predictor.predict_per_protein(
                ids,
                sequences,
                embeddings,
                labels,
                max_length,
                structures=structures,
                distance_cutoff_embd=distance_cutoff_embd,
                distance_cutoff_struc=distance_cutoff_struc,
            )

            for k in curr_proteins.keys():
                if k in proteins.keys():
                    prot = proteins[k]
                    prot.add_predictions(curr_proteins[k].predictions)
                else:
                    proteins[k] = curr_proteins[k]

        for k in proteins.keys():
            proteins[k].normalize_predictions(5)

        if result_folder is not None:
            # FileManager.write_predictions(proteins, result_folder, cutoff, ri)
            pass

        return proteins

    @staticmethod
    def GCN_prediction_pipeline(
        h5filepath,
        model_prefix,
        cutoff,
        result_folder,
        ids,
        fasta_file,
        ri,
        distance_cutoff_embd=1,
        distance_cutoff_struc=1,
    ):
        """
        Use GCNConv to create predictions. Slightly different workflow, 
        since path to h5 file will be passed, 
        and structures substituted with diagonal matrices (see read_embeddings_for_GCN_inference).
        """

        print("Prepare data")
        sequences, max_length, labels = ProteinInformation.get_data_predictions(
            ids, fasta_file
        )
        embeddings, structures = FileManager.read_embeddings_for_GCN_inference(
            h5filepath
        )

        proteins = dict()
        for i in range(0, 5):
            print("Load model")
            model_path = "{}{}.pt".format(model_prefix, i + 1)
            model = FileManager.load_classifier_torch(model_path)

            print("Calculate predictions")
            ml_predictor = MLPredictor(model)

            curr_proteins = ml_predictor.predict_per_protein(
                ids,
                sequences,
                embeddings,
                labels,
                max_length,
                structures=structures,
                distance_cutoff_embd=distance_cutoff_embd,
                distance_cutoff_struc=distance_cutoff_struc,
            )

            for k in curr_proteins.keys():
                if k in proteins.keys():
                    prot = proteins[k]
                    prot.add_predictions(curr_proteins[k].predictions)
                else:
                    proteins[k] = curr_proteins[k]

        for k in proteins.keys():
            proteins[k].normalize_predictions(5)

        if result_folder is not None:
            FileManager.write_predictions(proteins, result_folder, cutoff, ri)
            pass

        return proteins
