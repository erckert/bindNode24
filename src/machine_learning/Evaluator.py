import math
import os
import torch
import json
import numpy as np
from scipy.stats import t

from setup.configProcessor import get_result_dir, get_weight_dir, select_model_type_from_config, \
    select_mode_from_config, get_model_parameter_dict, get_optimizer_arguments, get_cutoff
from misc.enums import LabelType


def average(performance_list: list):
    return sum(performance_list)/len(performance_list)


def sum_confusion_matrices(confusion_matrix_list: list):
    tp = sum([confusion_matrix['tp'] for confusion_matrix in confusion_matrix_list])
    fp = sum([confusion_matrix['fp'] for confusion_matrix in confusion_matrix_list])
    fn = sum([confusion_matrix['fn'] for confusion_matrix in confusion_matrix_list])
    tn = sum([confusion_matrix['tn'] for confusion_matrix in confusion_matrix_list])
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


class BindingResiduePredictionEvaluator:
    def __init__(self):
        self.performances = {
            "all": {
                "confusion_matrix": [],
                "loss": [],
                "mcc": [],
                "precision": [],
                "recall": [],
                "f1": [],
                "accuracy": []
            },
            LabelType.METAL.name: {
                "confusion_matrix": [],
                "mcc": [],
                "precision": [],
                "recall": [],
                "f1": [],
                "accuracy": []
            },
            LabelType.NUCLEAR.name: {
                "confusion_matrix": [],
                "mcc": [],
                "precision": [],
                "recall": [],
                "f1": [],
                "accuracy": []
            },
            LabelType.SMALL.name: {
                "confusion_matrix": [],
                "mcc": [],
                "precision": [],
                "recall": [],
                "f1": [],
                "accuracy": []
            },
        }

    def evaluate_per_epoch(self, predictions, labels, loss, loss_count):
        tp, fp, tn, fn = self.compute_confusion_matrix_per_epoch(predictions, labels)
        confusion_matrix = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }
        performances = {
            "loss": self.compute_loss(loss, loss_count),
            "mcc": float(self.compute_mcc(tp, fp, tn, fn)),
            "precision": float(self.compute_precision(tp, fp)),
            "recall": float(self.compute_recall(tp, fn)),
            "f1": float(self.compute_f1(tp, fp, fn)),
            "accuracy": float(self.compute_accuracy(tp, fp, tn, fn))
        }

        self.performances["all"]["loss"].append(performances["loss"])
        self.performances["all"]["mcc"].append(performances["mcc"])
        self.performances["all"]["precision"].append(performances["precision"])
        self.performances["all"]["recall"].append(performances["recall"])
        self.performances["all"]["f1"].append(performances["f1"])
        self.performances["all"]["accuracy"].append(performances["accuracy"])
        self.performances["all"]["confusion_matrix"].append(confusion_matrix)

        self.compute_per_class_performances(predictions, labels, LabelType.METAL)
        self.compute_per_class_performances(predictions, labels, LabelType.SMALL)
        self.compute_per_class_performances(predictions, labels, LabelType.NUCLEAR)

    def evaluate_per_protein(self, predictions, labels, loss, loss_count):
        for prediction, label in zip(predictions, labels):
            self.evaluate_per_epoch(prediction, label, loss, loss_count)

    def remove_last_x_performances_by_result_section(self, x, result_section):
        if result_section == "all":
            del self.performances[result_section]["loss"][-x:]

        del self.performances[result_section]["mcc"][-x:]
        del self.performances[result_section]["precision"][-x:]
        del self.performances[result_section]["recall"][-x:]
        del self.performances[result_section]["f1"][-x:]
        del self.performances[result_section]["accuracy"][-x:]
        del self.performances[result_section]["confusion_matrix"][-x:]

    def remove_last_x_performances(self, x):
        self.remove_last_x_performances_by_result_section(x, "all")
        self.remove_last_x_performances_by_result_section(x, LabelType.METAL.name)
        self.remove_last_x_performances_by_result_section(x, LabelType.NUCLEAR.name)
        self.remove_last_x_performances_by_result_section(x, LabelType.SMALL.name)

    @staticmethod
    def compute_loss(loss, loss_count):
        return loss/loss_count

    @staticmethod
    def compute_mcc(tp, fp, tn, fn):
        if (tp > 0 or fp > 0) and (tp > 0 or fn > 0) and (tn > 0 or fp > 0) and (tn > 0 or fn > 0):
            return (tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        else:
            return 0

    @staticmethod
    def compute_precision(tp, fp):
        if tp > 0 or fp > 0:
            return tp / (tp + fp)
        else:
            return 0

    @staticmethod
    def compute_recall(tp, fn):
        if tp > 0 or fn > 0:
            return tp / (tp + fn)
        else:
            return 0

    def compute_f1(self, tp, fp, fn):
        recall = self.compute_recall(tp, fn)
        precision = self.compute_precision(tp, fp)
        if recall > 0 or precision > 0:
            return 2 * recall * precision / (recall + precision)
        else:
            return 0

    @staticmethod
    def compute_accuracy(tp, fp, tn, fn):
        return (tp + tn) / (tp + tn + fn + fp)

    @staticmethod
    def compute_covOneBind(per_protein_confusion_matrices):
        predicted_binding = 0
        has_binding = 0
        for confusion_matrix in per_protein_confusion_matrices:
            if confusion_matrix["tp"] + confusion_matrix["fp"] >= 1 \
                    and confusion_matrix["tp"] + confusion_matrix["fn"] >= 1:
                predicted_binding += 1
            if confusion_matrix["tp"] + confusion_matrix["fn"] >= 1:
                has_binding += 1
        return predicted_binding/has_binding

    @staticmethod
    def compute_ci(per_protein_performances):
        ci = round(np.std(per_protein_performances) / math.sqrt(len(per_protein_performances))
                   * t.ppf((1 + 0.95) / 2, len(per_protein_performances)), 3)
        return ci

    @staticmethod
    def compute_confusion_matrix_per_epoch(predictions, labels):
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for prediction_graph, label_graph in zip(predictions, labels):
            # if ANY value in a per-residue vector of length 3 is above the cutoff, the position is considered as
            # binding. Only if ALL values in the vector are below, it is considered non-binding
            tp += torch.sum(
                torch.any(torch.ge(prediction_graph, get_cutoff()), 1) *
                torch.any(torch.ge(label_graph, get_cutoff()), 1)
            )
            tn += torch.sum(
                torch.all(torch.lt(prediction_graph, get_cutoff()), 1) *
                torch.all(torch.lt(label_graph, get_cutoff()), 1)
            )
            fp += torch.sum(
                torch.any(torch.ge(prediction_graph, get_cutoff()), 1) *
                torch.all(torch.lt(label_graph, get_cutoff()), 1)
            )
            fn += torch.sum(
                torch.all(torch.lt(prediction_graph, get_cutoff()), 1) *
                torch.any(torch.ge(label_graph, get_cutoff()), 1)
            )

        tp = float(tp)
        fp = float(fp)
        fn = float(fn)
        tn = float(tn)
        return tp, fp, tn, fn

    @staticmethod
    def compute_confusion_matrix_per_class_per_epoch(predictions, labels, prediction_class):
        prediction_class_entry = prediction_class.value
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for prediction_graph, label_graph in zip(predictions, labels):
            # only if the indicated class value in a per-residue vector of length 3 is above the cutoff, the position is
            # considered as binding. If this classes values in the vector is below, it is considered non-binding
            tp += torch.sum(
                torch.ge(prediction_graph[:, prediction_class_entry], get_cutoff()) *
                torch.ge(label_graph[:, prediction_class_entry], get_cutoff())
            )
            tn += torch.sum(
                torch.lt(prediction_graph[:, prediction_class_entry], get_cutoff()) *
                torch.lt(label_graph[:, prediction_class_entry], get_cutoff())
            )
            fp += torch.sum(
                torch.ge(prediction_graph[:, prediction_class_entry], get_cutoff()) *
                torch.lt(label_graph[:, prediction_class_entry], get_cutoff())
            )
            fn += torch.sum(
                torch.lt(prediction_graph[:, prediction_class_entry], get_cutoff()) *
                torch.ge(label_graph[:, prediction_class_entry], get_cutoff())
            )

        tp = float(tp)
        fp = float(fp)
        fn = float(fn)
        tn = float(tn)
        return tp, fp, tn, fn

    def compute_per_class_performances(self, predictions, labels, prediction_class):
        tp, fp, tn, fn = self.compute_confusion_matrix_per_class_per_epoch(predictions, labels, prediction_class)
        confusion_matrix = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }
        performances = {
            "mcc": float(self.compute_mcc(tp, fp, tn, fn)),
            "precision": float(self.compute_precision(tp, fp)),
            "recall": float(self.compute_recall(tp, fn)),
            "f1": float(self.compute_f1(tp, fp, fn)),
            "accuracy": float(self.compute_accuracy(tp, fp, tn, fn))
        }
        if (tp + fp + fn) > 0:
        # only add performance if this protein binds or if one residue was predicted to bind
            self.performances[prediction_class.name]["mcc"].append(performances["mcc"])
            self.performances[prediction_class.name]["precision"].append(performances["precision"])
            self.performances[prediction_class.name]["recall"].append(performances["recall"])
            self.performances[prediction_class.name]["f1"].append(performances["f1"])
            self.performances[prediction_class.name]["accuracy"].append(performances["accuracy"])
            self.performances[prediction_class.name]["confusion_matrix"].append(confusion_matrix)

    def write_evaluation_results(self, file_name):
        result_path = os.path.join(str(get_result_dir()), f'{file_name}.json')
        results = {
            "model_type": str(select_model_type_from_config()),
            "weight_dir": os.path.relpath(get_weight_dir()),
            "mode": str(select_mode_from_config()),
            "model_parameters": get_model_parameter_dict(True),
            "optimizer_parameters": get_optimizer_arguments(True),
            "early_stopping_after": len(self.performances["all"]["f1"]),
            "performance": {
                "all": {
                    "confusion_matrix": self.performances["all"]["confusion_matrix"][-1],
                    "loss": self.performances["all"]["loss"][-1],
                    "mcc": self.performances["all"]["mcc"][-1],
                    "precision": self.performances["all"]["precision"][-1],
                    "recall": self.performances["all"]["recall"][-1],
                    "f1": self.performances["all"]["f1"][-1],
                    "accuracy": self.performances["all"]["accuracy"][-1]
                },
                LabelType.METAL.name: {
                    "confusion_matrix": self.performances[LabelType.METAL.name]["confusion_matrix"][-1],
                    "mcc": self.performances[LabelType.METAL.name]["mcc"][-1],
                    "precision": self.performances[LabelType.METAL.name]["precision"][-1],
                    "recall": self.performances[LabelType.METAL.name]["recall"][-1],
                    "f1": self.performances[LabelType.METAL.name]["f1"][-1],
                    "accuracy": self.performances[LabelType.METAL.name]["accuracy"][-1]
                },
                LabelType.NUCLEAR.name: {
                    "confusion_matrix": self.performances[LabelType.NUCLEAR.name]["confusion_matrix"][-1],
                    "mcc": self.performances[LabelType.NUCLEAR.name]["mcc"][-1],
                    "precision": self.performances[LabelType.NUCLEAR.name]["precision"][-1],
                    "recall": self.performances[LabelType.NUCLEAR.name]["recall"][-1],
                    "f1": self.performances[LabelType.NUCLEAR.name]["f1"][-1],
                    "accuracy": self.performances[LabelType.NUCLEAR.name]["accuracy"][-1]
                },
                LabelType.SMALL.name: {
                    "confusion_matrix": self.performances[LabelType.SMALL.name]["confusion_matrix"][-1],
                    "mcc": self.performances[LabelType.SMALL.name]["mcc"][-1],
                    "precision": self.performances[LabelType.SMALL.name]["precision"][-1],
                    "recall": self.performances[LabelType.SMALL.name]["recall"][-1],
                    "f1": self.performances[LabelType.SMALL.name]["f1"][-1],
                    "accuracy": self.performances[LabelType.SMALL.name]["accuracy"][-1]
                }
            }
        }
        with open(result_path, 'w') as fh:
            json.dump(results, fh)

    def write_evaluation_results_per_protein_averages(self, file_name):
        result_path = os.path.join(str(get_result_dir()), f'{file_name}.json')
        results = {
            "model_type": str(select_model_type_from_config()),
            "weight_dir": os.path.relpath(get_weight_dir()),
            "mode": str(select_mode_from_config()),
            "model_parameters": get_model_parameter_dict(True),
            "optimizer_parameters": get_optimizer_arguments(True),
            "performance": {
                "all": {
                    "confusion_matrix": sum_confusion_matrices(self.performances["all"]["confusion_matrix"]),
                    "mcc": average(self.performances["all"]["mcc"]),
                    "ci_mcc": self.compute_ci(self.performances["all"]["mcc"]),
                    "precision": average(self.performances["all"]["precision"]),
                    "ci_precision": self.compute_ci(self.performances["all"]["precision"]),
                    "recall": average(self.performances["all"]["recall"]),
                    "ci_recall": self.compute_ci(self.performances["all"]["recall"]),
                    "f1": average(self.performances["all"]["f1"]),
                    "ci_f1": self.compute_ci(self.performances["all"]["f1"]),
                    "accuracy": average(self.performances["all"]["accuracy"]),
                    "ci_accuracy": self.compute_ci(self.performances["all"]["accuracy"]),
                    "covonebind": self.compute_covOneBind(self.performances["all"]["confusion_matrix"])
                },
                LabelType.METAL.name: {
                    "confusion_matrix": sum_confusion_matrices(self.performances[LabelType.METAL.name]["confusion_matrix"]),
                    "mcc": average(self.performances[LabelType.METAL.name]["mcc"]),
                    "ci_mcc": self.compute_ci(self.performances[LabelType.METAL.name]["mcc"]),
                    "precision": average(self.performances[LabelType.METAL.name]["precision"]),
                    "ci_precision": self.compute_ci(self.performances[LabelType.METAL.name]["precision"]),
                    "recall": average(self.performances[LabelType.METAL.name]["recall"]),
                    "ci_recall": self.compute_ci(self.performances[LabelType.METAL.name]["recall"]),
                    "f1": average(self.performances[LabelType.METAL.name]["f1"]),
                    "ci_f1": self.compute_ci(self.performances[LabelType.METAL.name]["f1"]),
                    "accuracy": average(self.performances[LabelType.METAL.name]["accuracy"]),
                    "ci_accuracy": self.compute_ci(self.performances[LabelType.METAL.name]["accuracy"]),
                    "covonebind": self.compute_covOneBind(self.performances[LabelType.METAL.name]["confusion_matrix"])
                },
                LabelType.NUCLEAR.name: {
                    "confusion_matrix": sum_confusion_matrices(self.performances[LabelType.NUCLEAR.name]["confusion_matrix"]),
                    "mcc": average(self.performances[LabelType.NUCLEAR.name]["mcc"]),
                    "ci_mcc": self.compute_ci(self.performances[LabelType.NUCLEAR.name]["mcc"]),
                    "precision": average(self.performances[LabelType.NUCLEAR.name]["precision"]),
                    "ci_precision": self.compute_ci(self.performances[LabelType.NUCLEAR.name]["precision"]),
                    "recall": average(self.performances[LabelType.NUCLEAR.name]["recall"]),
                    "ci_recall": self.compute_ci(self.performances[LabelType.NUCLEAR.name]["recall"]),
                    "f1": average(self.performances[LabelType.NUCLEAR.name]["f1"]),
                    "ci_f1": self.compute_ci(self.performances[LabelType.NUCLEAR.name]["f1"]),
                    "accuracy": average(self.performances[LabelType.NUCLEAR.name]["accuracy"]),
                    "ci_accuracy": self.compute_ci(self.performances[LabelType.NUCLEAR.name]["accuracy"]),
                    "covonebind": self.compute_covOneBind(self.performances[LabelType.NUCLEAR.name]["confusion_matrix"])
                },
                LabelType.SMALL.name: {
                    "confusion_matrix": sum_confusion_matrices(self.performances[LabelType.SMALL.name]["confusion_matrix"]),
                    "mcc": average(self.performances[LabelType.SMALL.name]["mcc"]),
                    "ci_mcc": self.compute_ci(self.performances[LabelType.SMALL.name]["mcc"]),
                    "precision": average(self.performances[LabelType.SMALL.name]["precision"]),
                    "ci_precision": self.compute_ci(self.performances[LabelType.SMALL.name]["precision"]),
                    "recall": average(self.performances[LabelType.SMALL.name]["recall"]),
                    "ci_recall": self.compute_ci(self.performances[LabelType.SMALL.name]["recall"]),
                    "f1": average(self.performances[LabelType.SMALL.name]["f1"]),
                    "ci_f1": self.compute_ci(self.performances[LabelType.SMALL.name]["f1"]),
                    "accuracy": average(self.performances[LabelType.SMALL.name]["accuracy"]),
                    "ci_accuracy": self.compute_ci(self.performances[LabelType.SMALL.name]["accuracy"]),
                    "covonebind": self.compute_covOneBind(self.performances[LabelType.SMALL.name]["confusion_matrix"])
                }
            }
        }
        with open(result_path, 'w') as fh:
            json.dump(results, fh)
