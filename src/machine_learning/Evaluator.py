import math
import os
import torch
import json

from setup.configProcessor import get_result_dir, get_weight_dir, select_model_type_from_config, \
    select_mode_from_config, get_model_parameter_dict, get_optimizer_arguments


class BindingResiduePredictionEvaluator:
    def __init__(self):
        self.per_epoch_performances = {
            "loss": [],
            "mcc": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "accuracy": []
        }
        self.final_performances = {
            "loss": [],
            "mcc": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "accuracy": []
        }

    def evaluate_per_epoch(self, predictions, labels, loss, loss_count):
        tp, fp, tn, fn = self.compute_confusion_matrix_per_epoch(predictions, labels)
        performances = {
            "loss": self.compute_loss(loss, loss_count),
            "mcc": float(self.compute_mcc(tp, fp, tn, fn)),
            "precision": float(self.compute_precision(tp, fp)),
            "recall": float(self.compute_recall(tp, fn)),
            "f1": float(self.compute_f1(tp, fp, fn)),
            "accuracy": float(self.compute_accuracy(tp, fp, tn, fn))
        }
        self.per_epoch_performances["loss"].append(performances["loss"])
        self.per_epoch_performances["mcc"].append(performances["mcc"])
        self.per_epoch_performances["precision"].append(performances["precision"])
        self.per_epoch_performances["recall"].append(performances["recall"])
        self.per_epoch_performances["f1"].append(performances["f1"])
        self.per_epoch_performances["accuracy"].append(performances["accuracy"])

    def compute_loss(self, loss, loss_count):
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
    def compute_confusion_matrix_per_epoch(predictions, labels):
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for prediction_graph, label_graph in zip(predictions, labels):
            tp += torch.sum(torch.ge(prediction_graph, 0.5) * torch.ge(label_graph, 0.5))
            tn += torch.sum(torch.lt(prediction_graph, 0.5) * torch.lt(label_graph, 0.5))
            fp += torch.sum(torch.ge(prediction_graph, 0.5) * torch.lt(label_graph, 0.5))
            fn += torch.sum(torch.lt(prediction_graph, 0.5) * torch.ge(label_graph, 0.5))

        tp = float(tp)
        fp = float(fp)
        fn = float(fn)
        tn = float(tn)
        return tp, fp, tn, fn

    def write_evaluation_results(self, file_name):
        result_path = os.path.join(str(get_result_dir()), f'{file_name}.json')
        results = {
            "model_type": str(select_model_type_from_config()),
            "weight_dir": os.path.relpath(get_weight_dir()),
            "mode": str(select_mode_from_config()),
            "model_parameters": get_model_parameter_dict(True),
            "optimizer_parameters": get_optimizer_arguments(True)
        }
        with open(result_path, 'w') as fh:
            json.dump(results, fh)
