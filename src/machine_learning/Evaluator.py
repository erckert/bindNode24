import math


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

    def evaluate_per_batch(self, predictions, labels, loss, loss_count):
        tp, fp, tn, fn = self.compute_confusion_matrix(predictions, labels)
        performances = {
            "loss": self.compute_loss(loss, loss_count),
            "mcc": self.compute_mcc(tp, fp, tn, fn),
            "precision": self.compute_precision(tp, fp),
            "recall": self.compute_recall(tp, fn),
            "f1": self.compute_f1(tp, fp, fn),
            "accuracy": self.compute_accuracy(tp, fp, tn, fn)
        }
        return performances

    def compute_loss(self, losses, loss_count):
        #TODO: implement me
        pass

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
    def compute_confusion_matrix(predictions, labels):
        #TODO: implement me
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        return tp, fp, tn, fn