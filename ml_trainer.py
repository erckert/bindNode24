from sklearn.model_selection import PredefinedSplit
from collections import defaultdict
import itertools as it
import torch
from assess_performance import ModelPerformance, PerformanceEpochs
from config import FileManager, GeneralInformation
from pytorchtools import EarlyStopping

from bindNode23.dataset import GraphDataset
from bindNode23 import gnn_architectures
from bindNode23.gnn_architectures import BindNode23GCN
from bindNode23.logging_utils import log_epoch_metrics_to_wandb, LOGGING
from torch_geometric.loader import DataLoader
from bindNode23.logging_utils import finish_run_in_wandb, init_log_run_to_wandb, LOGGING

class MLTrainer(object):
    def __init__(self, pos_weights, batch_size=32):
        self.batch_size = batch_size
        if torch.cuda.is_available():
            self.device = "cuda:0"
        else:
            self.device = "cpu"

        self.pos_weights = torch.tensor(pos_weights).to(self.device)

    def train_validate(
        self,
        params,
        train_ids,
        validation_ids,
        sequences,
        embeddings,
        labels,
        max_length,
        structures,
        verbose=False,
    ):
        """
        Train & validate predictor for one set of parameters and ids
        :param params:
        :param train_ids:
        :param validation_ids:
        :param sequences:
        :param embeddings:
        :param labels:
        :param max_length:
        :param verbose:
        :return:
        """

        model, train_performance, val_performance = self._train_validate(
            params,
            train_ids,
            validation_ids,
            sequences,
            embeddings,
            labels,
            max_length,
            structures=structures,
            verbose=verbose,
        )

        (
            train_loss,
            train_acc,
            train_prec,
            train_recall,
            train_f1,
            train_mcc,
        ) = train_performance.get_performance_last_epoch()
        (
            val_loss,
            val_acc,
            val_prec,
            val_recall,
            val_f1,
            val_mcc,
        ) = val_performance.get_performance_last_epoch()

        print(
            "Train loss: {:.3f}, Prec: {:.3f}, Recall: {:.3f}, F1: {:.3f}, MCC: {:.3f}".format(
                train_loss, train_prec, train_recall, train_f1, train_mcc
            )
        )
        print(
            "Val loss: {:.3f}, Prec: {:.3f}, Recall: {:.3f}, F1: {:.3f}, MCC: {:.3f}".format(
                val_loss, val_prec, val_recall, val_f1, val_mcc
            )
        )


        return model

    def cross_validate(
        self,
        params,
        ids,
        fold_array,
        sequences,
        embeddings,
        labels,
        max_length,
        structures,
        result_file,
        verbose=True,
    ):
        """
        Perform cross-validation to optimize hyperparameters
        :param params:
        :param ids:
        :param fold_array:
        :param sequences:
        :param embeddings:
        :param labels:
        :param max_length:
        :param result_file:
        :return:
        """
        ps = PredefinedSplit(fold_array)

        # create parameter grid
        param_sets = defaultdict(dict)
        sorted_keys = sorted(params.keys())
        param_combos = it.product(*(params[s] for s in sorted_keys))
        counter = 0
        for p in list(param_combos):
            curr_params = list(p)
            param_dict = dict(zip(sorted_keys, curr_params))
            param_sets[counter] = param_dict

            counter += 1

        best_score = 0
        best_params = dict()  # save best parameter set
        best_classifier = None  # save best classifier
        performance = defaultdict(
            list
        )  # save performance for each parameter combination

        params_counter = 1

        for p in param_sets.keys():
            curr_params = param_sets[p]
            try:
                self.batch_size = curr_params["batchsize"]
                print("{}\t{}".format(params_counter, curr_params))
                if LOGGING:
                    init_log_run_to_wandb(params=curr_params)
                model = None

                train_model_performance = ModelPerformance()
                val_model_performance = ModelPerformance()

                for train_index, test_index in ps.split():
                    train_ids, validation_ids = ids[train_index], ids[test_index]

                    model, train_performance, val_performance = self._train_validate(
                        curr_params,
                        train_ids,
                        validation_ids,
                        sequences,
                        embeddings,
                        labels,
                        max_length,
                        structures=structures,
                        verbose=verbose,
                    )

                    (
                        train_loss,
                        train_acc,
                        train_prec,
                        train_recall,
                        train_f1,
                        train_mcc,
                    ) = train_performance.get_performance_last_epoch()
                    (
                        val_loss,
                        val_acc,
                        val_prec,
                        val_recall,
                        val_f1,
                        val_mcc,
                    ) = val_performance.get_performance_last_epoch()

                    train_model_performance.add_single_performance(
                        train_loss, train_acc, train_prec, train_recall, train_f1, train_mcc
                    )

                    val_model_performance.add_single_performance(
                        val_loss, val_acc, val_prec, val_recall, val_f1, val_mcc
                    )

                if LOGGING:
                    finish_run_in_wandb()

                # take average over all splits
                (
                    train_loss,
                    train_acc,
                    train_prec,
                    train_recall,
                    train_f1,
                    train_mcc,
                ) = train_model_performance.get_mean_performance()
                (
                    val_loss,
                    val_acc,
                    val_prec,
                    val_recall,
                    val_f1,
                    val_mcc,
                ) = val_model_performance.get_mean_performance()

                performance["train_precision"].append(train_prec)
                performance["train_recall"].append(train_recall)
                performance["train_f1"].append(train_f1)
                performance["train_mcc"].append(train_mcc)
                performance["train_acc"].append(train_acc)
                performance["train_loss"].append(train_loss)

                performance["val_precision"].append(val_prec)
                performance["val_recall"].append(val_recall)
                performance["val_f1"].append(val_f1)
                performance["val_mcc"].append(val_mcc)
                performance["val_acc"].append(val_acc)
                performance["val_loss"].append(val_loss)

                for param in curr_params.keys():
                    performance[param].append(curr_params[param])

                if val_f1 > best_score:
                    best_score = val_f1
                    best_params = curr_params
                    best_classifier = model

                params_counter += 1
            except Exception as e:
                
                if LOGGING:
                    finish_run_in_wandb()
                print(e)
                


        FileManager.save_cv_results(performance, result_file)

        print(best_score)
        print(best_params)

        return best_classifier

    def _train_validate(
        self,
        params,
        train_ids,
        validation_ids,
        sequences,
        embeddings,
        labels,
        max_length,
        structures,
        verbose=False,
    ):
        """
        Train and validate bindEmbed21DL model
        :param params:
        :param train_ids:
        :param validation_ids:
        :param sequences:
        :param labels:
        :param max_length:
        :param verbose:
        :return:
        """

        # define data sets TODO pass distance cutoff higher in hierarchy
        train_set = GraphDataset(
            train_ids,
            embeddings,
            sequences,
            labels,
            distance_cutoff_embd=params["cutoff_embd"],
            distance_cutoff_struc=params["cutoff_struc"],
            structures=structures,
        )
        print(f"Len of train dataset: {len(train_set)}")
        validation_set = GraphDataset(
            validation_ids,
            embeddings,
            sequences,
            labels,
            distance_cutoff_embd=params["cutoff_embd"],
            distance_cutoff_struc=params["cutoff_struc"],
            structures=structures,
        )
        print(f"Len of val dataset: {len(validation_set)}")
        train_loader = DataLoader(
            train_set, batch_size=self.batch_size, shuffle=True, pin_memory=True
        )
        validation_loader = DataLoader(
            validation_set, batch_size=self.batch_size, shuffle=True
        )

        pos_weights = (
            self.pos_weights
        )  
        loss_fun = torch.nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weights)
        sigm = torch.nn.Sigmoid()
        architecture_ = getattr(
            gnn_architectures, params["architecture"]
        )  # TODO rework params to be a namedtuple!
        model = architecture_(
            in_channels=train_set.get_input_dimensions(),
            feature_channels=params["features"],
            out_channels=3,
            dropout=params["dropout"],
            heads = params["heads"],
            kernel_size = params["kernel"],
            stride = params["stride"],
            padding = params["kernel"]//2,
            activation = params["activation"],
            dropout_fcn = params["dropout_fcn"],
        )
        model.to(self.device)
        optim_args = {
            "lr": params["lr"],
            "betas": params["betas"],
            "eps": params["eps"],
            "weight_decay": params["weight_decay"],
        }
        optimizer = torch.optim.Adam(model.parameters(), **optim_args)

        checkpoint_file = "checkpoint_early_stopping.pt"
        early_stopping = EarlyStopping(
            patience=10, delta=0.01, checkpoint_file=checkpoint_file, verbose=True
        )

        train_performance = PerformanceEpochs()
        validation_performance = PerformanceEpochs()

        num_epochs = 0

        for epoch in range(params["epochs"]):
            torch.cuda.empty_cache()
            if verbose:
                print("Epoch {}".format(epoch))

            train_loss = val_loss = 0
            train_loss_count = val_loss_count = 0
            train_tp = train_tn = train_fn = train_fp = 0
            val_tp = val_tn = val_fn = val_fp = 0

            train_acc = train_prec = train_rec = train_f1 = train_mcc = 0
            val_acc = val_prec = val_rec = val_f1 = val_mcc = 0

            # training
            model.train()
            for in_graph in train_loader:
                optimizer.zero_grad()
                in_graph = in_graph.to(self.device)

                pred = model.forward(
                    in_graph.x, in_graph.edge_index, in_graph.edge_index2, edge_features=in_graph.edge_attr
                )
                # don't consider padded positions for loss calculation

                loss_el = loss_fun(
                    pred, in_graph.y
                )  # pred is a tensor of shape: 69203, 3 num_nodes, in_graph.batch is tensor 69k, 2

                loss_norm = torch.sum(loss_el)
                train_loss += loss_norm.item()
                train_loss_count += 1

                pred = sigm(pred)
                (
                    tp,
                    fp,
                    tn,
                    fn,
                    acc,
                    prec,
                    rec,
                    f1,
                    mcc,
                ) = train_performance.get_performance_batch(
                    pred.detach().cpu(), in_graph.y.detach().cpu()
                )

                train_tp += tp
                train_fp += fp
                train_tn += tn
                train_fn += fn

                train_acc += acc
                train_prec += prec
                train_rec += rec
                train_f1 += f1
                train_mcc += mcc

                loss_norm.backward()
                optimizer.step()
                torch.cuda.empty_cache()

            # validation
            model.eval()
            with torch.no_grad():
                for in_graph in validation_loader:
                    in_graph = in_graph.to(self.device)

                    pred = model.forward(
                        in_graph.x, in_graph.edge_index, in_graph.edge_index2, edge_features=in_graph.edge_attr
                    )
                    # don't consider padded positions for loss calculation

                    loss_el = loss_fun(
                        pred, in_graph.y
                    )  # pred is a tensor of shape: 69203, 3 num_nodes, in_graph.batch is tensor 69k, 2

                    loss_norm = torch.sum(loss_el)
                    # print("VAL LOSS IS: ", loss_norm.item())
                    val_loss += torch.sum(loss_norm).item()
                    val_loss_count += 1
                    pred = sigm(pred)
                    (
                        tp,
                        fp,
                        tn,
                        fn,
                        acc,
                        prec,
                        rec,
                        f1,
                        mcc,
                    ) = train_performance.get_performance_batch(
                        pred.detach().cpu(), in_graph.y.detach().cpu()
                    )

                    val_tp += tp
                    val_fp += fp
                    val_tn += tn
                    val_fn += fn

                    val_acc += acc
                    val_prec += prec
                    val_rec += rec
                    val_f1 += f1
                    val_mcc += mcc

            train_loss = train_loss / (train_loss_count)
            val_loss = val_loss / (val_loss_count)

            train_acc = train_acc / train_loss_count
            train_prec = train_prec / train_loss_count
            train_rec = train_rec / train_loss_count
            train_f1 = train_f1 / train_loss_count
            train_mcc = train_mcc / train_loss_count

            val_acc = val_acc / val_loss_count
            val_prec = val_prec / val_loss_count
            val_rec = val_rec / val_loss_count
            val_f1 = val_f1 / val_loss_count
            val_mcc = val_mcc / val_loss_count

            if verbose:
                print(
                    "Train loss: {:.3f}, Prec: {:.3f}, Recall: {:.3f}, F1: {:.3f}, MCC: {:.3f}".format(
                        train_loss, train_prec, train_rec, train_f1, train_mcc
                    )
                )
                print(
                    "TP: {}, FP: {}, TN: {}, FN: {}".format(
                        train_tp, train_fp, train_tn, train_fn
                    )
                )
                print(
                    "Val loss: {:.3f}, Prec: {:.3f}, Recall: {:.3f}, F1: {:.3f}, MCC: {:.3f}".format(
                        val_loss, val_prec, val_rec, val_f1, val_mcc
                    )
                )
                print(
                    "TP: {}, FP: {}, TN: {}, FN: {}".format(
                        val_tp, val_fp, val_tn, val_fn
                    )
                )

            # append average performance for this epoch
            train_performance.add_performance_epoch(
                train_loss, train_mcc, train_prec, train_rec, train_f1, train_acc
            )
            validation_performance.add_performance_epoch(
                val_loss, val_mcc, val_prec, val_rec, val_f1, val_acc
            )
            # build a dictionary from metrics
            metrics_dict = {
                "Epoch": epoch,
                "Train_loss": train_loss,
                "Train_Prec": train_prec,
                "Train_Recall": train_rec,
                "Train_F1": train_f1,
                "Train_MCC": train_mcc,
                "Val_loss": val_loss,
                "Val_Prec": val_prec,
                "Val_Recall": val_rec,
                "Val_F1": val_f1,
                "Val_MCC": val_mcc
            }
            if LOGGING:
                log_epoch_metrics_to_wandb(metrics=metrics_dict)

            num_epochs += 1

            # stop training if F1 score doesn't improve anymore
            if "early_stopping" in params.keys() and params["early_stopping"]:
                eval_val = val_f1* (-1)
                
                # eval_val = val_loss
                early_stopping(eval_val, model, verbose)
                if early_stopping.early_stop:
                    break

        if (
            "early_stopping" in params.keys() and params["early_stopping"]
        ):  # load best model
            model = torch.load(checkpoint_file)
        torch.cuda.empty_cache()

        return model, train_performance, validation_performance
