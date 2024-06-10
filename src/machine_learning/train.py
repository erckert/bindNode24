import torch
import os
from torch_geometric.loader import DataLoader
from pathlib import Path

from machine_learning.EarlyStopping import EarlyStopping
from machine_learning.Dataset import BindingResidueDatasetWithLabels
from machine_learning.ModelManager import initialize_untrained_model, save_classifier_torch
from machine_learning.Evaluator import BindingResiduePredictionEvaluator
from setup.configProcessor import get_cv_splits, get_optimizer_arguments, get_weight_dir, do_logging, \
    get_model_parameter_dict
from setup.generalSetup import select_device


def run_optimization(model_parameters):
    model_counter = 0
    performances = []

    for structure_cutoff in model_parameters["cutoff_structure"]:
        for nr_feature_channels in model_parameters["features"]:
            for dropout in model_parameters["dropout"]:
                for batch_size in model_parameters["batchsize"]:
                    for epochs in model_parameters["epochs"]:
                        for weights in model_parameters["weights"]:
                            for dropout_fn in model_parameters["droupout_fcn"]:
                                model_parameters_run = {
                                    "cutoff": model_parameters["cutoff"],
                                    "in_channels": model_parameters["in_channels"],
                                    "out_channels": model_parameters["out_channels"],
                                    "features": nr_feature_channels,
                                    "dropout": dropout,
                                    "epochs": epochs,
                                    "early_stopping": model_parameters["early_stopping"],
                                    "weights": weights,
                                    "cutoff_structure": structure_cutoff,
                                    "activation": model_parameters["activation"],
                                    "batchsize": batch_size,
                                    "droupout_fcn": dropout_fn,
                                    "use_dssp": model_parameters["use_dssp"],
                                    "nr_dssp_features": model_parameters["nr_dssp_features"]
                                }
                                print(f"-----------Training model{model_counter}-----------")
                                training_evaluators, validation_evaluators = \
                                    run_training(model_parameters_run, filename_model_prefix=f"model{model_counter}_")

                                training_performances = [training_evaluator.performances["all"]["f1"]
                                                         for training_evaluator in training_evaluators]
                                validation_performances = [validation_evaluator.performances["all"]["f1"]
                                                           for validation_evaluator in validation_evaluators]
                                average_training_performance = sum(training_performances)/len(training_performances)
                                average_validation_performance = \
                                    sum(validation_performances)/len(validation_performances)
                                performances.append(
                                    (average_validation_performance,
                                     average_training_performance,
                                     f"model{model_counter}",
                                     model_parameters_run)
                                )
                                model_counter += 1
    sorted_performances = sorted(performances, key=lambda tup: (tup[0], tup[1]), reverse=True)
    print(f"Best overall model was {sorted_performances[0][2]} with a validation F1 of {sorted_performances[0][0]} "
          f"and a training F1 of {sorted_performances[0][1]}.")
    print(f"Used parameters for best model:\n{sorted_performances[0][3]}")

    return None


def make_predictions(model, data_loader, optimizer, loss_function, sigmoid,
                     prediction_list, label_list, backpropagate=True):
    loss = 0
    loss_count = 0
    for data_batch, _ in data_loader:
        if backpropagate:
            optimizer.zero_grad()
        data_batch = data_batch.to(select_device())

        predictions = model.forward(
            data_batch.x,
            data_batch.edge_index,
            data_batch.edge_index_cutoff,
            data_batch.edge_features,
            data_batch.dssp_features
        )

        loss_el = loss_function(predictions, data_batch.y)
        # predictions is a tensor of shape: 69203, 3 num_nodes, data_graph.batch is tensor 69k, 2

        loss_norm = torch.sum(loss_el)
        loss += loss_norm.item()
        loss_count += 1

        predictions = sigmoid(predictions)
        prediction_list.append(predictions.detach().cpu())
        label_list.append(data_batch.y.detach().cpu())

        if backpropagate:
            loss_norm.backward()
            optimizer.step()
            torch.cuda.empty_cache()

    return loss, loss_count


def train_epoch(model, train_loader, optimizer, loss_function, sigmoid):
    prediction_list = []
    label_list = []

    model.train()
    train_loss, train_loss_count = make_predictions(
        model,
        train_loader,
        optimizer,
        loss_function,
        sigmoid,
        prediction_list,
        label_list,
        backpropagate=True
    )

    return prediction_list, label_list, train_loss, train_loss_count


def validate_epoch(model, validation_loader, loss_function, sigmoid):
    prediction_list = []
    label_list = []

    model.eval()
    validation_loss, validation_loss_count = make_predictions(
        model,
        validation_loader,
        None,
        loss_function,
        sigmoid,
        prediction_list,
        label_list,
        backpropagate=False
    )
    return prediction_list, label_list, validation_loss, validation_loss_count


def train_and_validate(model, training_dataset, validation_dataset, model_parameters):
    train_loader = DataLoader(
        training_dataset,
        batch_size=model_parameters["batchsize"],
        shuffle=True,
        pin_memory=True
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=model_parameters["batchsize"],
        shuffle=True
    )

    training_evaluator = BindingResiduePredictionEvaluator()
    validation_evaluator = BindingResiduePredictionEvaluator()

    loss_function = torch.nn.BCEWithLogitsLoss(
        reduction="none",
        pos_weight=torch.tensor(model_parameters["weights"])
    )
    sigmoid = torch.nn.Sigmoid()
    model.to(select_device())
    optimizer = torch.optim.Adam(model.parameters(), **get_optimizer_arguments(only_first_value=True))

    checkpoint_file = "checkpoint_file_early_stopping.pt"
    early_stopping_patience = 10
    early_stopping = EarlyStopping(
        patience=early_stopping_patience, delta=0.01, checkpoint_file=checkpoint_file, verbose=True
    )

    epoch_counter = 0
    for epoch in range(model_parameters["epochs"]):
        if do_logging():
            print("Epoch {}".format(epoch_counter))
            epoch_counter += 1

        torch.cuda.empty_cache()

        # training
        training_predictions, training_labels, train_loss, train_loss_count = train_epoch(
            model,
            train_loader,
            optimizer,
            loss_function,
            sigmoid
        )

        # validation
        validation_predictions, validation_labels, validation_loss, validation_loss_count = validate_epoch(
            model,
            validation_loader,
            loss_function,
            sigmoid
        )

        #evaluation
        training_evaluator.evaluate_per_epoch(
            training_predictions,
            training_labels,
            train_loss,
            train_loss_count
        )
        validation_evaluator.evaluate_per_epoch(
            validation_predictions,
            validation_labels,
            validation_loss,
            validation_loss_count
        )

        if model_parameters["early_stopping"]:
            f1_validation = validation_evaluator.performances["all"]["f1"][-1] * (-1)

            # eval_val = val_loss
            early_stopping(f1_validation, model, do_logging())
            if early_stopping.early_stop:
                training_evaluator.remove_last_x_performances(x=early_stopping_patience)
                validation_evaluator.remove_last_x_performances(x=early_stopping_patience)
                break

    if model_parameters["early_stopping"]:
        # load best model
        model = torch.load(checkpoint_file)

    return model, training_evaluator, validation_evaluator


def run_training(model_parameters, filename_model_prefix=""):
    structure_cutoff = model_parameters["cutoff_structure"]
    dataset = BindingResidueDatasetWithLabels(structure_cutoff)
    cv_split_ids = get_cv_splits()
    nr_of_splits = len(cv_split_ids)
    training_evaluators = []
    validation_evaluators = []

    for i in range(nr_of_splits):
        validation_ids = cv_split_ids[i]
        training_ids = []
        model = initialize_untrained_model(model_parameters)
        for j in range(nr_of_splits):
            if i != j:
                training_ids += cv_split_ids[j]

        training_dataset, validation_dataset = dataset.train_val_split(training_ids, validation_ids)
        
        trained_model, training_evaluator, validation_evaluator \
            = train_and_validate(model, training_dataset, validation_dataset, model_parameters)

        # save model
        if not Path(get_weight_dir()).exists():
            Path(get_weight_dir()).mkdir(parents=True, exist_ok=True)
        model_save_path = os.path.join(str(get_weight_dir()), f"{filename_model_prefix}trained_model_{i}.state_dict")
        save_classifier_torch(trained_model, model_save_path)

        # Write training details and final performance to file
        training_evaluator.write_evaluation_results(f"{filename_model_prefix}model_training_{i}")
        validation_evaluator.write_evaluation_results(f"{filename_model_prefix}model_validation_{i}")

        training_evaluators.append(training_evaluator)
        validation_evaluators.append(validation_evaluator)

    return training_evaluators, validation_evaluators
