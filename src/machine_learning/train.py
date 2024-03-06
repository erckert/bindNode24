import torch
from torch_geometric.loader import DataLoader

from machine_learning.Dataset import BindingResidueDatasetWithLabels
from machine_learning.ModelManager import initialize_untrained_model
from setup.configProcessor import get_cv_splits, get_batch_size, get_optimizer_arguments, get_epochs
from setup.generalSetup import select_device


def run_optimization():
    # TODO: implement pipeline workflow
    return None


def make_predictions(model, data_loader, loss, loss_count, optimizer, loss_function, sigmoid,
                     prediction_list, label_list, backpropagate=True):
    for data_graph in data_loader:
        if backpropagate:
            optimizer.zero_grad()
        data_graph = data_graph.to(select_device())

        predictions = model.forward(
            data_graph.x,
            data_graph.edge_index,
            data_graph.edge_index_cutoff,
            data_graph.edge_features
        )
        # don't consider padded positions for loss calculation

        loss_el = loss_function(predictions, data_graph.y)
        # pred is a tensor of shape: 69203, 3 num_nodes, data_graph.batch is tensor 69k, 2

        loss_norm = torch.sum(loss_el)
        loss += loss_norm.item()
        loss_count += 1

        predictions = sigmoid(predictions)
        prediction_list.append(predictions.detach().cpu())
        label_list.append(data_graph.y.detach().cpu())

        if backpropagate:
            loss_norm.backward()
            optimizer.step()
            torch.cuda.empty_cache()


def train_batch(model, train_loader, optimizer, loss_function, sigmoid):
    prediction_list = []
    label_list = []
    train_loss = 0
    train_loss_count = 0

    model.train()
    make_predictions(
        model,
        train_loader,
        train_loss,
        train_loss_count,
        optimizer,
        loss_function,
        sigmoid,
        prediction_list,
        label_list,
        backpropagate=True
    )

    return prediction_list, label_list


def validate_batch(model, validation_loader, loss_function, sigmoid):
    prediction_list = []
    label_list = []
    validation_loss = 0
    validation_loss_count = 0

    model.eval()
    make_predictions(
        model,
        validation_loader,
        validation_loss,
        validation_loss_count,
        None,
        loss_function,
        sigmoid,
        prediction_list,
        label_list,
        backpropagate=False
    )
    return prediction_list, label_list


def evaluate_batch(training_predictions, validation_predictions, training_labels, validation_labels):
    pass


def train_and_validate(model, training_dataset, validation_dataset):
    train_loader = DataLoader(
        training_dataset,
        batch_size=get_batch_size(only_first_value=True),
        shuffle=True,
        pin_memory=True
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=get_batch_size(only_first_value=True),
        shuffle=True
    )

    loss_function = torch.nn.BCEWithLogitsLoss(reduction="none")
    sigmoid = torch.nn.Sigmoid()
    model.to(select_device())
    optimizer = torch.optim.Adam(model.parameters(), **get_optimizer_arguments(only_first_value=True))

    for epoch in range(get_epochs(only_first_value=True)):
        torch.cuda.empty_cache()

        # training
        training_predictions, training_labels = train_batch(model, train_loader, optimizer, loss_function, sigmoid)

        # validation
        validation_predictions, validation_labels = validate_batch(model, validation_loader, loss_function, sigmoid)

        #evaluation
        evaluate_batch(training_predictions, validation_predictions, training_labels, validation_labels)
    return model


def run_training():
    dataset = BindingResidueDatasetWithLabels()
    cv_split_ids = get_cv_splits()
    nr_of_splits = len(cv_split_ids)

    for i in range(nr_of_splits):
        validation_ids = cv_split_ids[i]
        training_ids = []
        model = initialize_untrained_model()
        for j in range(nr_of_splits):
            if i != j:
                training_ids += cv_split_ids[j]

        training_dataset, validation_dataset = dataset.train_val_split(training_ids, validation_ids)
        
        train_and_validate(model, training_dataset, validation_dataset)

    return None
