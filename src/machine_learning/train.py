import torch
from torch_geometric.loader import DataLoader

from machine_learning.Dataset import BindingResidueDatasetWithLabels
from machine_learning.ModelManager import initialize_untrained_model
from setup.configProcessor import get_cv_splits, get_batch_size, get_optimizer_arguments, get_epochs
from setup.generalSetup import select_device


def run_optimization():
    # TODO: implement pipeline workflow
    return None


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

        train_loss = val_loss = 0
        train_loss_count = val_loss_count = 0

        # training
        model.train()
        for data_graph in train_loader:
            optimizer.zero_grad()
            data_graph = data_graph.to(select_device())

            pred = model.forward(
                data_graph.x,
                data_graph.edge_index, data_graph.edge_index_cutoff, data_graph.edge_features
            )
            # don't consider padded positions for loss calculation

            loss_el = loss_function(pred, data_graph.y)
            # pred is a tensor of shape: 69203, 3 num_nodes, data_graph.batch is tensor 69k, 2

            loss_norm = torch.sum(loss_el)
            train_loss += loss_norm.item()
            train_loss_count += 1

            pred = sigmoid(pred)

            loss_norm.backward()
            optimizer.step()
            torch.cuda.empty_cache()
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
