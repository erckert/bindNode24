import torch
from setup.configProcessor import select_model_type_from_config, get_in_channels, get_feature_channels, get_dropouts, \
    get_out_channels
from misc.enums import ModelType
from machine_learning.Models import GCNConvModel, SAGEConvModel, SAGEConvMLPModel, SAGEConvGATMLPModel


def initialize_model_with_config_params():
    model_type = select_model_type_from_config()
    classifier = None
    match model_type:
        case ModelType.GCNCONV:
            classifier = GCNConvModel(
                in_channels=get_in_channels(),
                feature_channels=get_feature_channels(only_first_value=True),
                out_channels=get_out_channels(),
                dropout=get_dropouts(only_first_value=True)
            )
        case ModelType.SAGECONV:
            classifier = SAGEConvModel()
        case ModelType.SAGECONVMLP:
            classifier = SAGEConvMLPModel()
        case ModelType.SAGECONVGATMLP:
            classifier = SAGEConvGATMLPModel()
    return classifier


def save_classifier_torch(classifier, model_path):
    """Save pre-trained model"""
    torch.save(classifier.state_dict(), model_path)


def load_classifier_torch(model_path):
    """ Load pre-saved model """
    classifier = initialize_model_with_config_params()
    classifier.load_state_dict(torch.load(model_path))
    return classifier


def initialize_untrained_model():
    classifier = initialize_model_with_config_params()
    return classifier
