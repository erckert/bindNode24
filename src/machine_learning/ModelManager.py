import torch
import torch.nn.functional as F

from setup.configProcessor import select_model_type_from_config, get_in_channels, get_feature_channels, get_dropouts, \
    get_out_channels, get_activation, get_dropouts_fcn, get_additional_channels, use_dssp
from misc.enums import ModelType
from machine_learning.Models import GCNConvModel, SAGEConvModel, SAGEConvMLPModel, SAGEConvGATMLPModel


def initialize_model_with_config_params(model_parameters):
    model_type = select_model_type_from_config()
    classifier = None
    match model_type:
        case ModelType.GCNCONV:
            classifier = GCNConvModel(
                in_channels=model_parameters["in_channels"],
                feature_channels=model_parameters["features"],
                additional_channels=model_parameters["nr_dssp_features"],
                out_channels=model_parameters["out_channels"],
                dropout=model_parameters["dropout"],
                use_additional_channels=model_parameters["use_dssp"]
            )
        case ModelType.SAGECONV:
            classifier = SAGEConvModel(
                in_channels=model_parameters["in_channels"],
                feature_channels=model_parameters["features"],
                additional_channels=model_parameters["nr_dssp_features"],
                out_channels=model_parameters["out_channels"],
                dropout=model_parameters["dropout"],
                activation=eval(model_parameters["activation"]),
                use_additional_channels=model_parameters["use_dssp"]
            )
        case ModelType.SAGECONVMLP:
            classifier = SAGEConvMLPModel(
                in_channels=model_parameters["in_channels"],
                feature_channels=model_parameters["features"],
                additional_channels=model_parameters["nr_dssp_features"],
                out_channels=model_parameters["out_channels"],
                dropout=model_parameters["dropout"],
                activation=eval(model_parameters["activation"]),
                heads=4, #TODO: Add heads to config?
                dropout_fcn=model_parameters["droupout_fcn"],
                use_additional_channels=model_parameters["use_dssp"]
            )
        case ModelType.SAGECONVGATMLP:
            classifier = SAGEConvGATMLPModel(
                in_channels=model_parameters["in_channels"],
                feature_channels=model_parameters["features"],
                additional_channels=model_parameters["nr_dssp_features"],
                out_channels=model_parameters["out_channels"],
                dropout=model_parameters["dropout"],
                activation=eval(model_parameters["activation"]),
                heads=4, #TODO: Add heads to config?
                dropout_fcn=model_parameters["droupout_fcn"],
                use_additional_channels=model_parameters["use_dssp"]
            )
    return classifier


def save_classifier_torch(classifier, model_path):
    """Save pre-trained model"""
    torch.save(classifier.state_dict(), model_path)


def load_classifier_torch(model_path):
    """ Load pre-saved model """
    classifier = initialize_model_with_config_params()
    classifier.load_state_dict(torch.load(model_path))
    return classifier


def initialize_untrained_model(model_parameters):
    classifier = initialize_model_with_config_params(model_parameters)
    return classifier
