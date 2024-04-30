import sys
import colorama
import pathvalidate
import os
import re

from pathlib import Path
from config import App
from colorama import Fore
import torch.nn.functional as F
from misc.enums import Mode, ModelType, LabelType

modes = ["optimize-architecture", "training", "predict"]
RED = "\033[31m"
config = App.config()


def validate_config():
    # we need this for color coded print statements
    colorama.init(autoreset=True)

    if not is_valid_config():
        sys.exit("Config is invalid. Please provide a valid config file.")
    print("Config file is valid")


def select_mode_from_config():
    general_section = config["DEFAULT"]
    mode = general_section["mode"].lower()
    match mode:
        case "predict":
            return Mode.PREDICT
        case "training":
            return Mode.TRAIN
        case "optimize-architecture":
            return Mode.OPTIMIZE


def select_model_type_from_config():
    # we need this for color coded print statements
    colorama.init(autoreset=True)

    model_section = config["MODEL"]
    model = model_section["model_type"].lower()
    match model:
        case "gcnconv":
            return ModelType.GCNCONV
        case "sageconv":
            return ModelType.SAGECONV
        case "sageconvmlp":
            return ModelType.SAGECONVMLP
        case "sageconvgatmlp":
            return ModelType.SAGECONVGATMLP


def is_valid_config():
    # No sections => File doesn't exist or is missing sections
    if len(config.sections()) == 0:
        print("Config file not found or empty")
        return False

    # Return False if any section in general is not filled out with a valid option
    general_section = config["DEFAULT"]
    if general_section["mode"].lower() not in map(lambda x: x.lower(), modes):
        print(f"Config file does not specify a valid mode. Please select one of the following: "
              f"{modes}")
        return False

    paths_section = config["FILE_PATHS"]
    # check if result_dir path is a valid file path
    if not pathvalidate.is_valid_filepath(paths_section["result_dir"], platform="auto"):
        print(Fore.RED + f"WARNING: File {paths_section.get('result_dir')} is not a valid path")
        return False
    else:
        path = Path(paths_section["result_dir"])
        # check if result_dir already exists. If yes, warn user that previous results could be overwritten, else,
        # create folder
        if path.exists():
            print(Fore.RED + f"WARNING: Path {paths_section.get('result_dir')} already exists. Previous results may "
                             f"be overwriten")
        else:
            Path(path).mkdir(parents=True, exist_ok=True)
    # check if embeddings path is a valid file path
    if not pathvalidate.is_valid_filepath(paths_section["embeddings"], platform="auto"):
        print(Fore.RED + f"WARNING: File {paths_section.get('embeddings')} is not a valid path")
        return False
    # check if embeddings path exists
    if not Path(paths_section["embeddings"]).exists():
        print(Fore.RED + f"WARNING: File {paths_section.get('embeddings')} does not exist")
        return False
    # check if structure path is a valid file path
    if not pathvalidate.is_valid_filepath(paths_section["3d_structure_dir"], platform="auto"):
        print(Fore.RED + f"WARNING: File {paths_section.get('3d_structure_dir')} is not a valid path")
        return False
    # check if structure path exists
    if not Path(paths_section["3d_structure_dir"]).exists():
        print(Fore.RED + f"WARNING: File {paths_section.get('3d_structure_dir')} does not exist")
        return False

    model_section = config["MODEL"]
    # check if cutoff is a number smaller 1
    if not model_section.getfloat('cutoff') < 1.0:
        print(Fore.RED + f"WARNING: Cutoff {model_section.getfloat('cutoff')} is not a valid cutoff. "
                         f"Set cutoff to a value between 0 and 1")
        return False

    return True


def get_id_list_path():
    paths_section = config["FILE_PATHS"]
    return Path(paths_section.get('id_list'))


def get_embeddings_path():
    paths_section = config["FILE_PATHS"]
    return Path(paths_section.get('embeddings'))


def get_sequence_path():
    paths_section = config["FILE_PATHS"]
    return Path(paths_section.get('fasta_file_path'))


def get_3d_structure_dir():
    paths_section = config["FILE_PATHS"]
    return Path(paths_section.get('3d_structure_dir'))


def get_DSSP_dir():
    paths_section = config["FILE_PATHS"]
    return Path(paths_section.get('precomputed_DSSP_feature_dir'))


def get_result_dir():
    paths_section = config["FILE_PATHS"]
    return Path(paths_section.get('result_dir'))


def get_cache_dir():
    paths_section = config["FILE_PATHS"]
    return Path(paths_section.get('cache_folder'))


def get_cv_splits():
    cv_split_ids = []
    paths_section = config["FILE_PATHS"]
    cv_split_folder = Path(paths_section.get('cv_split_id_files_folder'))
    cv_split_files = os.listdir(cv_split_folder)

    for file in cv_split_files:
        cv_split = []
        with open(os.path.join(cv_split_folder, file), 'r') as fh:
            lines = fh.readlines()
            for line in lines:
                protein_id = line.strip()
                if protein_id != "":
                    cv_split.append(protein_id)
            cv_split_ids.append(cv_split)

    return cv_split_ids


def get_label_path(label_type):
    label_section = config["LABELS"]
    match label_type:
        case LabelType.METAL:
            return Path(label_section.get('metal_label_file'))
        case LabelType.SMALL:
            return Path(label_section.get('small_label_file'))
        case LabelType.NUCLEAR:
            return Path(label_section.get('nuclear_label_file'))


def do_logging():
    utility_section = config["UTILITY"]
    return utility_section.getboolean("do_logging")


def use_cache():
    utility_section = config["UTILITY"]
    return utility_section.getboolean("use_cache")


def get_weight_dir():
    model_section = config["MODEL"]
    return Path(model_section.get('weight_folder'))


def get_in_channels():
    model_section = config["MODEL"]
    return model_section.getint('in_channels')


def get_out_channels():
    model_section = config["MODEL"]
    return model_section.getint('out_channels')


def get_cutoff():
    model_section = config["MODEL"]
    return model_section.getfloat('cutoff')


def get_output_file_name():
    output_section = config["OUTPUT"]
    return output_section.get('output_file_name')


def write_ri():
    output_section = config["OUTPUT"]
    return output_section.getboolean('write_ri')


def get_feature_channels(only_first_value=False):
    training_section = config["MODEL_PARAMETERS"]
    feature_list = [int(item) for item in training_section.get('features').split(',')]
    if not only_first_value:
        return feature_list
    else:
        return feature_list[0]


def get_dropouts(only_first_value=False):
    training_section = config["MODEL_PARAMETERS"]
    dropout_list = [float(item) for item in training_section.get('dropout').split(',')]
    if not only_first_value:
        return dropout_list
    else:
        return dropout_list[0]


def get_batch_size(only_first_value=False):
    training_section = config["MODEL_PARAMETERS"]
    batch_size_list = [int(item) for item in training_section.get('batchsize').split(',')]
    if not only_first_value:
        return batch_size_list
    else:
        return batch_size_list[0]


def get_epochs(only_first_value=False):
    training_section = config["MODEL_PARAMETERS"]
    epochs_list = [int(item) for item in training_section.get('epochs').split(',')]
    if not only_first_value:
        return epochs_list
    else:
        return epochs_list[0]


def is_early_stopping():
    training_section = config["MODEL_PARAMETERS"]
    return training_section.getboolean("early_stopping")


def get_weights(only_first_value=False):
    training_section = config["LOSS_FUNCTION_PARAMETERS"]
    weights_list = [[float(weight) for weight in item.split(',')] for item in re.findall(r'\[(.*?)\]', training_section.get('weights'))]
    if not only_first_value:
        return weights_list
    else:
        return weights_list[0]


def get_structure_cutoff(only_first_value=False):
    training_section = config["DATASET"]
    structure_cutoff_list = [int(item) for item in training_section.get('cutoff_structure').split(',')]
    if not only_first_value:
        return structure_cutoff_list
    else:
        return structure_cutoff_list[0]


def get_activation():
    training_section = config["MODEL_PARAMETERS"]
    return eval(training_section.get("activation"))


def get_activation_as_string():
    training_section = config["MODEL_PARAMETERS"]
    return training_section.get("activation")


def get_dropouts_fcn(only_first_value=False):
    training_section = config["MODEL_PARAMETERS"]
    dropout_fcn_list = [float(item) for item in training_section.get('dropout_fcn').split(',')]
    if not only_first_value:
        return dropout_fcn_list
    else:
        return dropout_fcn_list[0]


def get_optimizer_arguments(only_first_value=False):
    optimizer_section = config["OPTIMIZER_PARAMETERS"]
    learning_rates = [float(item) for item in optimizer_section.get('learning_rate').split(',')]
    betas = [[float(beta) for beta in item.split(',')] for item in re.findall(r'\[(.*?)\]', optimizer_section.get('betas'))]
    epsilons = [float(item) for item in optimizer_section.get('epsilon').split(',')]
    weight_decays = [float(item) for item in optimizer_section.get('weight_decay').split(',')]
    if not only_first_value:
        optimizer_arguments = {
            "lr": learning_rates,
            "betas": betas,
            "eps": epsilons,
            "weight_decay": weight_decays
        }
    else:
        optimizer_arguments = {
            "lr": learning_rates[0],
            "betas": betas[0],
            "eps": epsilons[0],
            "weight_decay": weight_decays[0]
        }
    return optimizer_arguments


def get_model_parameter_dict(only_first_value=False):
    model_parameter_dict = {
        "cutoff": get_cutoff(),
        "in_channels": get_in_channels(),
        "out_channels": get_out_channels(),
        "features": get_feature_channels(only_first_value),
        "dropout": get_dropouts(only_first_value),
        "epochs": get_epochs(only_first_value),
        "early_stopping": is_early_stopping(),
        "weights": get_weights(only_first_value),
        "cutoff_structure": get_structure_cutoff(only_first_value),
        "activation": get_activation_as_string(),
        "batchsize": get_batch_size(only_first_value),
        "droupout_fcn": get_dropouts_fcn(only_first_value)
    }
    return model_parameter_dict

