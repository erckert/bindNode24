import sys
import colorama
import pathvalidate

from pathlib import Path
from config import App
from colorama import Fore
from misc.enums import Mode, ModelType

modes = ["optimize-architecture", "best-training", "testing"]
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
        case "best-training":
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


def get_weight_dir():
    model_section = config["MODEL"]
    return Path(model_section.get('weight_folder'))


def get_in_channels():
    model_section = config["MODEL"]
    return model_section.getint('in_channels')


def get_out_channels():
    model_section = config["MODEL"]
    return model_section.getint('out_channels')


def get_feature_channels(only_first_value=False):
    training_section = config["TRAINING_PARAMETERS"]
    feature_list = [int(item) for item in training_section.get('features').split(',')]
    if not only_first_value:
        return feature_list
    else:
        return feature_list[0]


def get_dropouts(only_first_value=False):
    training_section = config["TRAINING_PARAMETERS"]
    dropout_list = [float(item) for item in training_section.get('dropout').split(',')]
    if not only_first_value:
        return dropout_list
    else:
        return dropout_list[0]