import sys
import colorama
import pathvalidate

from pathlib import Path
from config import App
from colorama import Fore

modes = ["optimize-architecture", "best-training", "testing"]
RED = "\033[31m"
config = App.config()


def select_model_from_config(device):
    # we need this for color coded print statements
    colorama.init(autoreset=True)

    if not is_valid_config():
        sys.exit("Config is invalid. Please provide a valid config file.")
    print("Config file is valid")
    # TODO: add model selection for pretrained bindNode models
    return None


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

    return True
