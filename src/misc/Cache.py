import os
import pickle
from pathlib import Path

from setup.configProcessor import get_cache_dir, do_logging


def setup_chache():
    cache_directory = get_cache_dir()
    if not Path(cache_directory).exists():
        Path(cache_directory).mkdir(parents=True, exist_ok=True)

    distance_matrix_directory_path = os.path.join(cache_directory, "distance_matrices")
    Path(distance_matrix_directory_path).mkdir(parents=True, exist_ok=True)


def cache_distance_matrix(protein_id, distance_matrix):
    distance_matrix_directory_path = os.path.join(get_cache_dir(), "distance_matrices")
    distance_matrix_file_path = os.path.join(distance_matrix_directory_path, f"{protein_id}.p")
    pickle.dump(distance_matrix, open(distance_matrix_file_path, "wb"))


def load_distance_matrix(protein_id):
    distance_matrix_directory_path = os.path.join(get_cache_dir(), "distance_matrices")
    distance_matrix_file_path = os.path.join(distance_matrix_directory_path, f"{protein_id}.p")
    if Path(distance_matrix_file_path).exists():
        return pickle.load(open( distance_matrix_file_path, "rb"))
    else:
        if do_logging():
            print(f"No cached distance matrix available for {protein_id}. Distance matrix will be computed.")
        return None
