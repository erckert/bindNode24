from setup.configProcessor import get_id_list_path, get_embeddings_path, get_sequence_path

import numpy as np
import h5py
import fastapy


class BindingResidueDataset:
    def __init__(self):
        self.protein_ids = self.get_id_list()
        self.embeddings = self.get_embeddings(self.protein_ids)
        self.sequences = self.get_sequences(self.protein_ids)
        self.structures = {}

    def __len__(self):
        return len(self.protein_ids)

    def __getitem__(self, item):
        protein_id = self.protein_ids[item]
        return self.embeddings[protein_id], self.sequences[protein_id], self.structures[protein_id]

    @staticmethod
    def get_id_list():
        id_list = []
        with open(get_id_list_path(), 'r') as fh:
            lines = fh.readlines()
            for line in lines:
                protein_id = line.strip()
                if protein_id != "":
                    id_list.append(protein_id)
        return id_list

    @staticmethod
    def get_embeddings(protein_ids):
        embeddings = {}
        embedding_h5 = h5py.File(get_embeddings_path(), 'r')
        for key, value in embedding_h5.items():
            if key in protein_ids:
                embeddings[key] = np.array(value[:, :], dtype=np.float32)
        return embeddings

    @staticmethod
    def get_structures(protein_ids):
        protein_structures = {}
        # TODO
        return protein_structures

    @staticmethod
    def get_sequences(protein_ids):
        protein_sequences = {}
        for record in fastapy.parse(get_sequence_path()):
            if record.id in protein_ids:
                protein_sequences[record.id] = record.seq
        return protein_sequences
