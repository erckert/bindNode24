from data_preparation import MyDataset, ProteinResults
from config import GeneralInformation

from bindNode23.dataset import GraphDataset
from torch_geometric.loader import DataLoader
import torch


class MLPredictor(object):

    def __init__(self, model):
        if torch.cuda.is_available():
            self.device = 'cuda:0'
        else:
            self.device = 'cpu'

        self.model = model.to(self.device)

    def predict_per_protein(self, ids, sequences, embeddings, labels, max_length, structures, distance_cutoff_embd, distance_cutoff_struc):

        # validation_set = MyDataset(ids, embeddings, sequences, labels, max_length, structures=structures, protein_prediction=True)
        validation_set = GraphDataset(ids, embeddings, sequences, labels, distance_cutoff_embd=distance_cutoff_embd, distance_cutoff_struc=distance_cutoff_struc, structures=structures, protein_prediction=True)
        validation_loader = DataLoader(validation_set, batch_size=1, shuffle=True)
        sigm = torch.nn.Sigmoid()

        proteins = dict()
        
        self.model.eval()
        for in_graph, prot_id in validation_loader:
            with torch.no_grad():
                prot_id = prot_id[0]
                in_graph = in_graph.to(self.device)

                pred = self.model.forward(
                        in_graph.x, in_graph.edge_index, in_graph.edge_index2, edge_features=in_graph.edge_attr
                    )
                pred = sigm(pred)
                pred = pred.detach().cpu()

                prot = ProteinResults(prot_id)
                prot.set_predictions(pred)
                proteins[prot_id] = prot

        return proteins
