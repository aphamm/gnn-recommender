import torch
from torch_geometric.data import Data, Dataset

class CustomData(Data):
    """
    override __inc__ so DataLoader doesn't increment indices
    """
    def __inc__(self, key, value, *args, **kwargs):
        return 0

class SpotData(Dataset):
    """
    dataset with supervision/evaluation edges.
    get(idx) return ALL outgoing edges of the graph of playlist "idx" since calculating metrics like recall@k needs all the playlist's positive edges
    """
    def __init__(self, root, edge_index, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.edge_index = edge_index
        # playlists will all be in row 0, b/c sorted by RandLinkSplit
        self.unique_idxs = torch.unique(edge_index[0,:]).tolist() 
        self.num_nodes = len(self.unique_idxs)

    def len(self):
        return self.num_nodes

    # returns all outgoing edges associated with playlist idx
    def get(self, idx):
        edge_index = self.edge_index[:, self.edge_index[0,:] == idx]
        return CustomData(edge_index=edge_index)
        