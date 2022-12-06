import torch
import torch.nn as nn
from lightGCN import LightGCN
from utils import recall

class GNN(nn.Module):
    """
    full GNN with learnable playlist/song embeddings and LightGCN layers
    """
    def __init__(self, emb_dim, num_nodes, num_playlists, num_layers):
        super(GNN, self).__init__()
        self.emb_dim = emb_dim
        self.num_nodes = num_nodes
        self.num_playlists = num_playlists
        self.num_layers = num_layers
        self.embeddings = torch.nn.Embedding(
          num_embeddings=self.num_nodes, 
          embedding_dim=self.emb_dim
        )
        nn.init.normal_(self.embeddings.weight, std=0.1)
        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.layers.append(LightGCN())
        self.sigmoid = torch.sigmoid

    def forward(self):
        raise NotImplementedError("Do not use.")

    def gnn_propagation(self, edge_index_mp):
        """
        func:
          performn linear embedding propagation via lightGCN layers
        args:
          edge_index_mp: a tensor of all message passing edges
        returns:
          final multi-scale embeddings for all playlist/songs
        """
        l = self.embeddings.weight # layer-0 embeddings
        layers = [l]
        # GNN propagation loop
        for i in range(self.num_layers):
            l = self.layers[i](l, edge_index_mp)
            layers.append(l)
        return torch.stack(layers, dim=0).mean(dim=0)

    def predict_scores(self, edge_index, embs):
        """
        func:
          computes predicted scores for each playlist/song pair via dot product
        args:
          edge_index: tensor of playlist/song edges we compute
          embs: node embeddings for calculating predicted scores (typically the multi-scale embeddings from gnn_propagation())
        returns:
          predicted scores for each playlist/song pair in edge_index
        """
        # dot product for each playlist/song pair
        scores = embs[edge_index[0,:], :] * embs[edge_index[1,:], :]
        return self.sigmoid(scores.sum(dim=1))

    def loss(self, data_mp, data_pos, data_neg):
        """
        func:
          training set. GNN propagation on message passing edges. predict scores on training examples. calculate Bayesian Personalized Ranking.
        args:
          data_mp: tensor of message passing edges
          data_pos: set of positive edges for loss calculation
          data_neg: set of negative edges for loss calculation
        returns:
          loss calculated on the positive/negative training edges
        """
        final_embs = self.gnn_propagation(data_mp.edge_index)
        # get edge prediction scores for all positive/negative evaluation edges
        pos_scores = self.predict_scores(data_pos.edge_index, final_embs)
        neg_scores = self.predict_scores(data_neg.edge_index, final_embs)
        loss = -torch.log(self.sigmoid(pos_scores - neg_scores)).mean()
        return loss

    def evaluation(self, data_mp, data_pos, k):
        """
        func:
          calculate recall@k on validation/test set 
        args:
          data_mp: tensor of message passing edges
          data_pos: set of positive edges for scoring metric
          k: k for recall@k
        returns:
          hashmap of { PID: recall@k } 
        """
        final_embs = self.gnn_propagation(data_mp.edge_index)
        # embeddings of all unique playlists in the batch of evaluation edges
        unique_playlists = torch.unique_consecutive(data_pos.edge_index[0,:])
        # shape [ playlists_in_batch, EMB_DIM ]
        playlist_emb = final_embs[unique_playlists, :]
        # shape [ songs_in_dataset, EMB_DIM ]
        song_emb = final_embs[self.num_playlists:, :]
        # entry i,j is rating of song j for playlist i
        ratings = self.sigmoid(torch.matmul(playlist_emb, song_emb.t()))
        result = recall(ratings.cpu(), k, self.num_playlists, data_pos.edge_index.cpu(), unique_playlists.cpu(), data_mp.edge_index.cpu())
        return result
