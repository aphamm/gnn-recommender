import torch
import numpy as np
import os
import json
from GNN import GNN
from torch_geometric import seed_everything
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.data import Data
from utils import sampleNeg
from data import SpotData

def train(model, data_mp, loader, opt, num_playlists, num_nodes, device):
    """
    args:
       model: GNN model
       data_mp: message passing edges
       loader: dataloader for eval edges
       opt: the optimizer
       num_playlists: number of total playlists
       num_nodes: number of total nodes
       device: CPU/GPU
    returns:
       epoch's training loss
    """
    loss, samples = 0, 0
    model.train() # set module in training mode
    for batch in loader:
        data_mp, batch = data_mp.to(device), batch.to(device)
        negs = sampleNeg(batch, num_playlists, num_nodes).to(device)
        # forward + backprop + loss + update
        loss = model.loss(data_mp, batch, negs)
        opt.zero_grad()
        loss.backward()
        opt.step()
        # compute metrics
        sample = batch.edge_index.shape[1]
        loss += loss.item() * sample
        samples += sample
    return loss / samples

def test(model, data_mp, loader, k, device, save_dir, epoch):
    """
    args:
       model: GNN model
       data_mp: message passing edges
       loader: dataloader for eval edges
       k: k for recall@k
       device: CPU/GPU
       save_dir: dir to save embeddings
       epoch: number of the current epoch
    returns:
       recall@k for this epoch
    """
    model.eval()
    recalls = {}
    with torch.no_grad():
        data_mp = data_mp.to(device)
        if save_dir is not None:
            embs = gnn.gnn_propagation(data_mp.edge_index)
            torch.save(embs, os.path.join(save_dir, f"epoch_{epoch}.pt"))
        for batch in loader:
            batch = batch.to(device)
            recalls.update(model.evaluation(data_mp, batch, k))
    return np.mean(list(recalls.values()))

if __name__ == "__main__":

    seed_everything(5)
    epochs = 30
    k = 300
    num_layers = 3 # number of LightGCN layers (number of hops)
    batch_size = 2048  # number of playlists in the batch
    emb_dim = 64 # dimension for the playlist/song embeddings
    emb_dir = 'embeddings'  # path to save embeddings
    if not os.path.exists(emb_dir):
        os.makedirs(emb_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = torch.load("data/data_object.pt")
    with open("data/graph_info.json", 'r') as f:
        info = json.load(f)
    num_playlists, num_nodes = info["num_playlists"], info["num_nodes"]

    # train/val/test split (70-15-15)
    transform = RandomLinkSplit(
        is_undirected=True, 
        add_negative_train_samples=False,
        neg_sampling_ratio=0, num_val=0.15, 
        num_test=0.15
    )
    train_split, val_split, test_split = transform(data)

    # create message passing edges for propagation AND evaluation edges
    train_mp = Data(edge_index=train_split.edge_index)
    val_mp = Data(edge_index=val_split.edge_index)  
    test_mp = Data(edge_index=test_split.edge_index)
    train_ev = SpotData('temp', edge_index=train_split.edge_label_index)
    val_ev = SpotData('temp', edge_index=val_split.edge_label_index)
    test_ev = SpotData('temp', edge_index=test_split.edge_label_index)  

    # dataLoaders for the supervision/evaluation edges
    train_loader = DataLoader(train_ev, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ev, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ev, batch_size=batch_size, shuffle=False)

    # initialize GNN model
    gnn = GNN(
        emb_dim=emb_dim, 
        num_nodes=data.num_nodes, 
        num_playlists=num_playlists, 
        num_layers=num_layers
    ).to(device)
    opt = torch.optim.Adam(gnn.parameters(), lr=1e-3)

    # list of [epoch, train loss / val recall@K ]
    losses, vals = [], []
    for e in range(epochs):
        loss = train(
            gnn, train_mp, train_loader, opt, 
            num_playlists,  num_nodes, device
        )
        losses.append((e, loss))
        if e % 5 == 0:
            val = test(gnn, val_mp, val_loader, k, device, emb_dir, e)
            vals.append((e, val))
            print(f"Epoch {e}: train_loss={loss}, val_recall={val}")
        else:
            print(f"Epoch {e}: train_loss={loss}")

    best_val = max(vals, key = lambda x: x[1])
    print(f"Best val recall@k: {best_val[1]} @ epoch {best_val[0]}")

    test_recall = test(gnn, test_mp, test_loader, k, device, None, None)
    print(f"Test recall@k: {test_recall}")
