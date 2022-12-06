import argparse
import json
import random
import numpy as np
import os
import snap
from tqdm import tqdm
import torch
from torch_geometric.data import Data

random.seed(5)
np.random.seed(5)

def parseArgs():
    '''
    returns:
        N: number of files to use for dataset
        K: value for K-core graph
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('N', type=int, help='use first N files for dataset')
    parser.add_argument('K', type=int, help='value for K-core graph')
    args = parser.parse_args()
    return args.N, args.K

def getFiles(N):
    '''
    returns:
        directory and files to use for datset
    '''
    cwd = os.getcwd()
    dir = os.path.join(cwd, 'spotify_million_playlist_dataset/data')
    files = os.listdir(dir)
    files = sorted(files, key=lambda x: int(x.split(".")[2].split("-")[0]))
    return dir, files[:N]

def makeGraph(dir, files):
    '''
    returns:
        graph, number of original playlists and hashmap of PIDs/URIs
    '''
    # create undirected SNAP graph
    G = snap.TUNGraph.New()
    # create node for all playlist IDs
    for file in files:
        with open(os.path.join(dir, file), 'r') as f:
            data = json.load(f)['playlists']
            for playlist in data:
                G.AddNode(playlist['pid'])

    # playlists indexed from [0, num_playlists - 1]
    # songs indexed from [num_playlists, num_playlist + num_songs]
    lastPID = max([n.GetId() for n in G.Nodes()])
    assert lastPID == len([n for n in G.Nodes()]) - 1
    SID = lastPID + 1
    # hashmap of { PID: {'name': _ } } and { URI : {'SID': _ , ... } }
    p_meta, uris = {}, {}

    for file in files:
        with open(os.path.join(dir, file), 'r') as f:
            data = json.load(f)['playlists']
            for playlist in data:
                p_meta[playlist['pid']] = {'name': playlist['name']}
                for song in playlist['tracks']:
                    uri = song['track_uri']
                    if uri not in uris:
                        uris[uri] = {
                            'SID': SID, 
                            'track_name': song['track_name'], 
                            'artist_name': song['artist_name'], 
                            'artist_uri': song['artist_uri'] 
                        }
                        G.AddNode(SID)
                        SID += 1
                    # add edge between playlist and song
                    G.AddEdge(playlist['pid'], uris[uri]['SID'])
    orig_playlists = len([n for n in G.Nodes() if n.GetId() <= lastPID])
    return G, orig_playlists, lastPID, p_meta, uris

def getKCore(G, K, lastPID):
    '''
    returns:
        K-core graph, number of playlists, songs and edges
    '''
    G = G.GetKCore(K)
    if G.Empty():
        raise Exception(f"No graph exists for K={K}")
    num_playlists = len([n for n in G.Nodes() if n.GetId() <= lastPID])
    num_songs = len([n for n in G.Nodes() if n.GetId() > lastPID])
    num_edges = len([x for x in G.Edges()])
    return G, num_playlists, num_songs, num_edges

def reindexGraph(G, orig_playlists, num_playlists, num_songs, p_meta, uris):
    # create hashmap converting old IDs to new IDs
    ID, PIDs, SIDs = 0, {}, {}
    for N in G.Nodes():
        oldID = N.GetId()
        assert oldID not in PIDs and oldID not in SIDs
        if oldID <= orig_playlists - 1:
            PIDs[oldID] = ID
        else:
            SIDs[oldID] = ID
        ID += 1
    assert max(PIDs.values()) == num_playlists - 1
    assert len(PIDs.values()) == num_playlists
    assert max(SIDs.values()) == len([n for n in G.Nodes()]) - 1
    assert len(SIDs.values()) == num_songs

    # hashmap of { SID : {'track_uri': _ , ... } }
    s_meta = {}
    for uri, info in uris.items():
        if info['SID'] in SIDs:
            newID = SIDs[info['SID']]
            s_meta[newID] = {
                'track_uri': uri, 
                'track_name': info['track_name'], 
                'artist_name': info['artist_name'],
                'artist_uri': info['artist_uri']
            }
    p_meta = { PIDs[k]: v for k, v in p_meta.items() if k in PIDs }
    return G, p_meta, s_meta, PIDs, SIDs

def createPyObject(G, PIDs, SIDs):
    # convert to edge_index and storing in a PyG Data object
    edges = []
    for E in tqdm(G.Edges()):
        # create all edges from playlist -> song
        assert (E.GetSrcNId() in PIDs) and (E.GetSrcNId() not in SIDs)
        assert (E.GetDstNId() in SIDs) and (E.GetDstNId() not in PIDs)
        edge = [PIDs[E.GetSrcNId()], SIDs[E.GetDstNId()]]
        edges.append(edge)
        edges.append(edge[::-1])
    edge_idx = torch.LongTensor(edges)
    return Data(edge_index=edge_idx.t().contiguous(), num_nodes=G.GetNodes())

def saveObject(data, p_meta, s_meta, num_playlists, num_songs, num_edges, K, N):

    cwd = os.getcwd()
    dir = os.path.join(cwd, 'data')

    torch.save(data, os.path.join(dir, 'data_object.pt'))
    dataset = {'num_playlists': num_playlists, 'num_nodes': num_playlists + num_songs, 'kcore_value_k': K, 'num_spotify_files_used': N, 'num_edges_directed': 2 * num_edges, 'num_edges_undirected': num_edges}
    with open(os.path.join(dir, 'dataset_stats.json'), 'w') as f:
        json.dump(dataset, f)
    with open(os.path.join(dir, 'playlist_info.json'), 'w') as f:
        json.dump(p_meta, f)
    with open(os.path.join(dir, 'song_info.json'), 'w') as f:
        json.dump(s_meta, f)

if __name__ == "__main__":
    N, K = parseArgs()
    dir, files = getFiles(N)
    G, orig_playlists, lastPID, p_meta, uris = makeGraph(dir, files)
    G, num_playlists, num_songs, num_edges = getKCore(G, K, lastPID)
    G, p_meta, s_meta, PIDs, SIDs  = reindexGraph(G, orig_playlists, num_playlists, num_songs, p_meta, uris)
    data = createPyObject(G, PIDs, SIDs)
    saveObject(data, p_meta, s_meta, num_playlists, num_songs, num_edges, K, N)
