# gnn-playlist

We will use Neural Graph Collaborative Filtering to learn the embeddings for playlists and songs listed in the Spotify Million Playlist Dataset. I plan to use a [LightGCN](https://arxiv.org/pdf/2002.02126.pdf) architecture to power a Graph Convolution Network for Recommendation. Our goal is to recommend which songs to add to a playlist and so we use recall@k or proportion of relevant songs in top k recommendations as our final performance metric.

```
$ pip3 install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1
$ pip3 install torch-scatter==2.0.9 torch-sparse==0.6.15 torch-cluster==1.6.0 torch-spline-conv==1.2.1 torch-geometric==2.1.0.post1 -f https://data.pyg.org/whl/torch-1.12.0+cpu.html
$ pip3 install snap-stanford
```

# preprocessing

Download the [Spotify Million Playlist Dataset](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge) and place spotify_million_playlist_dataset in the root directory. Run the preprocessing.py script where N represents using the first N thousand playlists in the Spotify Million Playlist Dataset and K is the value used for the [K-core](https://en.wikipedia.org/wiki/Degeneracy_(graph_theory)) graph.

```
# This creates a data_object.pt, graph_info.json, playlist_info.json and song_info.json file in /data.
$ python3 preprocessing.py N K
```
