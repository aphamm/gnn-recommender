from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree

class LightGCN(MessagePassing):
  """
  a single LightGCN layer
  """
  def __init__(self):
    super(LightGCN, self).__init__(aggr='add')

  def message(self, x_j, norm):
    '''
    args:
      x_j: node embeddings of neighbors of shape [E, emb_dim]
      norm: normalization calculated in forward()
    returns:
      message from neighboring nodes j to central node i
    '''
    return norm.view(-1, 1) * x_j

  def forward(self, x, edge_index):
    """
    args:
      x: current node embeddings of shape [N, emb_dim]
      edge_index: message passing edges of shape [2, E]
    returns:
      updated embeddings after this layer
    """
    row, col = edge_index
    deg = degree(col)
    deg_inv_sqrt = deg.pow(-0.5)
    norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    return self.propagate(edge_index, x=x, norm=norm)
