import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn.init import xavier_uniform_ as xavier
from utils import zeros


class GCNConv(nn.Module):
    """
    The graph convolutional operator from the "Semi-supervised
    Classification with Graph Convolutional Networks"
    https://arxiv.org/abs/1609.02907

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        bias (bool, optional): The layer will not learn any bias if `False`.
        (default: `True`)
        normalize (bool, optional): Add Self Loops & Apply symmetric normalization.
        (default: `True`)
    """
    def __init__(self, adj, in_channels, out_channels, normalize=True, use_bias=False, glorot=False):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels

        self.weight = Parameter(torch.rand(self.in_channels, self.out_channels, requires_grad=True, dtype=torch.float32))        
        
        if use_bias:
            self.bias = Parameter(torch.FloatTensor(self.out_channels))
        else:
            self.register_parameter('bias', None)
        
        if glorot:
            # Set parameter initializations to Glorot
            self.reset_parameters()    
        
        if normalize:
            # Add self loop. `A = Adjacency Mat + Identity Mat`
            self.adj   = adj + torch.eye(adj.size(0), dtype=torch.float32)
            
            # Diagonal Node Degree Matrix D
            self.D     = torch.diag(torch.sum(self.adj, 1))
            
            # Normalize the adjacency matrix using D 
            self.D     = self.D.inverse().sqrt()
            self.adj   = torch.mm(torch.mm(self.D, self.adj), self.D)
            
        else:
            self.adj = adj


    def forward(self, x):
        x = torch.mm(x, self.weight)
        out = torch.mm(self.adj, x)

        return out


    def reset_parameters(self):
        xavier(self.weight)
        zeros(self.bias)
        print("Glorot Initialized Weights")


    def __repr__(self):
        return '{} (InChannels:{}, OutChannels:{})'.format(
            self.__class__.__name__, self.in_channels,
            self.out_channels)


# NOTE: Skipping the Dropout layer.
class GCN(nn.Module):
    def __init__(self, A, nfeat, nhid, nout):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(A, nfeat, nhid)
        self.conv2 = GCNConv(A, nhid, nout)
        
    def forward(self, x):
        h1   = F.relu(self.conv1(x)) 
        h2   = F.relu(self.conv2(h1))

        return h2
