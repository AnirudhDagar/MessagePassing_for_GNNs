import torch
import numpy as np

def read_karate(datapath):
    with open(datapath) as file_content:
        file_content = [i for i in file_content.readlines()]

    # Remove meta info at document starting
    file_content = file_content[2:]
    
    edges = []
    for i in file_content:
        edge_tuple = list(map(int, i.strip().split()))
        edges.append(edge_tuple)
    
    return edges

def create_adjacency(edges, sparse=False):
    max_node=0
    for edge in edges:
        if edge[0] > max_node:
            max_node = edge[0]
        if edge[1] > max_node:
            max_node = edge[1]

    size = max_node
    adj = [[0 for i in range(size)] for j in range(size)]

    # Build bi-directional graph adj matrix
    for edge in edges:
        adj[edge[0]-1][edge[1]-1] = 1
        adj[edge[1]-1][edge[0]-1] = 1

    # Convert numpy array to torch tensor
    adj_tensor = torch.tensor(adj, dtype=torch.float32)

    ## Use Sparse tensors for larger adjacency matrix
    if sparse:
        adj_tensor = adj_tensor.to_sparse()
    
    return adj_tensor

def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0, dtype=torch.float32)