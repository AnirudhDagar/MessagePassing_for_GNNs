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

def read_enzyme_A(datapath):
    with open(datapath) as file_content:
        file_content = [i for i in file_content.readlines()]
    
    edges = []
    for i in file_content:
        i = i.replace(',', '')
        edge_tuple = list(map(int, i.strip().split()))
        edges.append(edge_tuple)
    return edges

def read_enzyme_graph_label(datapath):
    with open(datapath) as file_content:
        file_content = [i for i in file_content.readlines()]
    
    graph_labels = []
    for i in file_content:
        label = int(i.strip())
        graph_labels.append(label)
    return graph_labels

def read_enzyme_node_label(datapath):
    with open(datapath) as file_content:
        file_content = [i for i in file_content.readlines()]
    
    node_labels = []
    for i in file_content:
        label = int(i.strip())
        node_labels.append(label)
    return node_labels

def create_one_hot_feats(node_labels):
    feats = np.zeros((len(node_labels), max(node_labels)), dtype=float)
    for i in range(len(node_labels)):
        feats[i][node_labels[i]-1]=1
    
    return feats

def read_enzyme_graph_indicator(datapath):
    with open(datapath) as file_content:
        file_content = [i for i in file_content.readlines()]
    
    graph_indicator = []
    for i in file_content:
        graph_num = int(i.strip())
        graph_indicator.append(graph_num)
    return graph_indicator

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