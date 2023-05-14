import dgl
import torch
import pandas as pd
import numpy as np

def build_graph(weight):
    """
    input weight matrices of graphs, and build the graphs.
    weight: weight of edges, numpy matrix
    """
    # transfor weight to edges and edge features
    (a, b) = weight.shape
    edge = []
    for i in range(a):
        for j in range(b):
            if weight[i,j] > 0:
                edge.append([i, j, weight[i,j]])
    edge = torch.tensor(edge)
    edges_start = edge[:, 0].long()
    edges_end = edge[:, 1].long()

    # build graph
    g = dgl.graph((edges_start,edges_end), num_nodes=a)
    edge_weights = edge[:, 2]
    g.edata['ew'] = edge_weights.reshape(-1, 1).half()
    g = dgl.add_self_loop(g)
    return g

