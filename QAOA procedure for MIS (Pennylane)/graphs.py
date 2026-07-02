"""
graphs.py
=========
Graph generation.
"""

import numpy as np
import networkx as nx

def simple():
    edges = [(0, 1), (1,2)]
    return nx.Graph(edges)

def paragon():
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), 
            (4, 5), (5, 6), (6, 7), (7, 4), 
            (0, 4), (1, 5), (2, 6), (3, 7),
            ]
    graph = nx.Graph(edges)

    positions = nx.spring_layout(graph, seed=1)
    # nx.draw(graph, with_labels=True, pos=positions)
    # plt.show()

    return graph

# Kim and Vu’s paper [2] shows that this algorithm samples in an 
# asymptotically uniform way from the space of random graphs when d = O(n^(1/3 - eps))
def randomDRegular(N, d):    # N nodes with same degree d
    return nx.random_regular_graph(d, N)

# Inefficient
# def randomGilbert(N, q):
#     edges = []
#     graph = nx.Graph()
#     graph.add_nodes_from(range(N))
#     for i in range(N):
#         for j in range(i+1, N):
#             if np.random.uniform(0, 1) < q:
#                 edges.append((i, j))
#     graph.add_edges_from(edges)
#     return graph

def randomGilbert(N, q):
    return nx.fast_gnp_random_graph(N, q)

def draw_select(graph, x):
    colors = ["tab:grey" if int(i) == 0 else "tab:purple" for i in x]
    pos = nx.spring_layout(graph, seed=1)
    nx.draw(graph, with_labels=True, pos=pos, node_color=colors)

def is_legal(graph, x):
    for u, v in graph.edges:
        if int(x[u]) == 1 and int(x[v]) == 1:
            return False
    return True