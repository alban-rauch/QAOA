"""
graph.py
=======
Define the graphs.
"""

import matplotlib.pyplot as plt

import rustworkx as rx
from rustworkx.visualization import mpl_draw as draw_graph


def weight(node_idx, graph):
    return graph[node_idx]["weight"]


def neighbors(node_idx, graph):
    return list(graph.neighbors(node_idx))


def degree(node_idx, graph):
    return graph.degree(node_idx)


def initialize_graph(node_list, edge_list):
    graph = rx.PyGraph()
    
    for node, weight in node_list.items():
        graph.add_node({"label": node, "weight": weight})
    
    graph.add_edges_from(edge_list)
    
    return graph


def mis_is_legal(bits, graph: rx.PyGraph):
    for u, v in graph.edge_list():
        if int(bits[u]) == 1 and int(bits[v]) == 1:
            return False
    return True


def random_graph_data(num_nodes, probability=0.25):
        random_g = rx.undirected_gnp_random_graph(num_nodes, probability)
        rand_node_list = {node: 1.0 for node in random_g.node_indices()}
        rand_edge_list = []
        for u, v in random_g.edge_list():
            rand_edge_list.append((u, v, 1.0))
        return rand_node_list, rand_edge_list


def draw(graph):
    draw_graph(graph, node_size=600, with_labels=True)

    
def draw_select(graph, x):
    colors = ["tab:grey" if int(i) == 0 else "tab:purple" for i in x]
    pos, _default_axes = rx.spring_layout(graph), plt.axes(frameon=True)
    rx.visualization.mpl_draw(
        graph, node_color=colors, node_size=100, alpha=0.8, pos=pos
    )