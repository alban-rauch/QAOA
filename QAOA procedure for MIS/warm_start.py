"""
warm_start.py
=============
Warm starting via relaxation for improved reference state initialization.
Use "Warm-starting quantum optimization" (Egger et al.)
"""

# With scipy.optimize.linprog:
#
# linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method)
#        min c^T x
#   s.t. A_ub x <= b_ub 
#        A_eq x  = b_eq 
#
# Parameters:
## c = (1, ..., 1)^T because we maximize sum_i x_i -> important take -c because linprog minimizes
## Constraint: x_i + x_j <= 1 for every (i, j) in E
## A_ub = matrix with |E| rows where the row associated with (i, j) has entries | 1 if column i or j
##                                                                              | 0 otherwise
## b_ub = (1, ..., 1)^T
## bounds : [0,1]^n
## method : 'highs'


import numpy as np
from scipy.optimize import linprog
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

# Given a graph as in graphs.py

def mis_lp_relaxation(graph):
    nodes = graph.node_indices()
    edges = graph.edge_list()
    n = len(nodes)
    num_edge = len(edges)
    
    c = - np.ones(n)
    b_ub = np.ones(num_edge)
    A_ub = np.zeros((num_edge, n))
    for k, (i, j) in enumerate(edges):
        A_ub[k, i] = 1
        A_ub[k, j] = 1
    
    bounds = [(0, 1)] * n
    
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    x_star = res.x
    
    return dict(zip(nodes, x_star)), -res.fun


# d = dict(nodes, opt)

def lp_relaxation_angles(d, eps=0.25):
    angles = d.copy()
    for node in d:
        x = angles[node]
        if x < eps:
            x = eps
        elif x > 1-eps:
            x = 1-eps
        th = 2 * np.arcsin(np.sqrt(x))
        angles[node] = th
    return angles


def lp_init_from_angles(graph, angles):
    qc = QuantumCircuit(len(angles))
    for i in graph.node_indices():
        qc.ry(angles[i], i)
    return qc


def lp_initialize_state(graph, eps=0.25):
    d, lp_value = mis_lp_relaxation(graph)
    angles = lp_relaxation_angles(d, eps)
    initial_state = lp_init_from_angles(graph, angles)
    return initial_state, angles


# H_m = - sin(theta) X - cos(theta) Z

def lp_relaxed_mixer(graph, angles):
    beta = Parameter("β")
    qc = QuantumCircuit(len(angles))
    for i in graph.node_indices():
        theta = angles[i]
        qc.ry(-theta, i)
        qc.rx(-2 * beta, i)
        qc.ry(theta, i)
    return qc