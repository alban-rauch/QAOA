"""
warm_start.py
=============
Warm starting via relaxation for improved reference state initialization.
"""

import pennylane as qp
from pennylane import numpy as np

from scipy.optimize import linprog

# ================================================================ #
#                        INITIAL PARAMETERS                        #
# ================================================================ #

def random_init_param(p):
    rng = np.random.default_rng()
    betas = rng.uniform(0, np.pi, p)
    gammas = rng.uniform(0, 2 * np.pi, p)
    raw_params = np.array([gammas, betas], requires_grad=True)
    return raw_params


# ================================================================ #
#                           INITIAL STATE                          #
# ================================================================ #

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


def relaxation(graph, wires):
    nodes = wires
    edges = list(graph.edges)
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



def relaxation_angles(graph, wires, eps=0.25):
    d, _ = relaxation(graph, wires)
    angles = {}
    for node in d:
        x = d[node]
        if x < eps:
            x = eps
        elif x > 1-eps:
            x = 1-eps
        th = 2 * np.arcsin(np.sqrt(x))
        angles[node] = th
    return angles


# H_m = - sin(theta) X - cos(theta) Z

# def relaxed_mixer(graph, angles):
#     coeffs = []
#     ops = []
#     for node in graph.nodes:
#         theta = angles[node]
#         coeffs.append(-np.sin(theta))
#         ops.append(qp.PauliX(node))
#         coeffs.append(-np.cos(theta))
#         ops.append(qp.PauliZ(node))
#     return qp.Hamiltonian(coeffs, ops)



## exp(- i β H_M) = exp(i β (sin(θ) X + cos(θ) Z))
##                = cos(β) I + i sin(β) (cos(θ) Z + sin(θ) X)
##                = RY(θ) RZ(-2β) RY(-θ)

def std_relaxed_mixer_layer(beta, graph, angles):
    for node in graph.nodes:
        theta = angles[node]
        qp.RY(-theta, wires=node)
        qp.RZ(-2 * beta, wires=node)
        qp.RY(theta, wires=node)

def cst_relaxed_mixer_layer(beta, graph, angles):
    for node in graph.nodes:
        theta = angles[node]
        neighbors = list(graph.neighbors(node))
        
        if not neighbors:
            qp.RY(-theta, wires=node)
            qp.RZ(-2 * beta, wires=node)
            qp.RY(theta, wires=node)
            continue
        
        qp.ControlledSequence(
            qp.RY(-theta, wires=node),
            control_wires=neighbors
        )
        qp.ControlledSequence(
            qp.RYZ(-2 * beta, wires=node),
            control_wires=neighbors
        )
        qp.ControlledSequence(
            qp.RY(theta, wires=node),
            control_wires=neighbors
        )