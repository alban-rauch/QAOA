"""
hamiltonians.py
===============
Initialize the cost and mixer hamiltonians.
"""

from qiskit.quantum_info import SparsePauliOp
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

from graphs import *


### Cost Hamiltonian ###

def build_cost_paulis(graph, penalizer):
    pauli_list = []
    for node in graph.node_indices():
        w = weight(node, graph) - 0.5 * penalizer * degree(node, graph)
        pauli_list.append(("Z", [node], w))
    for edge in graph.edge_list():
        pauli_list.append(("ZZ", [edge[0], edge[1]], 0.5 * penalizer))
    return pauli_list


def cost_hamiltonian(graph, penalizer):
    return SparsePauliOp.from_sparse_list(build_cost_paulis(graph, penalizer), graph.num_nodes())


### Mixer Hamiltonian ###

# def mixer(graph):
#     beta = Parameter("β")
    
#     qc = QuantumCircuit(graph.num_nodes())
    
#     for i in graph.node_indices():
#         neighbors = neighbors(i, graph)
#         if not neighbors:
#             qc.rx(2*beta, i)
#         else:
#             for q in neighbors:
#                 qc.x(q)
#             qc.mcrx(2*beta, neighbors, i)
#             for q in neighbors:
#                 qc.x(q)
    
#     return qc

