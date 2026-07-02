"""
ansatz.py
=========
Build the QAOA circuit.
"""

import pennylane as qp
from pennylane import qaoa


def cost_hamiltonian(node_list, edge_list, degrees, penalizer):
    coeffs = []
    ops = []
    for i in node_list:
        coeffs.append((1.0 - penalizer * degrees[i] / 2.0))
        ops.append(qp.PauliZ(i))
    coupling_coeff = penalizer / 2.0
    for i, j in edge_list:
        coeffs.append(coupling_coeff)
        ops.append(qp.PauliZ(i) @ qp.PauliZ(j))
    return qp.Hamiltonian(coeffs, ops)

def cst_cost_hamiltonian(node_list, edge_list, degrees, penalizer):
    coeffs = []
    ops = []
    for i in node_list:
        coeffs.append((1.0 - penalizer * degrees[i] / 2.0))
        ops.append(qp.PauliZ(i))
    coupling_coeff = penalizer / 2.0
    for i, j in edge_list:
        coeffs.append(coupling_coeff)
        ops.append(qp.PauliZ(i) @ qp.PauliZ(j))
    return qp.Hamiltonian(coeffs, ops)

### Direct evolution (not needed as no accuracy/performance difference)
# def unconstrained_cost_layer(gamma, graph, penalizer):
#     linear_coeff = 2.0 * gamma * (1.0 - penalizer / 2.0)
#     for node in graph.nodes:
#         qp.RZ(-linear_coeff, wires=node)
#     coupling_coeff = 2.0 * gamma * penalizer / 2.0
#     for u, v in graph.edges:
#         qp.CNOT(wires=[u, v])
#         qp.RZ(-coupling_coeff, wires=v)
#         qp.CNOT(wires=[u, v])

### Native (slow)
# def hamiltonians(graph, constrained=False):
#     cost_h, mixer_h = qaoa.max_independent_set(graph, constrained=constrained)
#     print("Cost Hamiltonian:", cost_h)
#     print("Mixer Hamiltonian:", mixer_h)
#     return cost_h, mixer_h

# def std_hamiltonians(graph, penalizer, angles):
#     cost_h = std_cost_hamiltonian(graph, penalizer)
#     mixer_h = ws.relaxed_mixer(graph, angles)
#     return cost_h, mixer_h

def qaoa_layer(gamma, beta, cost_h, mixer_layer):
    qaoa.cost_layer(gamma, cost_h)
    mixer_layer(beta)

def circuit(wires, p, params, cost_h, mixer_layer, angles):
    for w in wires:
        qp.RY(angles[w], wires=w)
    qp.layer(qaoa_layer, p, params[0], params[1], cost_h=cost_h, mixer_layer=mixer_layer)


# print(qp.draw(circuit)([[0.3, 0.4], [0.5, 0.1]]))
# qp.draw_mpl(circuit, layer='device')([[0.3, 0.4], [0.5, 0.1]])
# plt.show()

def estimator(params, wires, p, cost_h, mixer_layer, angles):
    circuit(wires, p, params, cost_h, mixer_layer, angles)
    return qp.expval(cost_h)

def sampler(params, wires, p, cost_h, mixer_layer, angles):
    circuit(wires, p, params, cost_h, mixer_layer, angles)
    return qp.probs(wires=wires)