"""
main.py
=======
Main file.
"""

import numpy as np

from qiskit_aer import AerSimulator

from graphs import initialize_graph
from hamiltonians import cost_hamiltonian
from qaoa import qaoa_run

node_list = {
    0: 1.0,
    1: 1.0,
    2: 1.0,
    3: 1.0,
    4: 1.0,
    5: 1.0,
    6: 1.0,
    7: 1.0
}


edge_list = [
    (0, 1, 1.0),
    (1, 2, 1.0),
    (2, 3, 1.0),
    (3, 0, 1.0),
    (4, 5, 1.0),
    (5, 6, 1.0),
    (6, 7, 1.0),
    (7, 4, 1.0),
    (0, 4, 1.0),
    (1, 5, 1.0),
    (2, 6, 1.0),
    (3, 7, 1.0),
]

graph = initialize_graph(node_list, edge_list)

eps = 0.5
penalizer = np.max(list(node_list.values())) + eps

show_figs = {
    "init_circ": False,
    "transp_circ": False,
    "energy_evo": True,
    "opt_circ": False,
    "config_dist": True,
    "optimal_graph": True
}


cost_op = cost_hamiltonian(graph, penalizer)
mixer_op = None

one_run = qaoa_run(graph, cost_op, mixer_op, reps=3, n_restarts=10, opt_shots=10000, 
        sample_shots=10000, backend=AerSimulator(), rng=np.random.default_rng(), show_figs=show_figs)
print(f'{one_run}\n')