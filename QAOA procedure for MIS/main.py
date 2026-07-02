"""
main.py
=======
Main file.
"""

import numpy as np

from qiskit_aer import AerSimulator

from graphs import initialize_graph, random_graph_data
from hamiltonians import cost_hamiltonian
from qaoa import qaoa_run

from warm_start import lp_initialize_state, lp_relaxed_mixer
from variance_landscape import energy_variance_in_sample, full_energy_landscape_shell, plot_variance

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

node_list, edge_list = random_graph_data(8, 0.6)

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

## Standard QAOA
# mixer_op = None
# initial_state = None

## WS-QAOA
initial_state, angles = lp_initialize_state(graph, eps=0.25)
mixer_op = lp_relaxed_mixer(graph, angles)

p = 3

one_run = qaoa_run(graph, cost_op, mixer_op, initial_state, reps=p, n_restarts=20, opt_shots=10000, 
        sample_shots=10000, backend=AerSimulator(), rng=np.random.default_rng(), show_figs=show_figs)
print(f'{one_run}\n')


circuit = one_run["circuit"]
theta0 = one_run["optimal_parameters"]
radii, variances = full_energy_landscape_shell(2*p, circuit, cost_op, theta0, num_samples=10, backend=AerSimulator(), estimator_shots=10000)

# np.savez_compressed('energy_landscape_data.npz', radii=radii, variances=variances)
plot_variance(radii, variances)


# If file from remote
# data = np.load(r'C:\Users\Alban Rauch\Documents\Alban\Polytechnique\STAGE2\QAOA\energy_landscape_shell__p_4.npz')
# radii = data['radii']
# variances = data['variances']
# data.close()
# plot_variance(radii, variances)