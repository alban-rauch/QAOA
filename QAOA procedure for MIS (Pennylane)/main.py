"""
main.py
=======
Main file.

-------------------------------------------------------------------
Variables:
    ∘ Circuit variables:
        ⇢ Size (N) and graph
        ⇢ Layers (p)
        ⇢ Estimator shots
        ⇢ Sampler shots
    ∘ Optimization:
        ⇢ Optimizer type
        ⇢ Optimization steps
    ∘ Variance landscape variables:
        ⇢ Values of r
        ⇢ Sampling points per shell
        ⇢ Estimator shots
-------------------------------------------------------------------
QAOA type:
    ∘ Here:
        ⇢ Standard
            - Penalizer λ
        ⇢ Constrained
        ⇢ Warm start
            - Coeff const parameter ε
            - Relaxation for init state
            - INTERP for angles init
                OR n_restarts if random
    ∘ Other folder:
        ⇢ ma-QAOA
        ⇢ ADAPT-QAOA
        ⇢ own scheme
--------------------------------------------------------------------
Results:
    ① Approximation ratio E(best) / E(optimal)
    ② Barren plateaux: r_max and d
"""

import pennylane as qp

import graphs as gph
import qaoa_run as qr


# ================================================================ #
#                             VARIABLES                            #
# ================================================================ #

# ---------------------- Circuit variables ----------------------- #

N = 16                      # ⇢ Problem size (N)              #!!!#
graph = gph.randomDRegular(N, 2)
# graph = gph.paragon()
node_list = list(graph.nodes)
edge_list = list(graph.edges)
degrees = {i: graph.degree(i) for i in node_list}
wires = range(N)
p = 2                       # ⇢ Layers/depth (p)              #!!!#

device = "lightning.qubit"  # "lightning.qubit" / "lightning.amdgpu" 

estimator_shots = 10000     # ⇢ Estimator shots
sampler_shots = 10000       # ⇢ Sampler shots


optimizer = qp.AdamOptimizer(stepsize=0.03) # or qp.GradientDescentOptimizer()
opt_steps = 300

# ∘ Gradient based (parameter-shift rules or backpropagation):
#     ⇢ Gradient Descent:
#         - Clean, ideal simulators (no noise)
#         - qp.GradientDescentOptimizer()
#     ⇢ Adam Optimizer:
#         - VQAs on simulators with statistical noise
#         - qp.AdamOptimizer(stepsize=0.1)
#     ⇢ L-BFGS-B:
#         - High-precision tuning on simulators with cheap-to-compute exact gradients
#         - State vector simulators like 'default.qubit' (NOT real hardware)
# ∘ Gradient-free:
#     ⇢ COBYLA:
#         - Real quantum hardware / noisy simulators (expensive-to-compute gradients)
#         - Small number of parameters
#     ⇢ Nelder-Mead:
#         - Real quantum hardware / noisy simulators (expensive-to-compute gradients)
#         - Small number of parameters
#     ⇢ SPSA:
#         - Noisy real hardware
#         - Large circuit

# ----------------- Variance lanscape variables ------------------ #


# ================================================================ #
#                             QAOA TYPE                            #
# ================================================================ #

# -------------------------- constrained ------------------------- #

constrained = False             # ⇢ Constrained

if constrained == False:
    penalizer = 1.5             # by default: 1.333
else:
    penalizer = 0.0

# -------------------------- warm  start ------------------------- #

relaxation_type = 'continuous'  # None or 'continuous' or 'rounded'
eps = 0.6                       # if relaxation_type == 'continuous'

angle_init_type = 'random'      # 'given' or 'random' or 'interp'
n_restarts = 20                 # if angle_init_type == 'random'
angles_given = None             # if angle_init_type == 'given'


# ================================================================ #
#                             QAOA RUN                             #
# ================================================================ #

problem_config = {
    "graph": graph,
    "node_list": node_list,
    "edge_list": edge_list,
    "degrees": degrees,
    "N": N,
    "wires": wires,
}

strategy_config = {
    "constrained": constrained,
    "penalizer": penalizer,
    "relaxation_type": relaxation_type,
    "eps": eps, 
    "angle_init_type": angle_init_type,
    "n_restarts": n_restarts,
    "angles_given": angles_given
}

apparatus_config = {
    "p": p,
    "device": device,
    "estimator_shots": estimator_shots,
    "sampler_shots": sampler_shots,
    "optimizer": optimizer,
    "opt_steps": opt_steps,
}


one_qaoa_run = qr.qaoa_pipeline(
    problem=problem_config,
    strategy=strategy_config,
    apparatus=apparatus_config,
)

print(one_qaoa_run)

cost_function = one_qaoa_run["cost_function"]
theta0 = one_qaoa_run["best_params"]