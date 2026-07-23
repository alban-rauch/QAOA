"""
main.py
=======
Main file.
"""

import pennylane as qp
import numpy as np

import graphs as gph
import qaoa_run as qr
import variance_lanscape as vl


# ================================================================ #
#                             VARIABLES                            #
# ================================================================ #

# ---------------------- Circuit variables ----------------------- #

N = 8                       # ⇢ Problem size (N)              #!!!#
graph = gph.randomGilbert(N, 0.25)
# graph = gph.randomDRegular(N, 3)
# graph = gph.paragon()
p = 5                       # ⇢ Layers/depth (p)              #!!!#

device = "lightning.qubit"  # "lightning.qubit" / "lightning.amdgpu" 

estimator_shots = 10000     # ⇢ Estimator shots
sampler_shots = 10000       # ⇢ Sampler shots


optimizer = "L-BFGS-B"      # or "Adam"
opt_steps = 400


# ================================================================ #
#                             QAOA TYPE                            #
# ================================================================ #

# -------------------------- constrained ------------------------- #

constrained = False

# -------------------------- warm  start ------------------------- #

relaxation_type = None              # None / 'continuous'
param_transfer_type = 'interp'     # 'given' / 'random' /  
                                    # 'interp' / 'fourier'
(fourier_q, fourier_R) = (None, 5)


# ================================================================ #
#                             QAOA RUN                             #
# ================================================================ #

problem_config = {
    "N": N,
    "graph": graph,
}

strategy_config = {
    "constrained": constrained,
    "relaxation_type": relaxation_type,
    "param_transfer_type": param_transfer_type,
    "fourier_qR": (fourier_q, fourier_R)
}

apparatus_config = {
    "p": p,
    "device": device,
    "estimator_shots": estimator_shots,
    "sampler_shots": sampler_shots,
    "optimizer": optimizer,
    "opt_steps": opt_steps,
}

strategy_config["relaxation_type"] = 'continuous'

one_qaoa_run = qr.standard_qaoa(
    problem=problem_config,
    strategy=strategy_config,
    apparatus=apparatus_config,
)

print(one_qaoa_run)

# cost_function = one_qaoa_run["cost_function"]
# theta0 = one_qaoa_run["best_params"]
# radii, variances = vl.full_energy_landscape_shell(cost_function, theta0, 100)
# vl.plot_variance(radii, variances)

