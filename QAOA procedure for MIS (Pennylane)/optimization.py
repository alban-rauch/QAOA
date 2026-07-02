"""
optimization.py
===============
Classical parameter optimizer.
"""

from pennylane import numpy as np


def run_optimization(cost_function, init_params, optimizer, steps=70):
    params = np.array(init_params, requires_grad=True)
    energies = []
    for i in range(steps):
        params, energy = optimizer.step_and_cost(cost_function, params)
        energies.append(energy)
        if i % 10 == 0: print(f"Step {i:3d} | Energy: {energy:.6f}")
    return params, energies