"""
qaoa_run.py
===========
One full run of QAOA.
"""

import pennylane as qp
from pennylane import numpy as np

import copy
from functools import partial
from matplotlib import pyplot as plt

import graphs as gph
import warm_start as ws
import ansatz as qa
import optimization as opt
import classical as clas


def qaoa_pipeline(problem, strategy, apparatus, silence=False):
    
    # ----------------------  Expand variables  ---------------------- #

    graph = problem["graph"]
    node_list = problem["node_list"]
    edge_list = problem["edge_list"]
    degrees = problem["degrees"]
    N = problem["N"]
    wires = problem["wires"]

    constrained = strategy["constrained"]
    penalizer = strategy["penalizer"]
    relaxation_type = strategy["relaxation_type"]
    eps = strategy["eps"]
    angle_init_type = strategy["angle_init_type"]
    n_restarts = strategy["n_restarts"]
    angles_given = strategy["angles_given"]

    p = apparatus["p"]
    device = apparatus["device"]
    estimator_shots = apparatus["estimator_shots"]
    sampler_shots = apparatus["sampler_shots"]
    optimizer = apparatus["optimizer"]
    opt_steps = apparatus["opt_steps"]
    

    # -----------------  STEP 1:  Build QAOA ansatz  ----------------- #

    # Cost Hamiltonian
    cost_h = qa.cost_hamiltonian(node_list, edge_list, degrees, penalizer)

    # Warm started mixer
    if relaxation_type is None:
        angles = [0.5 * np.pi] * len(wires)
    elif relaxation_type == 'continuous':
        angles = ws.relaxation_angles(graph, wires, eps)
    if not constrained:
        mixer_layer = lambda beta: ws.std_relaxed_mixer_layer(beta, graph, angles)
    else:
        mixer_layer = lambda beta: ws.cst_relaxed_mixer_layer(beta, graph, angles)


    # Device
    dev = qp.device(device, wires=wires)

    cost_qnode = qp.QNode(
        qa.estimator,
        device=dev,
        diff_method="adjoint",  # or "parameter-shift"
        shots=None
    )
    cost_function = partial(
        cost_qnode,
        wires=wires,
        p=p,
        cost_h=cost_h,
        mixer_layer=mixer_layer,
        angles=angles
    )

    sampling_qnode = qp.QNode(
        qa.sampler,
        device=dev,
        shots=sampler_shots
    )
    probability_circuit = partial(
        sampling_qnode,
        wires=wires,
        p=p,
        cost_h=cost_h,
        mixer_layer=mixer_layer,
        angles=angles
    )


    # ----------------  STEP 2:  Optimize parameters  ---------------- #

    if angle_init_type == 'given':
        optimal_params, energies = opt.run_optimization(
                cost_function=cost_function,
                init_params=angles_given, 
                optimizer=optimizer, 
                steps=opt_steps
                )
        best_params = optimal_params
        best_energies = energies

    elif angle_init_type == 'random':
        best_params = None
        best_energies = None
        for i in range(n_restarts):
            if not silence: print(f"Restart {i}")
            optimizer_i = copy.deepcopy(optimizer)
            init_params_i = ws.random_init_param(p)
            optimal_params, energies = opt.run_optimization(
                cost_function=cost_function,
                init_params=init_params_i, 
                optimizer=optimizer_i, 
                steps=opt_steps
                )
            if best_energies is None or energies[-1] < best_energies[-1]:
                best_params = optimal_params
                best_energies = energies
            if not silence: print("----------------------------")

    elif angle_init_type == 'interp':
        if p == 1:
            angles_given = qaoa_pipeline(problem, strategy, apparatus, silence=True) ## MODIFY!!!!
        for p_small in range(1, p-1):
            new_apparatus = apparatus.copy()
            new_apparatus["p"] = p_small
            new_strategy = strategy.copy()
            new_strategy["angle_init_type"] = 'given'
            new_strategy["angles_given"] = angles_given
            p_result = qaoa_pipeline(problem, strategy, apparatus, silence=True)
            if not silence:
                print(f'Energy achieved: {p_result["best_energy"]}')
                print(f'Approximation ratio: {p_result["approximation_ratio"]}')
            angles_given = p_result["best_params"]
        ## Problem initialize the first p-1 but for layer p need initialize

    
    if not silence: print("Optimal Parameters:", best_params)

    if not silence:
        plt.plot(best_energies)
        plt.xlabel("Step")
        plt.ylabel("Energy")
        plt.title("Energy vs. Optimization Step")
        plt.grid(alpha=0.3)
        plt.show()


    # ----------------------  STEP 3: Sampling  ---------------------- #

    probs = probability_circuit(best_params)
    
    if not silence:
        plt.style.use("seaborn-v0_8") 
        plt.bar(range(2 ** len(wires)), probs)
        plt.show()


    # ------------------  STEP 4: Extract solution  ------------------ #

    most_likely_idx = np.argmax(probs)
    most_likely_bin = [(most_likely_idx >> i) & 1 for i in reversed(range(len(wires)))]
    if not silence:
        print("Optimal:", most_likely_bin)
        gph.draw_select(graph, most_likely_bin)
        plt.show()

    best_energy = best_energies[-1]
    best_cost = 0.5 * (len(node_list) - 0.5 * penalizer * len(edge_list) - best_energy)

    theoretical_best_cost = clas.best_cost(graph)
    approximation_ratio = best_cost / theoretical_best_cost


    return {
        "cost_function": cost_function,
        "cost_hamiltonian": cost_h,
        "best_params": best_params,
        "best_energy": best_energy,
        "approximation_ratio": approximation_ratio,
    }

