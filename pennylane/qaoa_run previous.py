"""
qaoa_run.py
===========
One full run of QAOA.
"""

from matplotlib import pyplot as plt
import pennylane as qp
import copy
import pennylane as qp

from pennylane import numpy as np
import graphs as gph
import ansatz as qa
import optimization as opt
import warm_start as ws


def qaoa_pipeline(problem, apparatus, constraint, warm_starting):
    
    # ----------------------  Expand variables  ---------------------- #
    graph = problem["graph"]
    node_list = problem["node_list"]
    edge_list = problem["edge_list"]
    degrees = problem["degrees"]
    N = problem["N"]
    wires = problem["wires"]
    p = problem["p"]
    
    device = apparatus["device"]
    estimator_shots = apparatus["estimator_shots"]
    sampler_shots = apparatus["sampler_shots"]
    optimizer = apparatus["optimizer"]
    opt_steps = apparatus["opt_steps"]
    n_restarts = apparatus["n_restarts"]

    penalizer = None
    if constraint is not None:
        penalizer = constraint["penalizer"]
        
    relaxation_type = None
    eps = None
    angle_init_type = None
    if warm_starting is not None:
        relaxation_type = warm_starting["relaxation_type"]
        eps = warm_starting["eps"]
        angle_init_type = warm_starting["angle_init_type"]
    else: angle_init_type = 'random'

    # -----------------  STEP 1:  Build QAOA ansatz  ----------------- #

    # Cost Hamiltonian
    cost_h = qa.std_cost_hamiltonian(node_list, edge_list, degrees, penalizer)

    # Warm started mixer
    if warm_starting:
        angles = ws.relaxation_angles(graph, wires, eps)
    else:
        angles = [0.5 * np.pi] * len(wires)
    mixer_layer = lambda beta: ws.relaxed_mixer_layer(beta, graph, angles)

    # Parameter initialization (now in the restarts)
    # init_params = ws.init_param(p, angle_init_type)


    # Device
    dev = qp.device(device, wires=wires)

    # diff_method="adjoint", "backprop" (for shots=None) / "parameter-shift"
    # @qp.qnode(dev, diff_method="parameter-shift", shots=estimator_shots)
    @qp.qnode(dev, diff_method="adjoint", shots=None)
    # @qp.qnode(dev, shots=estimator_shots)
    def cost_function(params):
        qa.circuit(wires, p, params, cost_h, mixer_layer, angles)
        return qp.expval(cost_h)

    @qp.qnode(dev, shots=sampler_shots)
    def probability_circuit(params):
        qa.circuit(wires, p, params, cost_h, mixer_layer, angles)
        return qp.probs(wires=wires)


    # ----------------  STEP 2:  Optimize parameters  ---------------- #

    best_params = None
    best_energies = None
    for i in range(n_restarts):
        print(f"Restart {i}")
        optimizer_i = copy.deepcopy(optimizer)
        init_params_i = ws.init_param(p, angle_init_type)
        optimal_params, energies = opt.run_optimization(cost_function, optimizer_i, init_params_i, steps=opt_steps)
        if best_energies is None or energies[-1] < best_energies[-1]:
            best_params = optimal_params
            best_energies = energies
        print("----------------------------")
    print("Optimal Parameters:", best_params)

    plt.plot(best_energies)
    plt.xlabel("Step")
    plt.ylabel("Energy")
    plt.title("Energy vs. Optimization Step")
    plt.grid(alpha=0.3)
    plt.show()

    # ----------------------  STEP 3: Sampling  ---------------------- #

    probs = probability_circuit(best_params)

    plt.style.use("seaborn-v0_8") 
    plt.bar(range(2 ** len(wires)), probs)
    plt.show()

    most_likely_idx = np.argmax(probs)
    most_likely_bin = [(most_likely_idx >> i) & 1 for i in reversed(range(len(wires)))]
    print("Optimal:", most_likely_bin)
    gph.draw_select(graph, most_likely_bin)
    plt.show()

    return {
        "cost_function": cost_function,
        "cost_hamiltonian": cost_h,
        "best_params": best_params,
        "best_energy": best_energies[-1],
    }

