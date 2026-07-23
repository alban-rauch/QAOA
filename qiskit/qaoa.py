"""
qaoa.py
=======
One qaoa run.
"""

import time
import matplotlib.pyplot as plt

import numpy as np
from scipy.optimize import minimize

from graphs import weight, mis_is_legal, draw_select

from qiskit import QuantumCircuit
from qiskit.circuit.library import QAOAAnsatz
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_aer import AerSimulator


def show_circuit(circuit):
    circuit.draw("mpl", fold=False, idle_wires=False)


### Build circuits ###

def build_circuit(graph, cost_hamiltonian, mixer_hamiltonian, initial_state, reps, backend):

    if initial_state is None: initial_state = QuantumCircuit(graph.num_nodes())
    
    if mixer_hamiltonian is None:
        circuit = QAOAAnsatz(
            cost_operator=cost_hamiltonian, 
            initial_state=initial_state,
            reps=reps
        )
    else:
        circuit = QAOAAnsatz(
            cost_operator=cost_hamiltonian, 
            mixer_operator=mixer_hamiltonian, 
            initial_state=initial_state,
            reps=reps
        )

        
    circuit.measure_all()

    pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
    candidate_circuit = pm.run(circuit)
    
    return circuit, candidate_circuit


### Optimize parameters ###

def optimize_parameters(candidate_circuit, cost_hamiltonian, reps, n_restarts, opt_shots, 
                        backend, rng, plot_energy=False):
    
    estimator = Estimator(mode=backend)
    estimator.options.default_shots = opt_shots
    
    objective_func_vals = [] if plot_energy else None
    
    def cost_func_estimator(params, ansatz, hamiltonian, estimator):
        if ansatz.layout is not None:
            isa_hamiltonian = hamiltonian.apply_layout(ansatz.layout)
        else:
            isa_hamiltonian = hamiltonian

        job = estimator.run([(ansatz, isa_hamiltonian, params)])

        cost = job.result()[0].data.evs
        
        if objective_func_vals is not None: objective_func_vals.append(cost)
        
        return cost
    
        
    lower_bound = np.zeros(2 * reps)
    upper_bound = np.concatenate([np.full(reps, np.pi), np.full(reps, 2 * np.pi)])
    
    best_result = None

    for _ in range(n_restarts):
        x0 = rng.uniform(low=lower_bound, high=upper_bound)
        result = minimize(
            cost_func_estimator,
            x0,
            args=(candidate_circuit, cost_hamiltonian, estimator),
            method="COBYLA",
            tol=1e-3,
        )
        if best_result is None or result.fun < best_result.fun:
            best_result = result
    
            
    optimized_circuit = candidate_circuit.assign_parameters(best_result.x)
        
    return best_result, optimized_circuit, objective_func_vals
    
    

### Sample configuration distribution ###

def sample_distribution(optimized_circuit, sample_shots, backend):
    
    sampler = Sampler(mode=backend)
    sampler.options.default_shots = sample_shots
    job = sampler.run([(optimized_circuit,)])
    counts_int = job.result()[0].data.meas.get_int_counts()
    counts_bin = job.result()[0].data.meas.get_counts()
    
    most_likely_int = max(counts_int, key=counts_int.get)
    most_likely_bitstring = [int(b) for b in np.binary_repr(most_likely_int, width=optimized_circuit.num_qubits)]
    most_likely_bitstring.reverse()
        
    return counts_bin, most_likely_bitstring


### Full QAOA run ###

def qaoa_run(graph, cost_hamiltonian, mixer_hamiltonian=None, initial_state=None, reps=2, n_restarts=1, opt_shots=10000, 
                sample_shots=10000, backend=AerSimulator(), rng=np.random.default_rng(), show_figs=None):
    
    if show_figs == None:
        show_figs = {
            "init_circ": False,
            "transp_circ": False,
            "energy_evo": False,
            "opt_circ": False,
            "config_dist": False,
            "optimal_graph": False
        }
    
    t0 = time.perf_counter()
    circuit, candidate_circuit = build_circuit(graph, cost_hamiltonian, mixer_hamiltonian, initial_state, 
                                                        reps=reps, backend=backend)
    t1 = time.perf_counter()
    t_build = t1 - t0
    
    if show_figs["init_circ"]:
        fig = show_circuit(circuit)
        plt.show()
            
    if show_figs["transp_circ"]:
        fig = show_circuit(candidate_circuit)
        plt.show()
    
    t2 = time.perf_counter()
    best_result, optimized_circuit, objective_func_vals = optimize_parameters(candidate_circuit, cost_hamiltonian, reps, n_restarts=n_restarts, 
                                                                opt_shots=opt_shots, backend=backend, rng=rng, plot_energy=show_figs["energy_evo"])
    t3 = time.perf_counter()
    t_opt = t3 - t2
    
    if show_figs["energy_evo"]:
        plt.figure(figsize=(12, 6))
        plt.plot(objective_func_vals)
        plt.xlabel("Iteration")
        plt.ylabel("Cost")
        plt.show()
    
    if show_figs["opt_circ"]:
        fig = show_circuit(optimized_circuit)
        plt.show()
    
    t4 = time.perf_counter()
    counts_bin, most_likely_bitstring = sample_distribution(optimized_circuit, sample_shots=sample_shots, backend=backend)
    t5 = time.perf_counter()
    t_sample = t5 - t4
    
    if show_figs["config_dist"]:
        final_bits = {key: val / sample_shots for key, val in counts_bin.items()}
        all_keys = list(final_bits.keys())
        values = list(final_bits.values())
        max_value = max(values)
        selected_keys = [key for key, val in final_bits.items() if abs(val) > 0.7*max_value]
        
        colors = ["tab:purple" if key in selected_keys else "tab:grey" for key in all_keys]
        labels = [key if key in selected_keys else "" for key in all_keys]
        
        pos = range(len(all_keys))
        
        plt.rcParams.update({"font.size": 10})
        fig = plt.figure(figsize=(11, 6))
        ax = fig.add_subplot(1,1,1)
        
        ax.bar(all_keys, final_bits.values(), color=colors)
        
        ax.set_xticks(list(pos))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        
        plt.title("Result Distribution")
        plt.xlabel("Bitstrings (reversed)")
        plt.ylabel("Probability")
        
        plt.tight_layout()
        plt.show()
    
    if show_figs["optimal_graph"]:
        fig = draw_select(graph, most_likely_bitstring)
        plt.show()

    size = sum(i * weight(i, graph) for i in most_likely_bitstring)
    
    return {
        "bitstring": tuple(most_likely_bitstring),
        "size": size,
        "legal": mis_is_legal(most_likely_bitstring, graph),
        "build_time": t_build,
        "classical_opt_time": t_opt,
        "sampling_time": t_sample,
        "total_time": t_build + t_opt + t_sample,
        "best_cost": float(best_result.fun),
        "optimal_parameters": best_result.x,
        "circuit": candidate_circuit,
    }