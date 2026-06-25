"""
variance_landscape.py
---------------------
File to plot the energy variance lanscape around an optimal parameter value theta_0.
"""

import numpy as np
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt

def energy_variance_in_sample(circuit, hamiltonian, parameter_samples, backend, shots=10000):
    """
    Calculates the variance of the expected cost across all LHS parameter points.
    """
    estimator = Estimator(mode=backend)
    estimator.options.default_shots = shots
    
    if circuit.layout is not None:
        isa_hamiltonian = hamiltonian.apply_layout(circuit.layout)
    else:
        isa_hamiltonian = hamiltonian

    # lhs_parameters shape: (num_samples, n_dimensions)
    pubs = [(circuit, isa_hamiltonian, params) for params in parameter_samples]
    
    job = estimator.run(pubs)
    results = job.result()
    
    expectation_values = [pub_result.data.evs for pub_result in results]
    
    return np.var(expectation_values)


def lhs_dist_r(n, theta0, r, num_samples):
    bins = np.arange(num_samples)
    design = np.zeros((num_samples, n))
    design[:, 0] = bins
    for d in range(1, n):
        design[:, d] = np.random.permutation(bins)
    offsets = np.random.uniform(0, 1, size=(num_samples, n))
    normal = (design + offsets) / num_samples
    return theta0 - r + normal * 2*r


def full_energy_lanscape_hypercube(n, circuit, hamiltonian, theta0, num_samples, backend=AerSimulator(), estimator_shots=10000):
    # radii = np.linspace(0.1, np.pi, dist_steps)
    part1 = np.linspace(0.005, 0.05, 10) * np.pi
    part2 = np.linspace(0.05, 0.10, 30+1)[1:] * np.pi
    part3 = np.linspace(0.10, 0.17, 30+1)[1:] * np.pi
    part4 = np.linspace(0.17, 0.23, 30+1)[1:] * np.pi
    part5 = np.linspace(0.23, 1.00, 80+1)[1:] * np.pi
    radii = np.concatenate([part1, part2, part3, part4, part5])
    
    variances = []
    for r in radii:
        samples = lhs_dist_r(n, theta0, r, num_samples=num_samples)
        var_r = energy_variance_in_sample(circuit, hamiltonian, samples, backend, estimator_shots)
        variances.append(var_r)
    return radii / np.pi, variances




def lhs_shell_samples(n, theta0, r_inner, r_outer, num_samples):
    bins = (np.arange(num_samples) + np.random.uniform(0, 1, num_samples)) / num_samples
    radii = (r_inner**n + bins * (r_outer**n - r_inner**n)) ** (1.0 / n)
    np.random.shuffle(radii)
 
    directions = np.random.normal(size=(num_samples, n))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
 
    return theta0 + directions * radii[:, None]

    
def full_energy_landscape_shell(n, circuit, hamiltonian, theta0, num_samples, backend=AerSimulator(), estimator_shots=10000):
    # Adapt to make the curve more or less precise in specific regions
    part1 = np.linspace(0.005, 0.05, 10) * np.pi
    part2 = np.linspace(0.05, 0.10, 30+1)[1:] * np.pi
    part3 = np.linspace(0.10, 0.17, 30+1)[1:] * np.pi
    part4 = np.linspace(0.17, 0.23, 30+1)[1:] * np.pi
    part5 = np.linspace(0.23, 1.00, 80+1)[1:] * np.pi
    radii = np.concatenate([part1, part2, part3, part4, part5])
        
    variances = []
    r_inner = 0.0
    
    for r_outer in radii:
        shell_samples = lhs_shell_samples(n, theta0, r_inner, r_outer, num_samples)
        var_r = energy_variance_in_sample(circuit, hamiltonian, shell_samples, backend, estimator_shots)
        variances.append(var_r)
        r_inner = r_outer
        print(r_outer, 'done')
 
    return radii / np.pi, variances


def plot_variance(radii, variances):
    fig, ax = plt.subplots()
    
    ax.plot(radii, variances, '-o', 
             markersize=2,
             markerfacecolor=(1, 1, 1, 0.4),
             markeredgecolor='blue',
             alpha=0.7)
    ax.set_xlabel(r"$r / \pi$")
    ax.set_ylabel(r"Var($E$)")
    ax.set_title("Energy variance lanscape around optimized angle")
    
    ax.set_xlim(left=-0.01, right=None)
    ax.set_ylim(bottom=-0.1, top=None)
    
    ax.minorticks_on()
    ax.grid(visible=True, which='major', linestyle='-', linewidth=0.7, color='darkgray')
    ax.grid(visible=True, which='minor', linestyle=':', linewidth=0.5, color='gray')
    
    ax.set_axisbelow(True)
    
    plt.show()