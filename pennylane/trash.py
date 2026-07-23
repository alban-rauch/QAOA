# Main #
# ---- #

"""-------------------------------------------------------------------
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
    ② Barren plateaux: r_max and d"""

# Graphs #
# ------ #

# Kim and Vu’s paper [2] shows that this algorithm samples in an 
# asymptotically uniform way from the space of random graphs when d = O(n^(1/3 - eps))

# Inefficient
# def randomGilbert(N, q):
#     edges = []
#     graph = nx.Graph()
#     graph.add_nodes_from(range(N))
#     for i in range(N):
#         for j in range(i+1, N):
#             if np.random.uniform(0, 1) < q:
#                 edges.append((i, j))
#     graph.add_edges_from(edges)
#     return graph


# Warm start #
# ---------- #

# With scipy.optimize.linprog:
#
# linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method)
#        min c^T x
#   s.t. A_ub x <= b_ub 
#        A_eq x  = b_eq 
#
# Parameters:
## c = (1, ..., 1)^T because we maximize sum_i x_i -> important take -c because linprog minimizes
## Constraint: x_i + x_j <= 1 for every (i, j) in E
## A_ub = matrix with |E| rows where the row associated with (i, j) has entries | 1 if column i or j
##                                                                              | 0 otherwise
## b_ub = (1, ..., 1)^T
## bounds : [0,1]^n
## method : 'highs'



# H_m = - sin(theta) X - cos(theta) Z

# def relaxed_mixer(graph, angles):
#     coeffs = []
#     ops = []
#     for node in graph.nodes:
#         theta = angles[node]
#         coeffs.append(-np.sin(theta))
#         ops.append(qp.PauliX(node))
#         coeffs.append(-np.cos(theta))
#         ops.append(qp.PauliZ(node))
#     return qp.Hamiltonian(coeffs, ops)



## exp(- i β H_M) = exp(i β (sin(θ) X + cos(θ) Z))
##                = cos(β) I + i sin(β) (cos(θ) Z + sin(θ) X)
##                = RY(θ) RZ(-2β) RY(-θ)


# Variance landscape #
# ------------------ #

# part1 = np.linspace(0.005, 0.05, 10) * np.pi
# part2 = np.linspace(0.05, 0.10, 30+1)[1:] * np.pi
# part3 = np.linspace(0.10, 0.17, 30+1)[1:] * np.pi
# part4 = np.linspace(0.17, 0.23, 30+1)[1:] * np.pi
# part5 = np.linspace(0.23, 1.00, 80+1)[1:] * np.pi
# radii = np.concatenate([part1, part2, part3, part4, part5])

# part1 = np.linspace(0.005, 0.05, 10) * np.pi
# part2 = np.linspace(0.05, 0.10, 30+1)[1:] * np.pi
# part3 = np.linspace(0.10, 0.17, 30+1)[1:] * np.pi
# part4 = np.linspace(0.17, 0.23, 30+1)[1:] * np.pi
# part5 = np.linspace(0.23, 1.00, 80+1)[1:] * np.pi
# radii = np.concatenate([part1, part2, part3, part4, part5])


# def lhs_dist_r(n, flat_theta0, r, num_samples):
#     bins = np.arange(num_samples)
#     design = np.zeros((num_samples, n))
#     design[:, 0] = bins
#     for d in range(1, n):
#         design[:, d] = np.random.permutation(bins)
#     offsets = np.random.uniform(0, 1, size=(num_samples, n))
#     normal = (design + offsets) / num_samples
#     samples = flat_theta0 - r + normal * 2*r
#     return samples.reshape(num_samples, 2, n // 2)


# def full_energy_lanscape_hypercube(cost_function, theta0, num_samples):
#     flat_theta0 = np.concatenate(theta0)
#     n = flat_theta0.size
#     radii = np.linspace(0.1, np.pi, 100)
#     total_samples = []
#     idx = 0
#     indices = [0]
#     for r in radii:
#         r_samples = lhs_dist_r(n, flat_theta0, r, num_samples=num_samples)
#         total_samples.extend(r_samples)
#         idx = len(total_samples)
#         indices.append(idx)
#     energies = measure_energies(cost_function, total_samples)
#     variances = []
#     for i in range(len(radii)):
#         var_r = variance(energies[indices[i]:indices[i+1]])
#         variances.append(var_r)
#     return radii / np.pi, variances

# Classical #
# --------- #

# import time

# def measure_time(f, *x):
#     start = time.perf_counter()
#     result = f(*x)
#     end = time.perf_counter()
#     return result, end - start

# for i in range(2, 21):
#     graph, t1 = measure_time(gph.randomDRegular, i, i-1)
#     _, t2 = measure_time(best_config_brute, graph)
#     _, t3 = measure_time(best_config_branch_bound, graph)
#     print(t2/t3)


# Ansatz #
# ------ #

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