import numpy as np
from utils import *
import qutip as qu


parameter_dim = 2
tol = 1e-9
steps = 1000
no_particles = 2500
K = 10
# true_particles = np.array([2.3, 7.8, 9.4])
true_particles = np.array([3, 400])


re_thresh = 0.5

# aux_bound = np.array([0.001, 1])
bounds = np.array([[0, 10], [10, 1000]])
# bounds = np.array([[0, 10]])

runs = 50
estimated_list = []
seed = 1
rng = np.random.default_rng(seed)

for i in range(runs):
    estimated_alpha, _ = generate_results(
        true_parameters=true_particles,
        steps=steps,
        tol=tol,
        parameter_dim=parameter_dim,
        bounds=bounds,
        no_particles=no_particles,
        resampling_threshold=re_thresh,
        filter_par=0.98,
    )

    estimated_list.append(estimated_alpha)

alpha_avg = np.mean(estimated_list)
alpha_std = np.std(estimated_list)
print("Final result after ", runs, " runs: ", alpha_avg, " pm ", alpha_std)

alpha_avg_mom = median_of_means(estimated_list, K)
print("Using median of means:", alpha_avg_mom)
# input("Press Enter to continue...")
