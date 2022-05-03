import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
import joblib
from numba import jit




from qutip import tensor, sigmaz, identity, basis, sigmax, sigmay

import qutip as qu

X = np.array([[0, 1], [1, 0]]).astype(np.complex128)
Y = (np.array([[0, -1], [1, 0]])*1j).astype(np.complex128)
Z = np.array([[1, 0], [0, -1]]).astype(np.complex128)
I = np.array([[1, 0],[0, 1]]).astype(np.complex128)

h_bar = 1
np.random.seed(1)



def initial_state(dim=1):
    """Produces |plus> state in a desired dimension

    Args:
        dim (int, optional): Number of qubits. Defaults to 1.

    Returns:
       Quobj: Output ket
    """
    plus = (basis(2, 0) + basis(2, 1)).unit()
    if dim ==1:
        return plus
    else:
        return tensor([plus for _ in range(dim)])

@jit(nopython=True)
def evolve_state(H, v, t):
    """Evolves a state v given a Hamiltonian by manual exponentiation.
    

    Args:
        H (np.array): Hamiltonian
        v (np.array): Vector to be evolved
        t (float): Time to be evolved

    Returns:
        np.array: Evolved state
    """
    evolved_state = mat_exp(-t*H*1j)@v
    return evolved_state/np.linalg.norm(evolved_state)




def H(free_model, *omega):
    return free_model(*omega)


def free_model(omega):
    return omega*sigmaz()


def free_model_2(omega):
    return omega[0]*X + omega[1]*Y

def free_model_3(omega):
    return omega[0]*X + omega[1]*Z





@jit(nopython=True)
def model_two_qubits_free(omega= np.array([])):
    H = (omega[0]*np.kron(Z, I ) + omega[1]*np.kron(I, Z))
    # H = H.astype(np.complex128)
    return H

@jit(nopython=True)
def model_three_qubits_free(omega):
    H = omega[0]*np.kron(np.kron(Z, I ), I) + omega[1]*np.kron(np.kron(I, Z), I ) + omega[2]*np.kron(np.kron(I,I), Z)
    return H


@jit(nopython=True)
def model_three_qubits_interact(omega):
    H = omega[0]*np.kron(np.kron(X, Z ), I) + omega[1]*np.kron(np.kron(I, X), Z ) + omega[2]*np.kron(np.kron(Z,I), X)
    return H

@jit(nopython=True)
def model_three_qubits_interact_2(omega):
    H = omega[0]*np.kron(np.kron(Z, Z ), I) + omega[1]*np.kron(np.kron(I, Z), Z ) + omega[2]*np.kron(np.kron(Z,I), Z)
    return H

@jit(nopython=True)
def model_three_qubits_interact_3(omega):
    H = omega[0]*np.kron(np.kron(Z, Z ), I) + omega[1]*np.kron(np.kron(I, Z), Z ) + omega[2]*np.kron(np.kron(Z,I), Z) + omega[3]*np.kron(np.kron(Z, I ), I) + omega[4]*np.kron(np.kron(I, Z), I ) + omega[5]*np.kron(np.kron(I,I), Z) 
    
    return H

@jit(nopython=True)
def model_three_qubits_interact_4(omega):
    H = omega[0]*np.kron(np.kron(Z, Z ), I) + omega[1]*np.kron(np.kron(I, Z), Z ) + omega[2]*np.kron(np.kron(Z,I), Z) + omega[3]*np.kron(np.kron(Z, I ), I) + omega[4]*np.kron(np.kron(I, Z), I ) + omega[5]*np.kron(np.kron(I,I), Z) +\
        omega[6]*np.kron(np.kron(X, X ), I) + omega[7]*np.kron(np.kron(I, X), X ) + omega[8]*np.kron(np.kron(X,I), X) 
    
    return H

@jit(nopython=True)
def model_two_qubits_free_interact(omega):
    H = (omega[0]*np.kron(Z, I ) + omega[1]*np.kron(I, Z) + omega[2]*np.kron(Z, Z))
    return H

@jit(nopython=True)
def model_two_qubits_interact(omega):
    H = (omega[0]*np.kron(Z, I ) + omega[1]*np.kron(I, Z) + omega[2]*np.kron(Z, Z))
    return H    

@jit(nopython=True)
def model_two_qubits_interact_oneparam(omega):
    H = (omega*np.kron(Z, X ) )
    return H   


def median_of_means(list_results, K):
    bag = np.array_split(list_results, K)
    return np.median(np.mean(bag, axis=1))


# def model_two_qubits_interact(omega):
    
#     H = omega[0]*tensor(sigmaz(), identity(2)) + omega[1]*tensor(identity(2), sigmaz()) + omega[2]*tensor( sigmaz(), sigmax())
#     return H


# def model_two_qubits_interact(omega):
    
#     H = 0.5*(omega[0]*tensor(sigmaz(), identity(2)) + omega[1]*tensor( sigmaz(), sigmax()))
#     return H


def model_zx_xy(omega):
    H = omega*tensor(sigmaz(), sigmax())# + omega[1]*tensor( sigmax(), sigmay())
    return H

@jit(nopython=True)
def mat_exp(A):
    """Manual matrix exponential by diagonalization and exponentiation.
    Using numba, this is faster than scipy.

    Args:
        A (np.array): Matrix to be exponentiated

    Returns:
        np.array: exp(A) (in the usual sense, not element-wise)
    """
    d, Y = np.linalg.eig(A)
    Yinv = np.linalg.pinv(Y)
    D = np.diag(np.exp(d))
    B = Y@D@Yinv
    return B

@jit(nopython=True)
def compute_prob(state, projector):
    """Compute the probability given a state (ket) and a projector.

    Args:
        state (np.array): State to compute probability.
        projector (np.array): The projector corresponding to the output state.

    Returns:
        np.array: Probability
    """
    rho = np.outer(state, np.conjugate(state.T))
    return np.abs(np.real(np.trace( rho@projector)))


# def get_projectors(matrix):
#     _, eig_v = np.linalg.eig(matrix)
#     proj = []
#     for eigen_1 in eig_v:
#         for eigen_2 in eig_v:
#             proj_i = np.outer(eigen_1,np.conjugate(eigen_1.T))
#             proj_j = np.outer(eigen_2,np.conjugate(eigen_2.T))
#             proj_i_corder = np.reshape(proj_i, newshape=proj_i.shape, order='C')
#             proj_j_corder = np.reshape(proj_j, newshape=proj_j.shape, order='C')
            
#             proj.append(np.kron(proj_i_corder, proj_j_corder) )
#     return proj

def get_projectors(matrix, dim):
    """Generate the projectors of a measurement basis in a given dimension.
    Given a matrix, computes its eigenstates and tensor product of
    projectors corresponding to the possible outputs.
    E.g.:
    If matrix is X, dim = 2. It computes (|++>x<++|, |+- >x<+- |
    , |+- >x<+- |, | -- >x< -- | )

    Args:
        matrix (np.array): Matrix corresponding to the measurement basis
        dim (int): Number of qubits.

    Returns:
        list of np.arrays: List with all the projectors.
    """
    _, eig_v = np.linalg.eig(matrix)
    a = np.outer(eig_v[0],np.conjugate(eig_v[0].T))
    b = np.outer(eig_v[1],np.conjugate(eig_v[1].T))
    # a = a.astype(np.complex128)
    # b = b.astype(np.complex128)
    projs_init = [a, b]
    projs = [a,b]
    for _ in (range(dim-1)):
        n_projs = [np.kron(j, k) for j in projs for k in projs_init] 
        projs = n_projs
    return np.array(projs)

@jit(nopython=True)
def get_all_probs(state, projectors):
    """Generate all the probabilities given a state and a list of projectors

    Args:
        state (np.array): State vector
        projectors (List(np.array)): List of projectors

    Returns:
        np.array: Array with probabilities.
    """
    probs_t = []
    for proj_i in projectors:
        proj_i=np.ascontiguousarray(proj_i)
        probs_t.append(compute_prob(  state, proj_i))
    output = np.array(probs_t)    
    return output/np.sum(output)

@jit(nopython=True)
def evolve_all_particles(t, state, hguess, particles, projectors):
    """Evolution of the initial state under the evolution of possible guesses
    Hamiltonians. It outputs the probabilities given an array of projectors.

    Args:
        t (float): Time to evolve particles
        state (np.array(complex128)): Initial state
        hguess (function): Function that outputs the hamiltonian. "free-parameter Hamiltonian"
        particles (np.array(float)): np.array with guesses particles. The shape must be the following:
            if the dimension is 2, a particle p^i will be (p_1, p_2). The whole array must be:
                np.array([[p^1_1, p^2_1, ..., p^n_1], [p^1_2, p^2_2, ..., p^n_2] ])
        projectors (np.array(np.array)): The projectors corresponding to the possible outcomes.

    Returns:
        List of np.arrays: List with the array of probabilities.
    """
    probs = []
    for part_i in particles.T:
        
        psi_t = evolve_state(hguess(part_i), state, t)
        probs_t = get_all_probs(psi_t, projectors)
        probs.append(probs_t)

    return probs    

# def evolve_state(H, v, t):
#     evolved_state = expm(-t*H*1j)@v
#     return evolved_state/np.linalg.norm(evolved_state)




def norm_H(H):
    return np.sqrt(np.max(np.linalg.eig(np.transpose(np.conjugate(H))@H)[0]))


def normalize_distribution(p):
    '''
    Normalize probability distribution p. If multidimensional, it assumes that each row (axis=1) contains the entire p
    Each column will correspond to a different distribution

    Args:
        p: numpy array containing the probability distribution(s)

    Returns:
        The normalized probability function.
    '''
    if len(p.shape) == 1:
        return p/p.sum()
    else:
        return p/p.sum(axis=1).reshape(-1, 1)




def PGH(particles, distribution):
    """Gives measurement time using Particle Guess Heuristic.

    Args:
        particles (np.array(np.array)): Array with particles, the second axis corresponds to the dimension
        of the matrix, i.e., a shape of (2000, 2) corresponds to 2000 two-dimensional particles.
        distribution (np.array): Discrete probability distribution of the particles

    Returns:
        float: Measurement time
    """
    rng = np.random.default_rng()
    if len(particles.shape) == 1 and len(distribution.shape) == 1:
        x1, x2 = rng.choice(
            particles, size=2, p=normalize_distribution(distribution), replace=False)
        t = 1 / np.linalg.norm(x1-x2)
        return t
    else:
        
        M = particles.shape[1] #no of particles
        l1, l2 = rng.choice(
                   M , size=2, p=normalize_distribution(distribution), replace=False)
        p1 = particles[:, l1]
        p2 = particles[:, l2]

        t = 1 / np.linalg.norm(p1 - p2)
        return t

# def pgh2(particles, distribution):
#     prob = normalize_distribution(distribution)
#     suma = 0

#     return 1/ np.sqrt(np.max(Cov(particles[0, :], prob), Cov(particles[1, :], prob)))
#     # for i in range(particles.shape[0]):
#     #     suma = suma + Cov(particles[i, :], prob)
#     # return 1/np.sqrt(suma/particles.shape[0])    


def Mean(particles, distribution):
    """Computes the mean of the particles given a discrete probability distribution.

    Args:
        particles (np.array(np.array)): Array with particles, the second axis corresponds to the dimension
        of the matrix, i.e., a shape of (2000, 2) corresponds to 2000 two-dimensional particles.
        distribution (np.array): Discrete probability distribution of the particles

    Returns:
        np.array: Mean value.
    """
    if len(particles.shape) == 1 and len(distribution.shape) == 1:
        p = normalize_distribution(distribution)
        return (p*particles).sum()
    else:
        return (particles*normalize_distribution(distribution)).sum(axis=1)




def Cov(particles, distribution):
    """Computes the covariance matrix of the particles given a discrete probability distribution.


    Args:
        particles (np.array(np.array)): Array with particles, the second axis corresponds to the dimension
        of the matrix, i.e., a shape of (2000, 2) corresponds to 2000 two-dimensional particles.
        distribution (np.array): Discrete probability distribution of the particles.

    Returns:
        np.array: Covariance matrix.
    """
    if len(particles.shape) == 1 and len(distribution.shape) == 1:
        p = normalize_distribution(distribution)
        mu = Mean(particles, p)
        sigma = (p*(particles**2)).sum() - (mu*mu)
        return sigma
    else:
        p = normalize_distribution(distribution)
        mu = Mean(particles, p).reshape(-1, 1)
        D = particles.shape[0]
        cov_sum = np.zeros([D, D])
        for i in range(particles.shape[1]):
            part = particles[:, i].reshape(-1, 1)
            cov_sum = cov_sum + p[i]*part@part.T
            
        mu = Mean(particles, p).reshape(-1, 1)
        return cov_sum - mu@mu.T
        




# #OLD
# def resample(particles, distribution, a):
#     prob = normalize_distribution(distribution)
#     h = np.sqrt(1-a**2)
    
#     new_weights = []
#     new_particles = []
#     if len(particles.shape) == 1 and len(distribution.shape) == 1:
#         mu = Mean(particles, prob)

#         for _ in range(len(particles)):
#             part_candidate = np.random.choice(
#                 particles, size=1, p=prob, replace=False)
#             mu_i = a*part_candidate + (1-a)*mu
#             Sigma = h**2 * np.sqrt(Cov(particles, prob))
#             part_prime = np.random.normal(mu_i, Sigma)
#             new_particles.append(part_prime[0])
#             new_weights.append(1/len(particles))

#         return (np.array(new_particles), np.array(new_weights))
#     else:
#         new_particles = np.zeros(particles.shape)
#         new_weights = np.zeros(distribution.shape)
#         M = particles.shape[1] #number of particles
#         for i in range(particles.shape[0]):
#             for j in range(particles.shape[1]):
#                 loc_candidate = np.random.choice(
#                     M , size=1, p=prob, replace=False)
#                 part_candidate = particles[i, loc_candidate]
#                 # print(part_candidate)
#                 mu=Mean(particles[i, :], prob)
#                 Sigma = h**2 * np.sqrt(Cov(particles[i, :], prob))
#                 mu_i = a*part_candidate + (1-a)*mu
#                 part_prime = np.random.normal(mu_i, Sigma)
#                 new_particles[i, j] = part_prime
#                 new_weights[j] = 1/M
   
#         return (new_particles, new_weights)




def resample(particles, distribution, a):
    """_summary_

    Args:
        particles (np.array(np.array)): Array with particles, the second axis corresponds to the dimension
        of the matrix, i.e., a shape of (2000, 2) corresponds to 2000 two-dimensional particles.

        distribution (np.array): Discrete probability distribution of the particles.
        
        a (float): Filter parameter. Selects how much are the new particles perturbed using the mean of the distribution. A value of
        1 does not perturb the position of the new particles.

    Returns:
        new_particles (np.array(np.array)): New particles obtained by resampling
        new_weights (np.array): Normalized uniform weights for the new population of particles.
    """
    prob = normalize_distribution(distribution)
    h = np.sqrt(1-a**2)
    rng = np.random.default_rng()
    new_weights = []
    new_particles = []
    if len(particles.shape) == 1 and len(distribution.shape) == 1:
        mu = Mean(particles, prob)
        Sigma = h**2 * np.sqrt(Cov(particles, prob))
        M = len(particles)
        part_candidates = rng.choice(particles, size=M, p=prob, replace=True)
        mu_i = a*part_candidates + (1-a)*mu
        part_prime = rng.normal(loc = mu_i, scale=Sigma)
        new_particles= part_prime
        new_weights = np.ones(M)/M
        
        # for _ in range(len(particles)):
        #     part_candidate = rng.choice(
        #         particles, size=1, p=prob, replace=False)
        #     mu_i = a*part_candidate + (1-a)*mu
        #     Sigma = h**2 * np.sqrt(Cov(particles, prob))
        #     part_prime = rng.normal(mu_i, Sigma)
        #     new_particles.append(part_prime[0])
        #     new_weights.append(1/len(particles))

        return (np.array(new_particles), np.array(new_weights))
    else:
        M = particles.shape[1] #number of particles
        new_particles = np.zeros(particles.shape)
        new_weights = np.ones(distribution.shape)
        new_weights = np.ones(distribution.shape)/M
        
        for i in range(particles.shape[0]):
            
            # for j in range(particles.shape[1]):
            loc_candidates = rng.choice(
                M , size=M, p=prob, replace=True)
            part_candidates = particles[i, loc_candidates]
            # print(part_candidate)
            mu=Mean(particles[i, :], prob)
            # print(particles[i, :])
            Sigma = h**2 * np.sqrt(Cov(particles[i, :], prob))
            mu_i = a*part_candidates + (1-a)*mu
            part_prime = rng.normal(loc = mu_i, scale=Sigma)
            # print(part_prime)
            new_particles[i, :] = part_prime
                
        
        return (new_particles, new_weights)


def Mse(par, true_par):
    """Computes the mean squared error between two lists or values.

    Args:
        par (list of floats or float): List of floats or float with predicted value.
        true_par (list of floats or float): List of floats or float with true value.

    Returns:
        float: Mean squared error.
    """
    suma = 0
    # for i in range(len(par)):
    #     suma = suma + (par[i] - true_par[i])**2

    if isinstance(par, list):
        for i in range(len(par)):
            suma = suma + (par[i] - true_par[i])**2
    elif  type(par).__module__ == np.__name__ and type(true_par).__module__ == np.__name__ :
        suma =  np.sum((par - true_par)**2 )
    else:
        suma = (par - true_par)**2        
    return suma   


# def update_SMC(t, particles, weights, h_true, h_guess, state):
#     sample = Sample(state, h_true, t_type="single", size=1, t_single=t)[0]

#     if len(particles.shape) == 1 and len(weights.shape) == 1:
#         probs_0 = [prob_0(evolve_state_fast(h_guess(particle_i), state, t)) 
#                     for particle_i in particles]
        
#     else:
#         probs_0 = [prob_0(evolve_state_fast(h_guess(particle_i), state, t)) 
#                     for particle_i in particles.T]
#     probs_sample = np.array([p0 if sample == 0 else 1 - p0 for p0 in probs_0])
#     # likelihoods[i_sample, :] = probs_sample
#     new_weights = weights * probs_sample
#     n_weights = normalize_distribution(new_weights)

#     return particles, n_weights

# def update_SMC(t, particles, weights, h_true, h_guess, state, projectors):


#     state_sample = evolve_state(h_true, state, t)
#     probs = get_all_probs(state_sample, projectors )
#     result_sample = np.random.choice(range(len(projectors)), p = normalize_distribution(probs) )
    
    

#     all_probs = evolve_all_particles(t, state, h_guess, particles, projectors)
#     probs = np.array(np.array(all_probs).T[result_sample])



#     # likelihoods[i_sample, :] = probs_sample
#     # probs = np.array(probs)
#     # new_weights = weights * probs
#     n_weights = normalize_distribution(weights * probs)

#     return particles, n_weights



def update_SMC(t, particles, weights, h_true, h_guess, state, projectors, k_mean=15):
    """Updates the particles' probability distribution. Given a measurement time t, performs a simulation of the 
    actual system and gets a result. We compute the probability of getting the result by simulating the evolution using
    all the particles. Each particle weight is multiplied using the probability of getting the same result as the true system.
    If k_mean>=1, we perform k_mean experiments at the same time and average the vector of probabilities per particle that will multiply the weights.
    This will supress unlikely results that distorts the distribution but will increase the number of iterations by a factor of "k_mean".
    Args:
        t (float): The time at which perform the experiments.

        particles (np.array(np.array)): Array with particles, the second axis corresponds to the dimension
        of the matrix, i.e., a shape of (2000, 2) corresponds to 2000 two-dimensional particles.

        distribution (np.array): Discrete probability distribution of the particles.

        h_true (np.array): Matrix which corresponds to the true hamiltonian describing the system.
        h_guess (function): A function that receives a vector of particles and outputs the matrix corresponding to the hamiltonian using those parameters.
        state (np.array): Initial ket state.
        projectors (list of np.arrays): list containing the projectors corresponding to the measurement basis. E.g. in the case of a qubit: [ |0> <0|, |1> <1| ]
        k_mean (int, optional): Number of experiments to perform to calculate probabilities before multiplying the distribution. Defaults to 1.

    Returns:
        particles (np.array(np.array)): Particles. They are the same as the input ones.
        n_weights (np.array): Weights after being updated
    """
    rng = np.random.default_rng()
    state_sample = evolve_state(h_true, state, t) #get a sample evolved state from the true system
    probs = get_all_probs(state_sample, projectors ) #compute all the probabilities for all the possible outcomes.

    
    all_probs = evolve_all_particles(t, state, h_guess, particles, projectors) #evolve all the particles to the measurement time and compute the probabilities

    mean_probs = np.zeros(len(weights)) 
    for _ in range(k_mean):
        
        result_sample = rng.choice(range(len(projectors)), p = normalize_distribution(probs) ) #select a result from the true system.
        
        probs_temp = np.array(np.array(all_probs).T[result_sample]) #select the probability from the particles that corresponds to the result
        mean_probs = mean_probs + probs_temp 
    mean_probs = mean_probs/k_mean #average the probabilities if k_mean >=1



    # likelihoods[i_sample, :] = probs_sample
    # probs = np.array(probs)
    # new_weights = weights * probs
    n_weights = normalize_distribution(weights * mean_probs) #update weights and normalize.

    return particles, n_weights

def adaptive_bayesian_learn(particles, weights, h_true, h_guess, true_parameters, state, steps,projectors, no_particles, filter_par, resampling_threshold, tol=1E-5 ):
    """Adaptive bayesian learning. Given a true Hamiltonian and a Hamiltonian with free parameter, we learn the true parameters of the system. The measurement time is
    obtained using PGH in each iteration, updating the distribution of the particle by comparing the result from the true system and each of the possible particles.
    If the weights of the particles satisfy  sum_i 1/(w_i)^2 < threshold * N_particles, the particles are resampled.
    The algorithm stops after a max of iterations; if the selected time is near infinity, which means that the standard deviation of the particles is very small; or if 
    the Mean Squared Error is lower than a certain tolerance.

    Args:
        
        particles (np.array(np.array)): Array with particles, the second axis corresponds to the dimension
        of the matrix, i.e., a shape of (2000, 2) corresponds to 2000 two-dimensional particles.

        distribution (np.array): Discrete probability distribution of the particles.

        h_true (np.array): Matrix which corresponds to the true hamiltonian describing the system.
        h_guess (function): A function that receives a vector of particles and outputs the matrix corresponding to the hamiltonian using those parameters.
        state (np.array): Initial ket state.
        true_parameters (np.array of floats): True parameters of the system.
        state (numpy array): Initial ket state.
        steps (int): Number of steps to perform.
        projectors (list of np.arrays): list containing the projectors corresponding to the measurement basis. E.g. in the case of a qubit: [ |0> <0|, |1> <1| ]
        no_particles (int): Number of particles.
        filter_par (float): Filter parameter. Selects how much are the new particles perturbed using the mean of the distribution. A value of
        1 does not perturb the position of the new particles.
        resampling_threshold (float): Controls the fraction of the number of particles after which the resampling is performed if sum_i 1/w_i^2 < resampling_threshold.
        tol (float, optional): If MSE(par_true, par_predicted)<tol, the program finishes. Defaults to 1E-5.

    Returns:
        estimated_parameter (np.array): Estimated parameters.
        evolution_dic (dictionary): Dictionary which contains the particles distribution, weights and estimated parameter at each step.
        This information is quite heavy, useful for monitoring purposes.
    """
    step_list = []
    part_list = []
    weight_list = []
    estimated_parameter_list = []
    time_tol = 1E6
    for i_step in range(steps):
        # print("\n--------------------")
        print("Experiment no. ", i_step)
        t = PGH(particles, weights)
        # t = 1/np.linalg.norm(Cov(particles, weights))
        
        print("Time: ", t)
        if t > time_tol:
            print("--------------------------")
            print("t > ", time_tol)
            break
        # t = 1/np.sqrt(Cov(particles, weights))
        # t = 1/np.sqrt(np.trace(Cov(particles, weights)))
        # t = 1/np.sqrt(np.linalg.det(Cov(particles, weights)))

        # print(weights)
        # print("time:", t)
        # print("Mean", Mean(particles, normalize_distribution(weights)))
        # print("Cov", Cov(particles, normalize_distribution(weights)) ) 


        particles, weights = update_SMC(
            t, particles, weights, h_true, h_guess, state, projectors)
        # print(weights)
        step_list.append(i_step)    
        part_list.append(particles)  
        weight_list.append(weights)

        # print(weights)
        # print("1/w^2: ", 1/np.sum(weights**2))
        
        if 1/np.sum(weights**2) < no_particles*(resampling_threshold):# and i_step>=5:
            print("----RESAMPLING----")
            particles, weights = resample(particles, weights, a=filter_par)
        
        # if i_step%5==0:
        estimated_parameter = Mean(particles, weights)
        estimated_parameter_list.append(estimated_parameter)
        print("MSE: ", Mse(estimated_parameter,true_parameters))
        if i_step%10==0:
            estimated_parameter = Mean(particles, weights)
            print("estimation: ", Mean(particles, weights))
        


        if Mse(estimated_parameter,true_parameters) < tol:
            print("--------------------------")
            print("MSE < ", tol)
            
            break
        
    # estimated_parameter = np.sum(np.dot(weights, particles))
    estimated_parameter = Mean(particles, weights)
    print("Estimated parameters: ", estimated_parameter, "pm ", np.linalg.norm(Cov(particles, weights)) )
    evolution_dic = {"steps":step_list,
    "particles":part_list,
    "weights":weight_list,
    "estimated_parameter" : estimated_parameter_list
    }
    return estimated_parameter, evolution_dic


def generate_results(
    true_parameters, steps, h_true, h_guess, tol, projectors, quantum_dim, parameter_dim,
    bounds, no_particles, filter_par , resampling_threshold):
    """Prepare auxiliary arrays needed to perform adaptive bayesian learning.

    Args:
        true_parameters (np.array of floats): True parameters of the system.
        steps (int): Number of steps to perform.
        h_true (np.array): Matrix which corresponds to the true hamiltonian describing the system.
        h_guess (function): A function that receives a vector of particles and outputs the matrix corresponding to the hamiltonian using those parameters.

        tol (float, optional): If MSE(par_true, par_predicted)<tol, the program finishes. Defaults to 1E-5.
        projectors (list of np.arrays): list containing the projectors corresponding to the measurement basis. E.g. in the case of a qubit: [ |0> <0|, |1> <1| ]
        quantum_dim (int): Number of qubits
        parameter_dim (int): Dimension of the parameter vector.
        bounds (np.array of np.arrays): Multidimensional array with the bounds of possible parameters values.
        no_particles (int): Number of particles
        filter_par (float): Filter parameter. Selects how much are the new particles perturbed using the mean of the distribution. A value of
        1 does not perturb the position of the new particles.
        resampling_threshold (float): Controls the fraction of the number of particles after which the resampling is performed if sum_i 1/w_i^2 < resampling_threshold.



    Returns:
        estimated_parameter (np.array): Estimated parameters.
        evolution_dic (dictionary): Dictionary which contains the particles distribution, weights and estimated parameter at each step.
        This information is quite heavy, useful for monitoring purposes.
    """
    weights = normalize_distribution(np.ones(no_particles))
    
    rng = np.random.default_rng()
    particles = np.zeros([parameter_dim, no_particles])

    for i in range(bounds.shape[0]):
        p_min, p_max = bounds[i, :]
        a = np.linspace(p_min, p_max, no_particles)
        rng.shuffle(a)
        # print(a)
        particles[i, :] = a #rng.permutation(np.linspace(p_min, p_max, no_particles))
        

    if parameter_dim==1:
        particles = particles[0, :]
    
    state = initial_state(dim=quantum_dim)[:]




    estimated_parameter, evolution_dic = adaptive_bayesian_learn(
    particles=particles, weights=weights, true_parameters=true_parameters, state=state,
    steps=steps, h_true=h_true, h_guess=h_guess,
    tol=tol, projectors=projectors, no_particles=no_particles, filter_par=filter_par, resampling_threshold=resampling_threshold)

    return estimated_parameter, evolution_dic

if __name__ == "__main__":
    print("Main auxiliary function ")
        
    # # state = np.array([1, 0], dtype=np.complex128)
    # # state = state/np.linalg.norm(state)
    # D = 2

    # state = initial_state(dim=2)[:]
    # # np.array(state[:, 0], dtype=np.complex128)

    # bounds = np.array([[0.1, 2],
    #                     [0.1, 2]])

    # # bounds = np.array([[0.01, 6]])
    # alpha1 = 0.7 # 0.834
    # alpha2 = 0.3
    # # alpha3 = 2.5

    # true_parameters = np.array([alpha1, alpha2])

    
    # h = model_two_qubits_free(np.array([alpha1, alpha2]))
    # # h = H(free_model, alpha1)
    # h_guess = model_two_qubits_free

    # measurement_basis = qu.tensor(qu.sigmax(), qu.sigmax())[:]
    # projectors = get_projectors(measurement_basis)
    # projectors = np.array(projectors)
    # # projector = sigmax()
    # no_particles = 500
    # weights = normalize_distribution(np.ones(no_particles))


    # particles = np.zeros([D, no_particles])

    # for i in range(bounds.shape[0]):
    #     p_min, p_max = bounds[i, :]
    #     particles[i, :] = np.linspace(p_min, p_max, no_particles)

    # if D==1:
    #     particles = particles[0, :]

    # steps = 1000

    # estimated_alpha = adaptive_bayesian_learn(
    #     particles=particles, weights=weights, true_parameters=true_parameters, state=state, steps=steps, h_true=h, h_guess=h_guess, tol=1E-9, projectors=projectors, no_particles=no_particles, filter_par=0.98)
    # print("Estimated results: ",estimated_alpha[0])
    # # print(MSE(estimated_alpha, alpha))
    # print("end")
    