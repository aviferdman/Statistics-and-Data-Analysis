###### Your ID ######
# ID1:
# ID2:
#####################

# imports
import scipy.special as sp
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture


### Question 2 ###

def q2a(X, Y, Z):
    """

    Input:
    - X: 2d numpy array: [[values], [probabilities]].
    - Y: 2d numpy array: [[values], [probabilities]].
    - Z: 2d numpy array: [[values], [probabilities]].

    Returns:
    The number of parameters that define the joint distribution of X, Y and Z.
    """

    # P(X = x, Y = y, Z = z) has  mnk parameters, but the restriction on cdf reduces one
    # degree of freedom thus:

    return len(X[0])*len(Y[0])*len(Z[0])-1

def q2b(X, Y, Z):
    """

    Input:
    - X: 2d numpy array: [[values], [probabilities]].
    - Y: 2d numpy array: [[values], [probabilities]].
    - Z: 2d numpy array: [[values], [probabilities]].

    Returns:
    The number of parameters that define the joint distribution of X, Y and Z if we know that they are independent.
    """

    # now, P(x, y, z) = P(x)P(y)P(z), and for each one of the numbers of parameneters there are
    # n - 1 params, so:

    return len(X[0])+len(Y[0])+len(Z[0])-3


def q2c(X, Y, Z):
    """

    Input:
    - X: 2d numpy array: [[values], [probabilities]].
    - Y: 2d numpy array: [[values], [probabilities]].
    - Z: 2d numpy array: [[values], [probabilities]].

    Returns:
    The number of parameters that define the joint distribution of X, Y and Z if we know that they are independent.
    """

    # In this case, P(x,y,z) = P(z)P(x|z)P(y|z)
    # for P(z) there are k-1 parameters (1 reduced due to cdf restriction being one)
    # for P(x|z) - for each outcome of z (k) there are n-1 parameters, so total of k*(n-1)
    # and similar for P(y|z). so:
    k = len(Z[0])
    m = len(Y[0])
    n = len(X[0])
    return ((k-1) + (k*(n-1)) + k*(m-1))

### Question 3 ###
def gaussian_pdf(data, mu, sigma):
    return (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((data - mu) / sigma)**2)

def my_EM(data_path='./GMD.csv', mus=[4,9,None], sigmas=[0.5,0.5,1.5], ws=[None, 0.25, None], max_iters=10):
    """

    Input:
    - data_path	: path to data csv
    - mus   	: a numpy array: holds the initial guess for means of the Gaussians.
    - sigmas	: a numpy array: holds the initial guess for std of the Gaussians.
    - ws    	: a numpy array: holds the initial guess for weights of the Gaussians.

    * The function should be generic and support any number of Gaussians.
      (you don't need to get this number as a parameter for the function. You can conclude it from the other parameters).

    Returns:
    The output of the EM algorithms (the GMM final parameters): mus, sigmas, ws.
    """
    data = pd.read_csv(data_path, header=None, names=["Index", "Values"])
    values = data["Values"].values.reshape(-1, 1)
    K = len(mus)
    N = len(values)

    # create bitmaps to know which parameters not to update
    mus_bitmap = [1 if mu is not None else 0 for mu in mus]
    sigmas_bitmap = [1 if sigma is not None else 0 for sigma in sigmas]
    ws_bitmap = [1 if wi is not None else 0 for wi in ws]

    """ 1. Randomly init the unknown parameters """
    for i in range(K):
        if mus[i] is None:
            mus[i] = np.random.choice(values.flatten())
        if sigmas[i] is None:
            sigmas[i] = 1

    ws_sum = sum([ws[i] if ws[i] is not None else 0 for i in range(K)])
    if sum(ws_bitmap) != K:
        w_init = (1 - ws_sum)/(K - sum(ws_bitmap))
        for i in range(K):
            if ws[i] is None:
                ws[i] = w_init

    last_log_likelihood = 0
    for _ in range(max_iters):
        """ 2. Estimation: calculate responsibilities -
        		       the probability of wach point to be in each Gaussian """
        # fit Gaussians
        gmm = GaussianMixture(n_components=K, covariance_type='diag', max_iter=100)

        gmm.weights_init = ws
        gmm.means_init = np.array(mus).reshape(-1, 1)
        gmm.precisions_init = (np.array(sigmas)**-2).reshape(-1, 1)

        gmm.fit(values)

        resp = gmm.predict_proba(values)

        """ 3.Maximization: """
        """ 3A. update weights """
        updated_ws_sum = 0
        for i in range(K):
            if ws_bitmap[i] == 0:
                ws[i] = resp[:,i].sum() / N
                updated_ws_sum += ws[i]

        for i in range(K):
            if ws_bitmap[i] == 0:
                ws[i] *= (1-ws_sum)/updated_ws_sum
                ws[i] = float(ws[i])

        """ 3B. update means """
        for i in range(K):
            if mus_bitmap[i] == 0:
                mus[i] = float((resp[: , i] @ values.ravel()) / (resp[:, i].sum()))

        """ 3C. update variances """
        for i in range(K):
            if sigmas_bitmap[i] == 0:
                sigmas[i] = float(((resp[:, i] * (values.ravel() - mus[i]) **2).sum())/resp[:,i].sum())

        """Check convergence"""
        log_likelihood = np.sum(np.log(np.sum([w * gaussian_pdf(values, mu, sigma) for w, mu, sigma in zip(ws, mus, sigmas)], axis=0)))
        if abs(log_likelihood - last_log_likelihood) <= 1e-6:
            break
        last_log_likelihood = log_likelihood

    return mus, sigmas, ws

def q3d(mus=[4,9,15], sigmas=[0.5,0.5,1.5], ws=[0.125, 0.25, 0.625], samples=777):
    """

    Input:
    - mus   : a numpy array: holds the means of the gaussians.
    - sigmas: a numpy array: holds the stds of the gaussians.
    - ws    : a numpy array: holds the weights of the gaussians.
    - samples: numbers of samples to generate

    * The function should be generic and support any number of Gaussians.
      (you don't need to get this number as a parameter for the function. You can conclude it from the other parameters).

    Returns:
    The generated data.
    """

    K = len(mus)
    if len(sigmas) != K or len(ws) != K:
        raise ValueError('mus, sigmas and ws do not have the same length as should')
    N = np.random.multinomial(samples, ws)

    gmd = []
    for i in range(K):
        gaussian_i_data = np.random.normal(loc=mus[i], scale=sigmas[i], size=N[i])
        gmd.append(ws[i] * gaussian_i_data)

    gmd = np.concatenate(gmd)
    np.random.shuffle(gmd)

    return gmd

### Question 4 ###

def q4a(mu=75000, sigma=37500, salary=50000):
    """

    Input:
    - mu   : The mean of the annual salaries of employees in a large Randomistan company.
    - sigma: The std of the annual salaries of employees in a large Randomistan company.
    The annual salary of employees in a large Randomistan company are approximately normally distributed.

    Returns:
    The percent of people earn less than 'salary'.
    """

    prob = stats.norm.cdf(salary, loc=mu, scale=sigma)

    return prob * 100 # probability to percentage

def q4b(mu=75000, sigma=37500, min_salary=45000, max_salary=65000):
    """

    Input:
    - mu   : The mean of the annual salaries of employees in a large Randomistan company.
    - sigma: The std of the annual salaries of employees in a large Randomistan company.
    The annual salary of employees in a large Randomistan company are approximately normally distributed.

    Returns:
    The percent of people earn between 'min_salary' and 'max_salary'.
    """

    min_prob = stats.norm.cdf(min_salary, loc=mu, scale=sigma)

    max_prob = stats.norm.cdf(max_salary, loc=mu, scale=sigma)

    prob = max_prob - min_prob

    return prob * 100 # probability to percentage

def q4c(mu=75000, sigma=37500, salary=85000):
    """

    Input:
    - mu   : The mean of the annual salaries of employees in a large Randomistan company.
    - sigma: The std of the annual salaries of employees in a large Randomistan company.
    The annual salary of employees in a large Randomistan company are approximately normally distributed.

    Returns:
    The percent of people earn more than 'salary'.
    """
    prob = 1 - stats.norm.cdf(salary, loc=mu, scale=sigma)

    return prob * 100 # probability to percentage

def q4d(mu=75000, sigma=37500, salary=140000, n_employees=1000):
    """

    Input:
    - mu         : The mean of the annual salaries of employees in a large Randomistan company.
    - sigma      : The std of the annual salaries of employees in a large Randomistan company.
    - n_employees: The number of employees in the company
    The annual salary of employees in a large Randomistan company are approximately normally distributed.

    Returns:
    The number of employees in the company that you expect to earn more than 'salary'.
    """
    prob = 1 - stats.norm.cdf(salary, loc=mu, scale=sigma)

    return prob * n_employees # probability to employees

### Question 5 ###

def CC_Expected(N=10):
    """

    Input:
    - N: Number of different coupons.

    Returns:
    E(T_N)
    """
    harmonic_sum = 0
    for i in range(1, N + 1):
        harmonic_sum += 1 / i
    expected_value = N * harmonic_sum

    return expected_value

def CC_Variance(N=10):
    """

    Input:
    - N: Number of different coupons.

    Returns:
    V(T_N)
    """
    sum_reciprocal = sum(1 / i for i in range(1, N + 1))
    sum_reciprocal_squared = sum(1 / (i ** 2) for i in range(1, N + 1))

    variance = (N ** 2) * sum_reciprocal_squared - N * sum_reciprocal # TODO validate it: https://en.wikipedia.org/wiki/Coupon_collector%27s_problem

    return variance

def CC_T_Steps(N=10, n_steps=30):
    """

    Input:
    - N: Number of different coupons.

    Returns:
    The probability that T_N > n_steps
    """
    # Step 1: Build the transition matrix (size N+1 x N+1)
    P = np.zeros((N + 1, N + 1))

    for i in range(N):
        P[i][i] = i / N  # Stay in the same state
        P[i][i + 1] = (N - i) / N  # Move to the next state

    P[N][N] = 1  # Absorbing state (once all coupons are collected)

    # Step 2: Raise the matrix to the power of n_steps
    P_n = np.linalg.matrix_power(P, n_steps)

    # Step 3: Calculate the probability of not reaching the absorbing state
    # The probability of being in state N after n_steps is P_n[0][N]
    probability_TN_leq = P_n[0][N]  # Probability T_N <= n_steps
    probability_TN_greater = 1 - probability_TN_leq  # P(T_N > n_steps)

    return probability_TN_greater

def CC_S_Steps(N=10, n_steps=30):
    """
    Input:
    - N: Number of different coupons.

    Returns:
    The probability that S_N > n_steps
    """
    # Initialize transition matrix for states (i, j)
    matrix_size = (N + 1) * (N + 2) // 2
    P = np.zeros((matrix_size, matrix_size))

    # Map (i,j) to matrix index
    def state_index(i, j):
        return (i * (i + 1)) // 2 + j

    # Fill in transition probabilities
    for i in range(N + 1):
        for j in range(i + 1):  # j <= i
            current_state = state_index(i, j)

            if i == 0 and j == 0:
                # (0,0) -> (1,0) with probability 1
                P[current_state][state_index(1, 0)] = 1
            elif i < N:
                # (i,j) -> (i+1,j) with probability (N-i)/N
                P[current_state][state_index(i + 1, j)] = (N - i) / N
                # (i,j) -> (i,j+1) with probability (i-j)/N
                P[current_state][state_index(i, j + 1)] = (i - j) / N
                # (i,j) -> (i,j) with probability j/N
                P[current_state][current_state] = j / N
            else:
                if j < N:
                    # (N,j) -> (N,j + 1) with probability (N-j)/N
                    P[current_state][state_index(i, j + 1)] = (N - j) / N
                    # (N,j) -> (N,j) with probability j/N
                    P[current_state][current_state] = j / N

    P[state_index(N, N)][state_index(N, N)] = 1

    # Raise matrix to the power of n_steps
    P_n = np.linalg.matrix_power(P, n_steps)

    # Calculate probability of not reaching absorbing state (N,N)
    start_state = state_index(0, 0)
    absorbing_state = state_index(N, N)
    probability_SN_leq = P_n[start_state][absorbing_state]
    probability_SN_greater = 1 - probability_SN_leq

    return probability_SN_greater

