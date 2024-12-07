###### Your ID ######
# ID1:
# ID2:
#####################

# imports
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


### Question 1 ###

def find_sample_size_binom(p=0.03, x=1, alpha=0.85):
    """
    Using Binom to returns the minimal number of samples required to have requested probability of receiving
    at least x defective products from a production line with a defective rate.
    """
    n = x  # Start with the minimum number of samples equal to x
    cdf = stats.binom.cdf(x - 1, n, p)  # Calculate the cumulative distribution function for Binomial
    while(1 - cdf < alpha):  # Check if the probability condition is met
        n += 1  # Increment the number of samples
        cdf = stats.binom.cdf(x - 1, n, p)  # Recalculate CDF for the updated sample size

    return n  # Return the minimal number of samples

def find_sample_size_nbinom(p=0.03, x=1, alpha=0.85):
    """
    Using NBinom to returns the minimal number of samples required to have requested probability of receiving
    at least x defective products from a production line with a defective rate.
    """
    n = x  # Start with the minimum number of samples equal to x
    cdf = stats.nbinom.cdf(x - 1, n, p)  # Calculate the cumulative distribution function for Negative Binomial
    while(1 - cdf < alpha):  # Check if the probability condition is met
        n += 1  # Increment the number of samples
        cdf = stats.nbinom.cdf(x - 1, n, p)  # Recalculate CDF for the updated sample size

    return n  # Return the minimal number of samples

def compare_q1():
    # Calculate the number of samples for the first case with 10% defective rate and at least 5 defective products
    n_independent_samples_first_part = find_sample_size_binom(p=0.1, x=5, alpha=0.9)
    # Calculate the number of samples for the second case with 30% defective rate and at least 15 defective products
    n_independent_samples_second_part = find_sample_size_binom(p=0.3, x=15, alpha=0.9)

    # Return the tuple of results for both cases
    return (n_independent_samples_first_part, n_independent_samples_second_part)

def same_prob(p=0.1, x=5, alpha=0.9):
    # Raise an exception if alpha is 0 to prevent invalid probability comparisons
    if alpha == 0:
        raise ValueError("Alpha cannot be 0. Please provide a non-zero value.")

    n = x  # Start with the minimum number of samples equal to x
    # Compute probabilities for both distributions
    prob_binom = 1 - stats.binom.cdf(x - 1, n, p)
    prob_nbinom = 1 - stats.nbinom.cdf(x - 1, n, p)
    # Check if the probabilities are close enough and meet the alpha threshold
    is_same_prob = np.isclose(prob_binom, prob_nbinom, atol=1e-2) and prob_binom > alpha and prob_nbinom > alpha

    # Increment n until the probabilities are close enough
    while(not is_same_prob):
        n += 1  # Increase the sample size
        prob_binom = 1 - stats.binom.cdf(x - 1, n, p)  # Recompute Binomial probability
        prob_nbinom = 1 - stats.nbinom.cdf(x - 1, n, p)  # Recompute Negative Binomial probability
        # Reevaluate the condition for probabilities being close
        is_same_prob = np.isclose(prob_binom, prob_nbinom, atol=1e-2) and prob_binom > alpha and prob_nbinom > alpha

    return n  # Return the sample size where probabilities match

### Question 2 ###

def empirical_centralized_third_moment(n=20, p=[0.2, 0.1, 0.1, 0.1, 0.2, 0.3]):
    """
    Create k=100 experiments where X is sampled. Calculate the empirical centralized third moment of Y based
    on your k experiments.
    """
    k = 100
    centralized_moments = []

    multinomial_dist = stats.multinomial(n, p)
    mean = multinomial_dist.mean()
    mean_Y = mean[1] + mean[2] + mean[3]

    X_k = multinomial_dist.rvs(size=k)
    for X_i in X_k:
        Y = X_i[1] + X_i[2] + X_i[3]
        centralized_moments.append((Y - mean_Y) ** 3)

    empirical_third_moment = np.mean(centralized_moments)

    return empirical_third_moment

def class_moment():

    return moment

def plot_moments():

    return dist_var

def plot_moments_smaller_variance():

    return dist_var


### Question 3 ###

def NFoldConv(P, n):
    """
    Calculating the distribution, Q, of the sum of n independent repeats of random variables,
    each of which has the distribution P.

    Input:
    - P: 2d numpy array: [[values], [probabilities]].
    - n: An integer.

    Returns:
    - Q: 2d numpy array: [[values], [probabilities]].
    """
    if n == 1:
        return P

    # create np array representin the distibution function starting at 0
    offset = -P[0].min()
    size = P[0].max() - P[0].min() + 1
    probs = np.zeros(int(size))

    for i in range(len(P[0])):
        probs[int(P[0][i] + offset)] = P[1][i]

    # Convolve - P*P*P... (n-1 times)
    results = probs
    for _ in range(n-1):
        results = np.convolve(results, probs)

    Q = np.vstack([np.arange(len(results))-offset*n, results])
    return Q

def plot_dist(P):
    """
    Ploting the distribution P using barplot.

    Input:
    - P: 2d numpy array: [[values], [probabilities]].
    """

    plt.figure(figsize=(8,6))
    plt.bar(P[0], P[1])

    plt.xlabel("Values")
    plt.ylabel("Probability")
    plt.show()


### Qeustion 4 ###

def evenBinom(n, p):
    """
    The program outputs the probability P(X\ is\ even) for the random variable X~Binom(n, p).

    Input:
    - n, p: The parameters for the binomial distribution.

    Returns:
    - prob: The output probability.
    """
    prob = 0
    for i in range(0, n + 1, 2):
        prob += stats.binom(n, p).pmf(i)

    return prob

def evenBinomFormula(n, p):
    """
    The program outputs the probability P(X\ is\ even) for the random variable X~Binom(n, p) Using a closed-form formula.
    It should also print the proof for the formula.

    Input:
    - n, p: The parameters for the binomial distribution.

    Returns:
    - prob: The output probability.
    """

    explanation = "1. For any a, b -> (a+b)^n = sum(k=0 to n) (n choose k) * a^k * b^(n-k)\n"
    explanation += "2. let a1 = p and b1 = 1 - p. Then (a1+b1)^n = sum(k=0 to n) (n choose k) * p^k * (1-p)^(n-k)\n"
    explanation += "3. Even k + Odds k = 1\n"
    explanation += "4. let a2 = p and b2 = (-1) * (1 - p).\n"
    explanation += "5. (a2+b2)^n = sum(k=0 to n) (n choose k) * p^k * (-1)(1-p)^(n-k)\n"
    explanation += "6. sum(k=0 to n) (n choose k) * p^k * (-1)(1-p)^(n-k) = Even k - Odd k\n"
    explanation += "7. (a2+b2)^n = (p - (1 - p))^n = (2p - 1)^n.\n"
    explanation += "8. from 7, 6, 3 => 2 * Even k = 1 + (2p - 1)^n\n"
    explanation += "9. Even k = (1 + (2p - 1)^n) / 2\n"

    print(explanation)

    return (1 + ((2*p - 1)**n)) / 2 # TODO: validate the explanation is for even and not for odd

### Question 5 ###

def expected_value(values, probabilities):
    expected_value = 0
    for i in range(len(values)):
        expected_value += values[i] * probabilities[i]
    
    return expected_value

def variance(values, probabilities):
    E_X2 = expected_value([x**2 for x in values], probabilities)
    E2_X = expected_value(values, probabilities) ** 2
    
    return E_X2 - E2_X

def three_RV(X, Y, Z, joint_probs):
    """
 
    Input:          
    - X: 2d numpy array: [[values], [probabilities]].
    - Y: 2d numpy array: [[values], [probabilities]].
    - Z: 2d numpy array: [[values], [probabilities]].
    - joint_probs: 3d numpy array: joint probability of X, Y and Z
    
    Returns:
    - v: The variance of X + Y + Z.
    """

    # V(X+Y+Z) = V(X)+V(Y)+V(Z)+2E(XY)+2E(XZ)+2E(YZ)-2E(X)E(Y)-2E(X)E(Z)-2E(Y)E(Z)

    V_X = variance(X[0], X[1])
    V_Y = variance(Y[0], Y[1])
    V_Z = variance(Z[0], Z[1])

    E_XY = 0
    E_XZ = 0
    E_YZ = 0
    
    for i, x in enumerate(X[0]):
        for j, y in enumerate(Y[0]):
            for k, z in enumerate(Z[0]):
                p_xyz = joint_probs[i, j, k]
                E_XY += x * y * p_xyz
                E_XZ += x * z * p_xyz
                E_YZ += y * z * p_xyz
    
    E_X = expected_value(X[0], X[1])
    E_Y = expected_value(Y[0], Y[1])
    E_Z = expected_value(Z[0], Z[1])

    v = V_X + V_Y + V_Z + 2*E_XY + 2*E_XZ + 2*E_YZ - 2*E_X*E_Y - 2*E_X*E_Z - 2*E_Y*E_Z

    return v

def three_RV_pairwise_independent(X, Y, Z, joint_probs):
    """
 
    Input:
    - X: 2d numpy array: [[values], [probabilities]].
    - Y: 2d numpy array: [[values], [probabilities]].
    - Z: 2d numpy array: [[values], [probabilities]].
    - joint_probs: 3d numpy array: joint probability of X, Y and Z
    
    Returns:
    - v: The variance of X + Y + Z.
    """

    # V(X+Y+Z) = V(X) + V(Y) + V(Z)

    V_X = variance(X[0], X[1])
    V_Y = variance(Y[0], Y[1])
    V_Z = variance(Z[0], Z[1])

    v = V_X + V_Y + V_Z

    return v

def is_pairwise_collectively(X, Y, Z, joint_probs):
    """

    Input:
    - X: 2d numpy array: [[values], [probabilities]].
    - Y: 2d numpy array: [[values], [probabilities]].
    - Z: 2d numpy array: [[values], [probabilities]].
    - joint_probs: 3d numpy array: joint probability of X, Y and Z
    
    Returns:
    TRUE or FALSE
    """

    P_X = np.sum(joint_probs, axis=(1, 2))  # Sums over Y and Z dimensions
    
    P_Y = np.sum(joint_probs, axis=(0, 2))  # Sums over X and Z dimensions
    
    P_Z = np.sum(joint_probs, axis=(0, 1))  # Sums over X and Y dimensions

    for i in range(len(X[0])):
        for j in range(len(Y[0])):
            joint_p_XY = joint_probs[i, j, :]
            marginal_p_XY = P_X[i] * P_Y[j]
            
            if not np.allclose(joint_p_XY, marginal_p_XY):
                return False
            
    for i in range(len(X[0])):
        for k in range(len(Z[0])):
            joint_p_XZ = joint_probs[i, :, k]
            marginal_p_XZ = P_X[i] * P_Z[k]
            
            if not np.allclose(joint_p_XZ, marginal_p_XZ):
                return False
            
    for j in range(len(Y[0])):
        for k in range(len(Z[0])):
            joint_p_YZ = joint_probs[:, j, k]
            marginal_p_YZ = P_Y[j] * P_Z[k]
            
            if not np.allclose(joint_p_YZ, marginal_p_YZ):
                return False

    return True

### Question 6 ###

def n_choose_i(n, i): # remove this manual impl if we can use 'math' library
    if i > n:
        return 0
    # Calculate factorial using np.prod
    def factorial(num):
        return np.prod(range(1, num + 1)) if num > 0 else 1

    return factorial(n) // (factorial(i) * factorial(n - i))

def expectedC(n, p):
    """
    The program outputs the expected value of the RV C as defined in the notebook.
    """

    # Initialize the expected value of C to zero
    expected_C = 0

    # Loop over all possible numbers of successes (0 to n)
    for i in range(n + 1):
        # Compute the probability of exactly i successes in n trials:
        # P(Y = i) = (n choose i) * (p^i) * ((1 - p)^(n - i))
        p_y_i = n_choose_i(n, i) * (p ** i) * ((1 - p) ** (n - i))

        # Compute the contribution to E[C] for this number of successes:
        # Contribution = (n choose i) * P(Y = i)
        expected_C += n_choose_i(n, i) * p_y_i

    # Return the total expected value of C
    return expected_C
















