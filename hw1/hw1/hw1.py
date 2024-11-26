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
    
    return Q
    
def plot_dist(P):
    """
    Ploting the distribution P using barplot.

    Input:
    - P: 2d numpy array: [[values], [probabilities]].
    """
    
    pass


### Qeustion 4 ###

def evenBinom(n, p):
    """
    The program outputs the probability P(X\ is\ even) for the random variable X~Binom(n, p).
    
    Input:
    - n, p: The parameters for the binomial distribution.

    Returns:
    - prob: The output probability.
    """
    
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
    
    return prob

### Question 5 ###

def three_RV(X, Y, Z):
    """
 
    Input:          
    - X: 2d numpy array: [[values], [probabilities]].
    - Y: 2d numpy array: [[values], [probabilities]].
    - Z: 2d numpy array: [[values], [probabilities]].
    
    Returns:
    - v: The variance of X + Y + Z.
    """
    
    return v

def three_RV_pairwise_independent(X, Y, Z):
    """
 
    Input:
    - X: 2d numpy array: [[values], [probabilities]].
    - Y: 2d numpy array: [[values], [probabilities]].
    - Z: 2d numpy array: [[values], [probabilities]].
    
    Returns:
    - v: The variance of X + Y + Z.
    """
    
    return v

def is_pairwise_collectively(values, probs):
    """ 
    See explanation in the notebook
    """
    
    pass


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











    
    
    
    
    
    