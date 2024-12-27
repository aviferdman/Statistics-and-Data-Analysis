###### Your ID ######
# ID1:
# ID2:
#####################

# imports
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


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

    return len(X[0])*len(Y[0])*len(Z[0])-3


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

def my_EM(mus, sigmas, ws):
    """

    Input:
    - mus   : a numpy array: holds the initial guess for means of the Gaussians.
    - sigmas: a numpy array: holds the initial guess for std of the Gaussians.
    - ws    : a numpy array: holds the initial guess for weights of the Gaussians.

    * The function should be generic and support any number of Gaussians.
      (you don't need to get this number as a parameter for the function. You can conclude it from the other parameters).

    Returns:
    The output of the EM algorithms (the GMM final parameters): mus, sigmas, ws.
    """

    return mus, sigmas, ws

def q3d(mus, sigmas, ws):
    """

    Input:
    - mus   : a numpy array: holds the means of the gaussians.
    - sigmas: a numpy array: holds the stds of the gaussians.
    - ws    : a numpy array: holds the weights of the gaussians.

    * The function should be generic and support any number of Gaussians.
      (you don't need to get this number as a parameter for the function. You can conclude it from the other parameters).

    Returns:
    The generated data.
    """

    pass


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

    pass

def CC_Variance(N=10):
    """

    Input:
    - N: Number of different coupons.

    Returns:
    V(T_N)
    """

    pass

def CC_T_Steps(N=10, n_steps=30):
    """

    Input:
    - N: Number of different coupons.

    Returns:
    The probability that T_N > n_steps
    """

    pass

def CC_S_Steps(N=10, n_steps=30):
    """

    Input:
    - N: Number of different coupons.

    Returns:
    The probability that S_N > n_steps
    """

    pass

