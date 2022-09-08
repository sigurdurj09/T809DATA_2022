# Author: Sigurður Ágúst Jakobsson
# Date:
# Project: Sequential Estimation
# Acknowledgements: 
#


from tools import scatter_3d_data, bar_per_axis

import matplotlib.pyplot as plt
import numpy as np


def gen_data(
    n: int,
    k: int,
    mean: np.ndarray,
    var: float
) -> np.ndarray:
    '''Generate n values samples from the k-variate
    normal distribution
    '''
    #Simple usage of NP according to instructions
    cov_identity = np.identity(k) * var**2
    data = np.random.multivariate_normal(mean, cov_identity, n)

    return data

def update_sequence_mean(
    mu: np.ndarray,
    x: np.ndarray,
    n: int
) -> np.ndarray:
    '''Performs the mean sequence estimation update
    '''
    #Formula from book
    mu_new = mu + 1/(n+1)*(x-mu)

    return mu_new


def _plot_sequence_estimate():
    
    data = gen_data(100, 3, np.array([0, 1, -1]), 3**0.5)
    estimates = [np.array([0, 0, 0])]
    
    #Create estimates vector to plot
    for i in range(data.shape[0]):
        estimates.append(update_sequence_mean(estimates[i], data[i], len(estimates)))

    fig, ax = plt.subplots()   
    ax.grid() 
    ax.plot([e[0] for e in estimates], label='First dimension')
    ax.plot([e[1] for e in estimates], label='Second dimension')
    ax.plot([e[2] for e in estimates], label='Third dimension')
    ax.legend(loc='upper center')
    fig.show() 
      

def _square_error(y, y_hat):
    return (y - y_hat)**2


def _plot_mean_square_error():
    y = np.array([0, 1, -1])
    data = gen_data(100, 3, y, 3**0.5)
    estimates = [np.array([0, 0, 0])]
    square_errors = []
    
    for i in range(data.shape[0]):
        estimates.append(update_sequence_mean(estimates[i], data[i], len(estimates)))
        square_errors.append(_square_error(y, estimates[i+1]).mean())
    
    fig, ax = plt.subplots()   
    ax.plot(square_errors)
    ax.grid() 
    fig.show()    
    
# Naive solution to the independent question.

def update_sequence_mean_Mpoint_moving_avg(
    mu: np.ndarray,
    x: np.ndarray,
    n: int,
    m: int
) -> np.ndarray:
    '''Performs the mean sequence estimation update
    '''
    #Approximate moving average
    #Lowering n to a constant after x time gives more weight to newer points
    if n > m:
        n = m

    #Formula from book
    mu_new = mu + 1/(n+1)*(x-mu)

    return mu_new

def gen_changing_data(
    n: int,
    k: int,
    start_mean: np.ndarray,
    end_mean: np.ndarray,
    var: float
) -> np.ndarray:

    ret_array = []
    mean_matrix = np.zeros((n,k))
    data_matrix = np.zeros((n,k))

    for dim in range(k):
        mean_matrix[:,dim] = np.linspace(start_mean[dim], end_mean[dim], num=n).T

    for sample in range(n):
        data_matrix[sample,:] = gen_data(1, 3, mean_matrix[sample,:], var)

    #Return y and data to compare later
    ret_array.append(mean_matrix)
    ret_array.append(data_matrix)

    return np.array(ret_array)

def _plot_changing_sequence_estimate():
    data = gen_changing_data(500, 3, np.array([0, 1, -1]), np.array([1, -1, 0]), 3**0.5)
    estimates = [np.array([0, 0, 0])]
    
    gen_data = data[1]

    #Create estimates vector to plot - find a good m
    for i in range(gen_data.shape[0]):
        estimates.append(update_sequence_mean_Mpoint_moving_avg(estimates[i], gen_data[i], len(estimates),50))

    fig, ax = plt.subplots()   
    ax.grid() 
    ax.plot([e[0] for e in estimates], label='First dimension')
    ax.plot([e[1] for e in estimates], label='Second dimension')
    ax.plot([e[2] for e in estimates], label='Third dimension')
    ax.legend(loc='upper center')
    fig.show()

    #Move to squared errors
    y = data[0]

    square_errors = []
    
    for i in range(y.shape[0]):
        square_errors.append(_square_error(y[i,:], estimates[i+1]).mean())
    
    fig, ax = plt.subplots()   
    ax.plot(square_errors)
    ax.grid() 
    fig.show()    
   
