# Author: Sigurður Ágúst Jakobsson
# Date:
# Project: Linear Models for Regression
# Acknowledgements: 
# Given plotting code from project 3 as a base for project 2

# NOTE: Your code should NOT contain any main functions or code that is executed
# automatically.  We ONLY want the functions as stated in the README.md.
# Make sure to comment out or remove all unnecessary code before submitting.

import numpy as np
import matplotlib.pyplot as plt

from tools import load_regression_iris
from scipy.stats import multivariate_normal


def mvn_basis(
    features: np.ndarray,
    mu: np.ndarray,
    sigma: float
) -> np.ndarray:
    '''
    Multivariate Normal Basis Function
    The function transforms (possibly many) data vectors <features>
    to an output basis function output <fi>
    Inputs:
    * features: [NxD] is a data matrix with N D-dimensional
    data vectors.
    * mu: [MxD] matrix of M D-dimensional mean vectors defining
    the multivariate normal distributions.
    * sigma: All normal distributions are isotropic with sigma*I covariance
    matrices (where I is the MxM identity matrix)
    Output:
    * fi - [NxM] is the basis function vectors containing a basis function
    output fi for each data vector x in features
    '''
    #Initialize necesary variables
    N = features.shape[0]
    M = mu.shape[0]
    D = mu.shape[1]
    fi = np.zeros((N, M))
    covar_matrix = np.identity(D) * sigma

    #Calculate Phi[N,M].  It is a pdf output of the points for each mean vector
    #Chapter 5.1 and 5.2
    for point in range(N):
        for basis in range(M):
            fi[point, basis] = multivariate_normal(mean=mu[basis,:], cov=covar_matrix).pdf(features[point,:])
        
    return fi


def _plot_mvn():
    
    #Initialize variables from example
    X, t = load_regression_iris()
    N, D = X.shape

    M, sigma = 10, 10
    mu = np.zeros((M, D))
    
    for i in range(D):
        mmin = np.min(X[i, :])
        mmax = np.max(X[i, :])
        mu[:, i] = np.linspace(mmin, mmax, M)

    #Get output
    fi = mvn_basis(X, mu, sigma)
    
    #Plot each phi over points
    fig, ax = plt.subplots()   
    ax.grid() 

    for phi in range(M):
        lab = 'Phi' + str(phi) + '(x)'
        ax.plot(fi[:,phi], label=lab)

    ax.legend(loc='upper right')
    plt.title('Output of basis functions')
    fig.show()
    input('Press any key to continue and close plot') 


def max_likelihood_linreg(
    fi: np.ndarray,
    targets: np.ndarray,
    lamda: float
) -> np.ndarray:
    '''
    Estimate the maximum likelihood values for the linear model

    Inputs :
    * Fi: [NxM] is the array of basis function vectors
    * t: [Nx1] is the target value for each input in Fi
    * lamda: The regularization constant

    Output: [Mx1], the maximum likelihood estimate of w for the linear model
    '''
    M = fi.shape[1]   

    #Return w_bar from chapter 5.4
    #w_bar = (lambda*I + phi_T * phi)^-1 * phi_T * t
    reg_matrix = np.identity(M) * lamda
    phiTphi = np.matmul(fi.T, fi)
    term1 = np.linalg.inv(reg_matrix + phiTphi) 
    intermediate = np.matmul(term1, fi.T)
    w_bar = np.matmul(intermediate, targets)
    
    return w_bar


def linear_model(
    features: np.ndarray,
    mu: np.ndarray,
    sigma: float,
    w: np.ndarray
) -> np.ndarray:
    '''
    Inputs:
    * features: [NxD] is a data matrix with N D-dimensional data vectors.
    * mu: [MxD] matrix of M D dimensional mean vectors defining the
    multivariate normal distributions.
    * sigma: All normal distributions are isotropic with s*I covariance
    matrices (where I is the MxM identity matrix).
    * w: [Mx1] the weights, e.g. the output from the max_likelihood_linreg
    function.

    Output: [Nx1] The prediction for each data vector in features
    '''
    #Initialize necessary variables
    N = features.shape[0]
    predictions = np.zeros(N) 

    #We are not passed phi
    phi = mvn_basis(features, mu, sigma)

    #Calculate prediction for each feature according to equation in chapter 5.2
    for feature in range(N):
        predictions[feature] = np.matmul(phi[feature,:], w)

    return predictions

def _square_error(y, y_hat):
    return (y - y_hat)**2

def _prediction_accuracy():
    #Initialize variables from example
    X, t = load_regression_iris()
    N, D = X.shape

    M, sigma = 10, 10
    mu = np.zeros((M, D))
    
    for i in range(D):
        mmin = np.min(X[i, :])
        mmax = np.max(X[i, :])
        mu[:, i] = np.linspace(mmin, mmax, M)

    fi = mvn_basis(X, mu, sigma)
    lamda = 0.001
    wml = max_likelihood_linreg(fi, t, lamda) 
    prediction = linear_model(X, mu, sigma, wml)
    
    sq_error = np.zeros(N)

    for point in range(N):
        sq_error[point] = _square_error(t[point], prediction[point])
    
    
    fig, ax = plt.subplots(2, 1)   
    ax[0].grid() 
    ax[1].grid() 

    ax[0].scatter(range(N), prediction, label='Prediction', color='red')
    ax[0].scatter(range(N), t, label='Actual', color='blue')
    ax[0].legend(loc='upper left')
    ax[0].set_title('Accuracy Analysis')
    ax[0].set_ylabel('[cm]')

    ax[1].scatter(range(N), sq_error, label='Squared Error', color='green')    
    ax[1].legend(loc='upper left')
    ax[1].set_xlabel('Data Point')
    ax[1].set_ylabel('[cm^2]')
    
    fig.show()
    input('Press any key to continue and close plot') 

def _prediction_accuracy_indep1():
    #Initialize variables from example
    X, t = load_regression_iris()
    N, D = X.shape

    M, sigma = 10, 10
    mu = np.zeros((M, D))
    
    #Fixed to look and min and max of columns, not rows
    for i in range(D):
        mmin = np.min(X[:, i])
        mmax = np.max(X[:, i])
        mu[:, i] = np.linspace(mmin, mmax, M)

    _plot_mu(mu) 

    fi = mvn_basis(X, mu, sigma)
    lamda = 0.001
    wml = max_likelihood_linreg(fi, t, lamda) 
    prediction = linear_model(X, mu, sigma, wml)
    
    sq_error = np.zeros(N)

    for point in range(N):
        sq_error[point] = _square_error(t[point], prediction[point])
    
    
    fig, ax = plt.subplots(2, 1)   
    ax[0].grid() 
    ax[1].grid() 

    ax[0].scatter(range(N), prediction, label='Prediction', color='red')
    ax[0].scatter(range(N), t, label='Actual', color='blue')
    ax[0].legend(loc='upper left')
    ax[0].set_title('Accuracy Analysis')
    ax[0].set_ylabel('[cm]')

    ax[1].scatter(range(N), sq_error, label='Squared Error', color='green')    
    ax[1].legend(loc='upper left')
    ax[1].set_xlabel('Data Point')
    ax[1].set_ylabel('[cm^2]')
    
    fig.show()
    input('Press any key to continue and close plot') 

def _prediction_accuracy_indep2():
    #Initialize variables from example
    X, t = load_regression_iris()
    N, D = X.shape

    points, sigma = 10, 10
    M = points**D
    feature_space = np.zeros((points, D))
    
    for i in range(D):
        mmin = np.min(X[:, i])
        mmax = np.max(X[:, i])
        feature_space[:, i] = np.linspace(mmin, mmax, points)

    mu = np.zeros((M, D))

    counter = 0

    for i in range(points):
        for j in range(points):
            for k in range(points):
                mu[counter] = np.array([feature_space[i, 0], feature_space[j, 1], feature_space[k, 2]])
                counter+=1

    _plot_mu(mu)

    fi = mvn_basis(X, mu, sigma)
    lamda = 0.001
    wml = max_likelihood_linreg(fi, t, lamda) 
    prediction = linear_model(X, mu, sigma, wml)
    
    sq_error = np.zeros(N)

    for point in range(N):
        sq_error[point] = _square_error(t[point], prediction[point])
    
    
    fig, ax = plt.subplots(2, 1)   
    ax[0].grid() 
    ax[1].grid() 

    ax[0].scatter(range(N), prediction, label='Prediction', color='red')
    ax[0].scatter(range(N), t, label='Actual', color='blue')
    ax[0].legend(loc='upper left')
    ax[0].set_title('Accuracy Analysis')
    ax[0].set_ylabel('[cm]')

    ax[1].scatter(range(N), sq_error, label='Squared Error', color='green')    
    ax[1].legend(loc='upper left')
    ax[1].set_xlabel('Data Point')
    ax[1].set_ylabel('[cm^2]')
    
    fig.show()
    input('Press any key to continue and close plot') 

def _plot_mu(mu):

    xdata = mu[:,0]
    ydata = mu[:,1]
    zdata = mu[:,2]

    ax = plt.axes(projection='3d')
    ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')
    plt.show()


