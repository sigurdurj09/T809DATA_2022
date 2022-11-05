# Author: Sigurður Ágúst Jakobsson
# Date:
# Project: 12 - Principal Component Analysis
# Acknowledgements: 
# Documentation for sklearn, numpy and matplotlib

# NOTE: Your code should NOT contain any main functions or code that is executed
# automatically.  We ONLY want the functions as stated in the README.md.
# Make sure to comment out or remove all unnecessary code before submitting.



import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from tools import load_cancer


def standardize(X: np.ndarray) -> np.ndarray:
    '''
    Standardize an array of shape [N x 1]

    Input arguments:
    * X (np.ndarray): An array of shape [N x 1]

    Returns:
    (np.ndarray): A standardized version of X, also
    of shape [N x 1]
    '''
    mean = np.mean(X, axis=0)
    sd = np.std(X, axis=0)

    return (X - mean) / sd


def scatter_standardized_dims(
    X: np.ndarray,
    i: int,
    j: int,
):
    '''
    Plots a scatter plot of N points where the n-th point
    has the coordinate (X_ni, X_nj)

    Input arguments:
    * X (np.ndarray): A [N x f] array
    * i (int): The first index
    * j (int): The second index
    '''
    X_hat = standardize(X)
    plt.scatter(X_hat[:, i], X_hat[:, j])

def _scatter_cancer():
    X, y = load_cancer()

    for i in range(30):
        plt.subplot(5, 6, i+1)
        plt.scatter(X[:, 0], X[:, i])

    plt.show()

def _plot_pca_components():
    
    pca = PCA()
    D = 30
    pca.n_components = D
    X, y = load_cancer()
    pca.fit_transform(X)

    components = pca.components_

    print(components)

    for i in range(D):
        plt.subplot(5, 6, i+1)
        plt.plot(components[i, :])
        plt.title("PCA " + str(i+1))
    plt.show()


def _plot_eigen_values():
    
    pca = PCA()
    D = 30
    pca.n_components = D
    X, y = load_cancer()
    pca.fit_transform(X)
    
    x = list(range(1, 31))

    plt.plot(x, pca.explained_variance_)

    plt.xlabel('Eigenvalue index')
    plt.ylabel('Eigenvalue')
    plt.grid()
    plt.show()

def _plot_log_eigen_values():
    
    pca = PCA()
    D = 30
    pca.n_components = D
    X, y = load_cancer()
    pca.fit_transform(X)
    
    x = list(range(1, 31))

    plt.plot(x, np.log10(pca.explained_variance_))

    plt.xlabel('Eigenvalue index')
    plt.ylabel('$\log_{10}$ Eigenvalue')
    plt.grid()
    plt.show()

def _plot_cum_variance():
    
    pca = PCA()
    D = 30
    pca.n_components = D
    X, y = load_cancer()
    pca.fit_transform(X)
    
    x = list(range(1, 31))

    plt.plot(x, np.cumsum(pca.explained_variance_) / np.sum(pca.explained_variance_))

    plt.xlabel('Eigenvalue index')
    plt.ylabel('Percentage variance')
    plt.grid()
    plt.show()
