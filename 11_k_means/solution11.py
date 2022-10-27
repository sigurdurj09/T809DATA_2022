# Author: Sigurður Ágúst Jakobsson
# Date:
# Project: K-means Clustering and Gaussian Mixture Models
# Acknowledgements: 
# SKlearn and numpy documentation.

# NOTE: Your code should NOT contain any main functions or code that is executed
# automatically.  We ONLY want the functions as stated in the README.md.
# Make sure to comment out or remove all unnecessary code before submitting.



from turtle import up
import numpy as np
import sklearn as sk
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from typing import Union

from tools import load_iris, image_to_numpy, plot_gmm_results


def distance_matrix(
    X: np.ndarray,
    Mu: np.ndarray
) -> np.ndarray:
    '''
    Returns a matrix of euclidian distances between points in
    X and Mu.

    Input arguments:
    * X (np.ndarray): A [n x f] array of samples
    * Mu (np.ndarray): A [k x f] array of prototypes

    Returns:
    out (np.ndarray): A [n x k] array of euclidian distances
    where out[i, j] is the euclidian distance between X[i, :]
    and Mu[j, :]
    '''
    
    #Initialize dimensions and return variables
    n = X.shape[0]
    k = Mu.shape[0]
    dist_mat = np.zeros((n, k))

    #Calculate Euclidian distance for all pairs
    for sample in range(n):
        for prototype in range(k):
            #One liner Euclidian distance
            distance = (((X[sample, :] - Mu[prototype])**2).sum())**0.5
            dist_mat[sample, prototype] = distance

    return dist_mat

def determine_r(dist: np.ndarray) -> np.ndarray:
    '''
    Returns a matrix of binary indicators, determining
    assignment of samples to prototypes.

    Input arguments:
    * dist (np.ndarray): A [n x k] array of distances

    Returns:
    out (np.ndarray): A [n x k] array where out[i, j] is
    1 if sample i is closest to prototype j and 0 otherwise.
    '''
    #Initialize dimensions and return variables
    n = dist.shape[0]
    k = dist.shape[1]
    resp_mat = np.zeros((n, k), dtype=int)

    #Iterate samples
    for sample in range(n):
        #Get true for smallest line, multiply true by 1 a
        filt = dist[sample, :] == np.amin(dist[sample, :])
        resp_mat[sample, :] = filt * 1

    return resp_mat

def determine_j(R: np.ndarray, dist: np.ndarray) -> float:
    '''
    Calculates the value of the objective function given
    arrays of indicators and distances.

    Input arguments:
    * R (np.ndarray): A [n x k] array where out[i, j] is
        1 if sample i is closest to prototype j and 0
        otherwise.
    * dist (np.ndarray): A [n x k] array of distances

    Returns:
    * out (float): The value of the objective function
    '''
    #Can do this with matrix algebra rather than 2 summing loops
    #J= sum(N)sum(K) r_nk * ||x_n - Mu_k||**2

    #Oneliner - filter out relevant distances, square them and sum
    return ((R * dist)).sum() / R.shape[0]


def update_Mu(
    Mu: np.ndarray,
    X: np.ndarray,
    R: np.ndarray
) -> np.ndarray:
    '''
    Updates the prototypes, given arrays of current
    prototypes, samples and indicators.

    Input arguments:
    Mu (np.ndarray): A [k x f] array of current prototypes.
    X (np.ndarray): A [n x f] array of samples.
    R (np.ndarray): A [n x k] array of indicators.

    Returns:
    out (np.ndarray): A [k x f] array of updated prototypes.
    '''
    #Initialize dimensions and return variables
    k = Mu.shape[0]
    f = Mu.shape[1]
    nu_mu = np.zeros((k, f), dtype=float)

    #For each mu filter X by assignment and calculate mean
    for m in range(k):
        filter = R[:, m] == 1
        nu_mu[m, :] = (X[filter]).sum(axis=0) / sum(filter)

    return nu_mu

def k_means(
    X: np.ndarray,
    k: int,
    num_its: int
) -> Union[list, np.ndarray, np.ndarray]:
    # We first have to standardize the samples
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_standard = (X-X_mean)/X_std
    # run the k_means algorithm on X_st, not X.

    # we pick K random samples from X as prototypes
    nn = sk.utils.shuffle(range(X_standard.shape[0]))
    Mu = X_standard[nn[0: k], :]

    #Return list
    Js = []
    
    #Iterate over E and M steps
    for i in range(num_its):
        dist = distance_matrix(X_standard, Mu)
        R = determine_r(dist)
        Js.append(determine_j(R, dist))
        Mu = update_Mu(Mu, X_standard, R)

    # Then we have to "de-standardize" the prototypes
    for i in range(k):
        Mu[i, :] = Mu[i, :] * X_std + X_mean

    return Mu, R, Js


def _plot_j():
    
    X, y, c = load_iris()
    its = 10
    Mu, R, Js = k_means(X, 4, its)

    x = list(range(its))

    plt.plot(x, Js)

    plt.xlabel("Iteration")
    plt.ylabel("Objective funtion value [J]")
    plt.title("Plot J")
    plt.grid()
    plt.show()  


def _plot_multi_j():    

    X, y, c = load_iris()
    its = 10
    ks = [2, 3, 5, 10]
    x = list(range(its))

    for k in ks:
        Mu, R, Js = k_means(X, k, its)
        plt.plot(x, Js)    
        #print(Js)

    plt.xlabel("Iteration")
    plt.ylabel("Objective funtion value [J]")
    plt.title("Plot J - By k means")
    plt.legend(ks, title="K")
    plt.grid()
    plt.show()


def k_means_predict(
    X: np.ndarray,
    t: np.ndarray,
    classes: list,
    num_its: int
) -> np.ndarray:
    '''
    Determine the accuracy and confusion matrix
    of predictions made by k_means on a dataset
    [X, t] where we assume the most common cluster
    for a class label corresponds to that label.

    Input arguments:
    * X (np.ndarray): A [n x f] array of input features
    * t (np.ndarray): A [n] array of target labels
    * classes (list): A list of possible target labels
    * num_its (int): Number of k_means iterations

    Returns:
    * the predictions (list)
    '''
    k = 10
    Mu, R, Js = k_means(X, k, num_its)
    predictions = np.zeros((t.shape[0]))

    #Find what class label is most common in each cluster
    for m in range(k):
        filter = R[:, m] == 1
        c = np.bincount(t[filter]).argmax()
        predictions[filter] = c

    return predictions

def _iris_kmeans_accuracy():
    X, y, c = load_iris()
    predictions = k_means_predict(X, y, c, 5)
    
    return accuracy_score(y, predictions)


def _my_kmeans_on_image():
    X, dimensions = image_to_numpy()
    return k_means(X, 7, 5)

def plot_image_clusters(n_clusters: int):
    '''
    Plot the clusters found using sklearn k-means.
    '''
    image, (w, h) = image_to_numpy()
    
    kmeans = KMeans(n_clusters=n_clusters).fit(image)

    plt.subplot(121)
    plt.imshow(image.reshape(w, h, 3))
    plt.subplot(122)
    # uncomment the following line to run
    plt.imshow(kmeans.labels_.reshape(w, h), cmap="plasma")
    plt.show()


def _gmm_info():
    X, y, c = load_iris()
    gmm = GaussianMixture(n_components=3).fit(X)
    print('Mixing coefficients:', gmm.weights_)
    print('Mean vectors:', gmm.means_)
    print('Covariance matrices:', gmm.covariances_)

def _plot_gmm():
    X, y, c = load_iris()
    gmm = GaussianMixture(n_components=3).fit(X)
    predictions = gmm.predict(X)
    means = gmm.means_
    covs = gmm.covariances_
    plot_gmm_results(X, predictions, means, covs)
