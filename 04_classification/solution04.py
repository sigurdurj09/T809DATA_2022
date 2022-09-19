# Author: Sigurður Ágúst Jakobsson
# Date: 09.09.22
# Project: Classification Based on Probability
# Acknowledgements: 
# For review of precision and recall: https://towardsdatascience.com/accuracy-recall-precision-f-score-specificity-which-to-optimize-on-867d3f11124
# Code from help to see how SciPy works
# I discussed the project, programming best practices and theory with Gylfi Andrésson and Sigríður Borghildur Jónsdóttir
# Otherwise general tips from class, looking at syntax tips online and in package documentation, and looking at given example code.

from tools import load_iris, split_train_test

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
import sklearn.datasets as datasets


def mean_of_class(
    features: np.ndarray,
    targets: np.ndarray,
    selected_class: int
) -> np.ndarray:
    '''
    Estimate the mean of a selected class given all features
    and targets in a dataset
    '''
    #Simple filter and mean of the columns
    filter = targets == selected_class
    filtered_features = features[filter]

    return filtered_features.mean(0)

def covar_of_class(
    features: np.ndarray,
    targets: np.ndarray,
    selected_class: int
) -> np.ndarray:
    '''
    Estimate the covariance of a selected class given all
    features and targets in a dataset
    '''
    #Simple filter and mean of the columns using numpy
    filter = targets == selected_class
    filtered_features = features[filter]

    return np.cov(filtered_features, rowvar=False)

def likelihood_of_class(
    feature: np.ndarray,
    class_mean: np.ndarray,
    class_covar: np.ndarray
) -> float:
    '''
    Estimate the likelihood that a sample is drawn
    from a multivariate normal distribution, given the mean
    and covariance of the distribution.
    '''
    #Simpy using scipy function with example from help.pdf
    return multivariate_normal(mean=class_mean, cov=class_covar).pdf(feature)

def maximum_likelihood(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    classes: list
) -> np.ndarray:
    '''
    Calculate the maximum likelihood for each test point in
    test_features by first estimating the mean and covariance
    of all classes over the training set.

    You should return
    a [test_features.shape[0] x len(classes)] shaped numpy
    array
    '''
    #Start by using helper functions to get class data
    means, covs = [], []

    for class_label in classes:
        means.append(mean_of_class(train_features, train_targets, class_label))
        covs.append(covar_of_class(train_features, train_targets, class_label))

    #Calculate a likelihood vector for each datapoint i, over each class j.
    likelihoods = []

    for i in range(test_features.shape[0]):
        point_likelihood = []
        for j in range(len(means)):
            point_likelihood.append(likelihood_of_class(test_features[i, :], means[j], covs[j]))

        likelihoods.append(point_likelihood)

    return np.array(likelihoods)


def predict(likelihoods: np.ndarray):
    '''
    Given an array of shape [num_datapoints x num_classes]
    make a prediction for each datapoint by choosing the
    highest likelihood.

    You should return a [likelihoods.shape[0]] shaped numpy
    array of predictions, e.g. [0, 1, 0, ..., 1, 2]
    '''
    class_prediction = []

    #Would be better to take in classes as well
    #to return the class for the index of prediction in edge case where classes aren't from 0.
    for sample in range(likelihoods.shape[0]):
        class_prediction.append(np.argmax(likelihoods[sample,:]))
    
    return np.array(class_prediction)

def maximum_aposteriori(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    classes: list
) -> np.ndarray:
    '''
    Calculate the maximum a posteriori for each test point in
    test_features by first estimating the mean and covariance
    of all classes over the training set.

    You should return
    a [test_features.shape[0] x len(classes)] shaped numpy
    array
    '''
    #Use helper function bases on project 1
    prior_probs = prior(train_targets, classes)
    max_like = maximum_likelihood(train_features, train_targets, test_features, classes)

    #Aposteriori is P{C_k|x} = p(x|C_k)*P{C_k} where the denominator is constant over classes
    #Denominator doesnt matter when maximising over classes.
    return max_like * prior_probs

def prior(targets: np.ndarray, classes: list) -> np.ndarray:
    '''
    Calculate the prior probability of each class type
    given a list of all targets and all class types
    '''
    
    #Make sure that script handles array as np array for bool summation
    targets = np.array(targets)

    #Initialize parameter values for arrays
    target_num = len(targets)
    class_num = len(classes)
    return_array = np.zeros(class_num)

    #Traverse classes
    for index, class_inst in enumerate(classes):
        
        #Create a bool filter and sum hits
        filter = targets[:] == class_inst
        class_count = filter.sum()

        #Add prior propability to return array for each class
        return_array[index] = class_count / target_num

    return return_array  

def accuracy(test_targets, predictions):
    #Compare targets to prediction
    n = predictions.shape[0]
    hits = (test_targets == predictions).sum()

    return hits / n

def confusion_matrix(test_targets, predictions, classes):
    D = len(classes)
    confusion_matrix = np.zeros((D,D), dtype=int)

    for index in range(len(predictions)):
        confusion_matrix[test_targets[index], predictions[index]] += 1    

    #Rows = actual, Columns = predictions - different from example, but there are different axis conventions.
    return confusion_matrix

def gen_data(
    n: int,
    k: int,
    mean: np.ndarray,
    var: float
) -> np.ndarray:
    '''Generate n values samples from the k-variate
    normal distribution
    '''
    cov_identity = np.identity(k) * var**2
    data = np.random.multivariate_normal(mean, cov_identity, n)

    return data

def scatter_2d_alien_data(data: np.ndarray, targets):
    fig = plt.figure()
    ax = fig.add_subplot()

    colors = ['green', 'blue']
    
    for i in range(data.shape[0]):
        [x, y] = data[i,:]
        plt.scatter(x, y, c=colors[targets[i]])
    
    plt.title('Aliens=0/Green, Humans=1/Blue')
    plt.grid()
    ax.set_xlabel('x (weight kg)')
    ax.set_ylabel('y (height cm)')
    plt.show()

def ml_ap_compare(plot=0):
    
    #Try to build unbalanced dataset with outliers
    classes = [0, 1]
    
    n_aliens = 30
    n_male = 100
    n_female = 100
    n_babies = 5 #'outliers'

    alien_features = gen_data(n_aliens, 2, np.array([50, 190]), 9) #Tall green men
    male_features = gen_data(n_male, 2, np.array([85, 180]), 16)
    female_features = gen_data(n_female, 2, np.array([70, 165]), 16)
    baby_features = gen_data(n_babies, 2, np.array([4, 50]), 4)

    features = np.vstack((alien_features, male_features))
    features = np.vstack((features, female_features))
    features = np.vstack((features, baby_features))    

    targets = np.concatenate((np.zeros(n_aliens, dtype=int), np.ones(n_male + n_female + n_babies, dtype=int)), axis=None)
    
    #Plot if relevant
    if plot == 1:
       scatter_2d_alien_data(features, targets) 
    
    (train_features, train_targets), (test_features, test_targets) = split_train_test(features, targets, train_ratio=0.8)

    #Calculations
    likelihoods_ml = maximum_likelihood(train_features, train_targets, test_features, classes)
    likelihoods_ap = maximum_aposteriori(train_features, train_targets, test_features, classes)

    predict_ml = predict(likelihoods_ml)
    predict_ap = predict(likelihoods_ap)

    accuracy_ml = accuracy(test_targets, predict_ml)
    accuracy_ap = accuracy(test_targets, predict_ap)
    #print(accuracy_ml)
    #print(accuracy_ap)

    confusion_ml = confusion_matrix(test_targets, predict_ml, classes)
    confusion_ap = confusion_matrix(test_targets, predict_ap, classes)
    #print(confusion_ml)
    #print(confusion_ap)

    return accuracy_ml, accuracy_ap, confusion_ml, confusion_ap

def multiple_alien_compare():
    
    iter = 0
    runs = 1000
    ml_accuracy_array = []
    ap_accuracy_array = []
    ml_confusion_array = []
    ap_confusion_array = []

    #Run multiple ML/AP comparison tests for statistical comparison
    while iter < runs:
        accuracy_ml, accuracy_ap, confusion_ml, confusion_ap = ml_ap_compare(iter) #Show dist on 2nd run
        ml_accuracy_array.append(accuracy_ml)
        ap_accuracy_array.append(accuracy_ap)
        ml_confusion_array.append(confusion_ml)
        ap_confusion_array.append(confusion_ap)

        iter += 1

    #Statistical analysis
    ml_accuracy_mean = np.mean(ml_accuracy_array)
    ap_accuracy_mean = np.mean(ap_accuracy_array)
    ml_accuracy_sd = np.var(ml_accuracy_array)**0.5
    ap_accuracy_sd = np.var(ap_accuracy_array)**0.5

    #One tailed comparison of results with 95% confidence
    dev_95 = 1.64

    min_ml = ml_accuracy_mean - ml_accuracy_sd * dev_95
    max_ml = ml_accuracy_mean + ml_accuracy_sd * dev_95
    min_ap = ap_accuracy_mean - ap_accuracy_sd * dev_95
    max_ap = ap_accuracy_mean + ap_accuracy_sd * dev_95

    ml_range = [ml_accuracy_mean, max_ml]
    ap_range = [min_ap, ap_accuracy_mean]

    print('ML Mean,Max 95% One Tail:', ml_range)
    print('AP Min,Mean 95% One Tail:', ap_range)

    #Calculate an overall confusion matrix
    average_confusion_ml = np.zeros((2,2))
    average_confusion_ap = np.zeros((2,2))

    for i in range(len(ml_confusion_array)):
        average_confusion_ml += ml_confusion_array[i]
        average_confusion_ap += ap_confusion_array[i]

    average_confusion_ml /= len(ml_confusion_array)
    average_confusion_ap /= len(ap_confusion_array)

    print('ML Confusion\n', average_confusion_ml)
    print('AP Confusion\n', average_confusion_ap)

    #Analyze overall precision and recall

    ml_precision = average_confusion_ml[0,0] / (average_confusion_ml[0,0] + average_confusion_ml[1,0])
    ap_precision = average_confusion_ap[0,0] / (average_confusion_ap[0,0] + average_confusion_ap[1,0])

    print('ML Precision:', ml_precision)
    print('AP Precision:', ap_precision)

    ml_recall = average_confusion_ml[0,0] / (average_confusion_ml[0,0] + average_confusion_ml[0,1])
    ap_recall = average_confusion_ap[0,0] / (average_confusion_ap[0,0] + average_confusion_ap[0,1])

    print('ML Recall:', ml_recall)
    print('AP Recall:', ap_recall)