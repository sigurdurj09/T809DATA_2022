# Author: Sigurður Ágúst Jakobsson
# Date:
# Project: Assignment 1 - Decision Trees
# Acknowledgements: 
#


from typing import Union
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree

from tools import load_iris, split_train_test

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

def split_data(
    features: np.ndarray,
    targets: np.ndarray,
    split_feature_index: int,
    theta: float
) -> Union[tuple, tuple]:
    '''
    Split a dataset and targets into two seperate datasets
    where data with split_feature < theta goes to 1 otherwise 2
    '''

    #Calculate reusable filters
    filter_1 = features[:,split_feature_index] < theta
    filter_2 = features[:,split_feature_index] >= theta

    #Use the bool arrays to filter the data arrays
    features_1 = features[filter_1]
    targets_1 = targets[filter_1]

    features_2 = features[filter_2]
    targets_2 = targets[filter_2]

    return (features_1, targets_1), (features_2, targets_2)

def gini_impurity(targets: np.ndarray, classes: list) -> float:
    '''
    Calculate:
        i(S_k) = 1/2 * (1 - sum_i P{C_i}**2)
    '''

    #Calculate all prior probabilities, then square array and sum.
    apriori_array = prior(targets, classes)
    apriori_squared_sum = (apriori_array**2).sum()

    return 0.5 * (1-apriori_squared_sum)

def weighted_impurity(
    t1: np.ndarray,
    t2: np.ndarray,
    classes: list
) -> float:
    '''
    Given targets of two branches, return the weighted
    sum of gini branch impurities
    '''
    g1 = gini_impurity(t1, classes)
    g2 = gini_impurity(t2, classes)
    n = t1.shape[0] + t2.shape[0]
    
    return t1.shape[0] * g1 / n + t2.shape[0] * g2 / n

def total_gini_impurity(
    features: np.ndarray,
    targets: np.ndarray,
    classes: list,
    split_feature_index: int,
    theta: float
) -> float:
    '''
    Calculate the gini impurity for a split on split_feature_index
    for a given dataset of features and targets.
    '''
    #Just encapsulating prior functions.
    (f_1, t_1), (f_2, t_2) = split_data(features, targets, split_feature_index, theta)
    return weighted_impurity(t_1, t_2, classes)

def brute_best_split(
    features: np.ndarray,
    targets: np.ndarray,
    classes: list,
    num_tries: int
) -> Union[float, int, float]:
    '''
    Find the best split for the given data. Test splitting
    on each feature dimension num_tries times.

    Return the lowest gini impurity, the feature dimension and
    the threshold
    '''
    best_gini, best_dim, best_theta = float("inf"), None, None
    # iterate feature dimensions
    for i in range(features.shape[1]):
        # create the thresholds - from example.py code
        min_value = features[:,i].min()
        max_value = features[:,i].max()
        thetas = np.linspace(min_value, max_value, num_tries+2)[1:-1]
        
        # iterate thresholds
        for theta in thetas:
            gini = total_gini_impurity(features, targets, classes, i, theta)
            
            if gini < best_gini:
                best_gini = gini
                best_dim = i
                best_theta = theta

    return best_gini, best_dim, best_theta

class IrisTreeTrainer:
    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        classes: list = [0, 1, 2],
        train_ratio: float = 0.8
    ):
        '''
        train_ratio: The ratio of the Iris dataset that will
        be dedicated to training.
        '''
        (self.train_features, self.train_targets),\
            (self.test_features, self.test_targets) =\
            split_train_test(features, targets, train_ratio)

        self.classes = classes
        self.tree = DecisionTreeClassifier()

    def train(self):
        self.tree.fit(self.train_features, self.train_targets)

    def accuracy(self):
        return self.tree.score(self.test_features, self.test_targets)

"""

    def plot(self):
        ...

    def plot_progress(self):
        # Independent section
        # Remove this method if you don't go for independent section.
        ...

    def guess(self):
        ...

    def confusion_matrix(self):
        ...
"""