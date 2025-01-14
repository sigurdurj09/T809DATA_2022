# Author: Sigurður Ágúst Jakobsson
# Date: 25.08.22
# Project: Assignment 2 - K-Nearest Neighbours
# Acknowledgements: 
# Used base code from tools.plot points in knn_plot_points.
# Material and numpy examples from class
# Numpy, scikit and matlib plot docs online, with some generic syntax review online as well.

from turtle import color, distance
import numpy as np
import matplotlib.pyplot as plt

from tools import load_iris, split_train_test, plot_points


def euclidian_distance(x: np.ndarray, y: np.ndarray) -> float:
    '''
    Calculate the euclidian distance between points x and y
    '''
    #Simple enough to put the distance formula in a one liner with array operations.
    return ((x-y)**2).sum()**0.5

def euclidian_distances(x: np.ndarray, points: np.ndarray) -> np.ndarray:
    '''
    Calculate the euclidian distance between x and and many
    points
    '''
    distances = np.zeros(points.shape[0])
    
    for i in range(points.shape[0]):
        distances[i] = euclidian_distance(x, points[i])
    
    return distances

def k_nearest(x: np.ndarray, points: np.ndarray, k: int):
    '''
    Given a feature vector, find the indexes that correspond
    to the k-nearest feature vectors in points
    '''
    distances = euclidian_distances(x, points)
    
    return np.argsort(distances)[:k]

def vote(targets, classes):
    '''
    Given a list of nearest targets, vote for the most
    popular
    '''
    #Instantiate tracking variables
    most_popular = 0
    most_matches = 0
    
    #Iterate the classes and find the one with most matches
    for class_instance in classes:
        matches = (targets == class_instance).sum()
        
        if matches > most_matches:
            most_matches = matches
            most_popular = class_instance

    return most_popular

def knn(
    x: np.ndarray,
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    '''
    Combine k_nearest and vote
    '''
    k_nearest_indexes = k_nearest(x, points, k)
    class_values = point_targets[k_nearest_indexes]
    
    return vote(class_values, classes)

def knn_predict(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:

    #Initialize return vector
    prediction_vector = []
    
    #Concatanate points and test them, adding to vector.
    for index in range(points.shape[0]):
        test_x = points[index]
        test_points = np.concatenate((points[0:index], points[index+1:]))
        test_targets = np.concatenate((point_targets[0:index], point_targets[index+1:]))

        prediction_vector.append(knn(test_x, test_points, test_targets, classes, k))

    return np.array(prediction_vector)

def knn_accuracy(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> float:
    
    #Get prediction
    prediction = knn_predict(points, point_targets, classes, k)
    n = points.shape[0]
    #Compare targets to prediction
    hits = (point_targets == prediction).sum()

    return hits / n

def knn_confusion_matrix(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    
    #Calculate and initiate matrix size from variable class lengths
    D = len(classes)
    confusion_matrix = np.zeros((D,D), dtype=int)

    #Get necessary data
    predictions = knn_predict(points, point_targets, classes, k)

    #Fill in matrix
    for index in range(len(predictions)):
        confusion_matrix[point_targets[index], predictions[index]] += 1    

    #Rows = actual, Columns = predictions - different from example, but there are different axis conventions.
    return confusion_matrix

def best_k(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
) -> int:
    
    #Initialize tracking variables
    best_accuracy = 0
    best_k = 0

    #Try all values of k from 1 to N-1
    for k in range(1, points.shape[0]):
        accuracy = knn_accuracy(points, point_targets, classes, k)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k

    return best_k

def knn_plot_points(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
):
    #Get predictions
    predictions = knn_predict(points, point_targets, classes, k)

    #Standard given code, since it was referenced as the benchmark.
    colors = ['yellow', 'purple', 'blue']
    
    for i in range(points.shape[0]):
        [x, y] = points[i,:2]

        #Customized marking logic
        if point_targets[i] == predictions[i]:
            plt.scatter(x, y, c=colors[point_targets[i]], edgecolors='green', linewidths=2)
        else:
            plt.scatter(x, y, c=colors[point_targets[i]], edgecolors='red', linewidths=2)
    
    plt.title('Yellow=0, Purple=1, Blue=2')
    plt.grid()
    plt.show()

def weighted_vote(
    targets: np.ndarray,
    distances: np.ndarray,
    classes: list
) -> int:
    '''
    Given a list of nearest targets, vote for the most
    popular
    '''
    #Result array
    class_qty = len(classes)
    result_array = np.zeros(class_qty)

    #Vector operations on sums
    c_j = 1 / distances
    sum_c_j = c_j.sum()
    c_j_over_sum = c_j / sum_c_j
    
    #Iterate the classes and find the one with most matches
    for index, class_instance in enumerate(classes):
        weight = ((targets == class_instance)*c_j_over_sum).sum()
        result_array[index] = weight
        
    #Get index of largest weight and return associated class
    return classes[np.argsort(result_array)[-1]]

def wknn(
    x: np.ndarray,
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    '''
    Combine k_nearest and vote
    '''
    k_nearest_indexes = k_nearest(x, points, k)
    class_values = point_targets[k_nearest_indexes]
    distances = euclidian_distances(x, points[k_nearest_indexes])
    
    return weighted_vote(class_values, distances, classes)

def wknn_predict(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    
    #Initialize return vector
    prediction_vector = []
    
    #Concatanate points and test them, adding to vector.
    for index in range(points.shape[0]):
        test_x = points[index]
        test_points = np.concatenate((points[0:index], points[index+1:]))
        test_targets = np.concatenate((point_targets[0:index], point_targets[index+1:]))

        prediction_vector.append(wknn(test_x, test_points, test_targets, classes, k))

    return np.array(prediction_vector)

def compare_knns(
    points: np.ndarray,
    targets: np.ndarray,
    classes: list
):

    k = 30
    n = points.shape[0]
    k_i = np.arange(1, k+1)
    knn_accuracy_vector = np.zeros(k)
    wknn_accuracy_vector = np.zeros(k)

    #Get predictions for different ks
    for k_I in k_i:
        knn_prediction = knn_predict(points, targets, classes, k_I)
        wknn_prediction = wknn_predict(points, targets, classes, k_I)
    
        #Compare targets to predictions
        knn_hits = (targets == knn_prediction).sum()
        wknn_hits = (targets == wknn_prediction).sum()

        knn_accuracy_vector[k_I - 1] = knn_hits / n
        wknn_accuracy_vector[k_I - 1] = wknn_hits / n

    #Plot results
    fig, axis = plt.subplots()
    axis.plot(k_i, knn_accuracy_vector, color='blue', label='Knn')
    axis.plot(k_i, wknn_accuracy_vector, color='red', label='wKnn')
    axis.set(xlabel='k', ylabel='Accuracy (%)', title='Knn vs. wKnn')
    axis.legend()
    axis.grid()

    fig.show()  
