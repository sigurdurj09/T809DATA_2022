# Author: Sigurður Ágúst Jakobsson
# Date: 25.09.22
# Project: The Back-propagation Algorithm
# Acknowledgements: 
# Wolfram Alpha for some readup on hyperbolic tangent
# I discussed the project, programming best practices and theory with Gylfi Andrésson
# Otherwise general tips from class, looking at syntax tips online and in package documentation, and looking at given example code.


from tkinter import E
from typing import Union
import numpy as np
import tools
import matplotlib.pyplot as plt

from tools import load_iris, split_train_test


def sigmoid(x: float) -> float:
    '''
    Calculate the sigmoid of x
    '''
    if x < -100:
        sigmoid = 0.0
    else:
        sigmoid = 1 / (1 + np.exp(-x))

    return sigmoid


def d_sigmoid(x: float) -> float:
    '''
    Calculate the derivative of the sigmoid of x.
    '''
    return sigmoid(x) * (1-sigmoid(x))


def perceptron(
    x: np.ndarray,
    w: np.ndarray
) -> Union[float, float]:
    '''
    Return the weighted sum of x and w as well as
    the result of applying the sigmoid activation
    to the weighted sum
    '''
    a = w.T @ x
    z = sigmoid(a)

    return a, z


def ffnn(
    x: np.ndarray,
    M: int,
    K: int,
    W1: np.ndarray,
    W2: np.ndarray,
) -> Union[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Computes the output and hidden layer variables for a
    single hidden layer feed-forward neural network.
    '''
    z0 = np.insert(x, 0, 1, axis=0)
    z1 = []
    a1 = []
    a2 = []
    y = []

    for neuron in range(M):
        a, z = perceptron(z0, W1[:, neuron])
        a1.append(a)
        z1.append(z)

    z1 = np.insert(z1, 0, 1, axis=0)

    for neuron in range(K):
        a, z = perceptron(z1, W2[:, neuron])
        a2.append(a)
        y.append(z)


    return np.array(y), np.array(z0), np.array(z1), np.array(a1), np.array(a2)    


def backprop(
    x: np.ndarray,
    target_y: np.ndarray,
    M: int,
    K: int,
    W1: np.ndarray,
    W2: np.ndarray
) -> Union[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Perform the backpropagation on given weights W1 and W2
    for the given input pair x, target_y
    '''
    y, z0, z1, a1, a2 = ffnn(x, M, K, W1, W2)

    d_k = y - target_y
    d_j = np.zeros(a1.shape[0])

    #Try my best to follow the formula
    for j in range(len(d_j)):
        #d_sigmoid of node * sum(weight of node to next node * error of next node)
        d_j[j] = d_sigmoid(a1[j]) * (W2[j+1, :] * d_k).sum()

    dE1 = np.zeros((W1.shape[0], W1.shape[1]))
    dE2 = np.zeros((W2.shape[0], W2.shape[1]))

    #Relatively simple 2D loops to fill in matrices according to d_x*z_y

    #Start using K at output
    for k in range(len(d_k)):
        for j in range(len(z1)):
            dE2[j, k] = d_k[k] * z1[j]

    #Go to earlier layer
    for j in range(len(d_j)):
        for i in range(len(z0)):
            dE1[i, j] = d_j[j] * z0[i]

    return np.array(y), np.array(dE1), np.array(dE2)

def train_nn(
    X_train: np.ndarray,
    t_train: np.ndarray,
    M: int,
    K: int,
    W1: np.ndarray,
    W2: np.ndarray,
    iterations: int,
    eta: float
) -> Union[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Train a network by:
    1. forward propagating an input feature through the network
    2. Calculate the error between the prediction the network
    made and the actual target
    3. Backpropagating the error through the network to adjust
    the weights.
    '''
    W1tr = np.zeros((W1.shape[0], W1.shape[1]))
    W2tr = np.zeros((W2.shape[0], W2.shape[1]))
    E_total = []
    misclassification_rate = []
    

    for it in range(iterations):
        dE1_total = np.zeros((W1.shape[0], W1.shape[1]))
        dE2_total = np.zeros((W2.shape[0], W2.shape[1]))
        guesses = []
        error = 0

        for point in range(X_train.shape[0]):
            #Put target in vector so it is handled right
            target = np.zeros(K)
            target[t_train[point]] = 1.0
            #Backpropogate
            y, dE1, dE2 = backprop(X_train[point], target, M, K, W1, W2)
            guess_index = np.argmax(y)
            guesses.append(guess_index)
            dE1_total += dE1
            dE2_total += dE2
            error -= (target * np.log(y)).sum() + ((1 - target) * np.log(1 - y)).sum()

        W1 = W1 - eta * dE1_total / X_train.shape[0]
        W2 = W2 - eta * dE2_total / X_train.shape[0]
        misclassification_rate.append((t_train != guesses).sum() /  t_train.shape[0])
        E_total.append(error / X_train.shape[0])

    W1tr = W1
    W2tr = W2
    
    return np.array(W1tr), np.array(W2tr), np.array(E_total), np.array(misclassification_rate), np.array(guesses)


def test_nn(
    X: np.ndarray,
    M: int,
    K: int,
    W1: np.ndarray,
    W2: np.ndarray
) -> np.ndarray:
    '''
    Return the predictions made by a network for all features
    in the test set X.
    '''
    guesses = np.zeros(X.shape[0],dtype=int)
    
    for feature in range(X.shape[0]):
       y, z0, z1, a1, a2 = ffnn(X[feature], M, K, W1, W2) 
       guesses[feature] = np.argmax(y)

    return guesses

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

def train_test_plot():
    features, targets, classes = tools.load_iris()
    (train_features, train_targets), (test_features, test_targets) = tools.split_train_test(features, targets,train_ratio=0.8)
   
    K = len(classes)  # number of classes
    M = 6
    D = train_features.shape[1]

    W1 = 2 * np.random.rand(D + 1, M) - 1
    W2 = 2 * np.random.rand(M + 1, K) - 1

    W1tr, W2tr, Etotal, misclassification_rate, last_guesses = train_nn(train_features, train_targets, M, K, W1, W2, 500, 0.1)
    guesses = test_nn(test_features, M, K, W1tr, W2tr)

    train_accuracy = accuracy(train_targets, last_guesses)
    test_accuracy = accuracy(test_targets, guesses)

    train_confusion = confusion_matrix(train_targets, last_guesses, classes)
    test_confusion = confusion_matrix(test_targets, guesses, classes)

    print('Train Accuracy:', train_accuracy)
    print('Test Accuracy:', test_accuracy)
    print('Train Confusion Matrix:\n', train_confusion)
    print('Test Confusion Matrix:\n', test_confusion)

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(Etotal)    
    plt.title('Error per training itertation')
    plt.grid()
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Total Error')
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(misclassification_rate)    
    plt.title('Misclassification per training itertation')
    plt.grid()
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Misclassification rate')
    plt.show()

#Independent

def relu(x: float) -> float:
    '''
    Calculate the relu of x
    '''
    if x >= 0:
        relu = x
    else:
        relu = 0

    return relu


def d_relu(x: float) -> float:
    '''
    Calculate the derivative of the relu of x.
    '''
    if x >= 0:
        d_relu = 1
    else:
        d_relu = 0

    return d_relu

def unit_step(x: float) -> float:
    '''
    Calculate the unit step of x
    '''
    if x >= 0:
        us = 1
    else:
        us = 0

    return us


def d_unit_step(x: float) -> float:
    '''
    Calculate the derivative of the unit step of x.
    '''
    return 0

def tanh(x: float) -> float:
    '''
    Calculate the hyperbolic tangent of x
    '''
    #https://www.wolframalpha.com/input?i=tanh
    return (np.exp(2*x) - 1) / (np.exp(2*x) + 1)


def d_tanh(x: float) -> float:
    '''
    Calculate the derivative of the hyperbolic tangent of x.
    '''
    #https://www.wolframalpha.com/input?i=d%2Fdx+tanh(x)
    return 4 / (np.exp(-x) + np.exp(x))**2 

def perceptron_mode(
    x: np.ndarray,
    w: np.ndarray,
    mode: str
) -> Union[float, float]:
    '''
    Return the weighted sum of x and w as well as
    the result of applying the sigmoid activation
    to the weighted sum
    '''
    a = w.T @ x

    if mode == 'sigmoid':
        z = sigmoid(a)
    elif mode == 'relu':
        z = relu(a)
    elif mode == 'US':
        z = unit_step(a)
    elif mode == 'tanh':
        z = tanh(a)
    else:
        #Default
        z = sigmoid(a)

    return a, z


def ffnn_mode(
    x: np.ndarray,
    M: int,
    K: int,
    W1: np.ndarray,
    W2: np.ndarray,
    mode: str
) -> Union[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Computes the output and hidden layer variables for a
    single hidden layer feed-forward neural network.
    '''
    z0 = np.insert(x, 0, 1, axis=0)
    z1 = []
    a1 = []
    a2 = []
    y = []

    for neuron in range(M):
        a, z = perceptron_mode(z0, W1[:, neuron], mode)
        a1.append(a)
        z1.append(z)

    z1 = np.insert(z1, 0, 1, axis=0)

    for neuron in range(K):
        a, z = perceptron_mode(z1, W2[:, neuron], mode)
        a2.append(a)
        y.append(z)


    return np.array(y), np.array(z0), np.array(z1), np.array(a1), np.array(a2)    


def backprop_mode(
    x: np.ndarray,
    target_y: np.ndarray,
    M: int,
    K: int,
    W1: np.ndarray,
    W2: np.ndarray,
    mode: str
) -> Union[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Perform the backpropagation on given weights W1 and W2
    for the given input pair x, target_y
    '''
    y, z0, z1, a1, a2 = ffnn_mode(x, M, K, W1, W2, mode)

    d_k = y - target_y
    d_j = np.zeros(a1.shape[0])

    #Try my best to follow the formula
    for j in range(len(d_j)):
        if mode == 'sigmoid':
            d_j[j] = d_sigmoid(a1[j]) * (W2[j+1, :] * d_k).sum() 
        elif mode == 'relu':
            d_j[j] = d_relu(a1[j]) * (W2[j+1, :] * d_k).sum()
        elif mode == 'US':
            d_j[j] = d_unit_step(a1[j]) * (W2[j+1, :] * d_k).sum()
        elif mode == 'tanh':
            d_j[j] = d_tanh(a1[j]) * (W2[j+1, :] * d_k).sum()
        else:
            #Default
            d_j[j] = d_sigmoid(a1[j]) * (W2[j+1, :] * d_k).sum()      

    dE1 = np.zeros((W1.shape[0], W1.shape[1]))
    dE2 = np.zeros((W2.shape[0], W2.shape[1]))

    #Relatively simple 2D loops to fill in matrices according to d_x*z_y

    #Start using K at output
    for k in range(len(d_k)):
        for j in range(len(z1)):
            dE2[j, k] = d_k[k] * z1[j]

    #Go to earlier layer
    for j in range(len(d_j)):
        for i in range(len(z0)):
            dE1[i, j] = d_j[j] * z0[i]

    return np.array(y), np.array(dE1), np.array(dE2)

def train_nn_mode(
    X_train: np.ndarray,
    t_train: np.ndarray,
    M: int,
    K: int,
    W1: np.ndarray,
    W2: np.ndarray,
    iterations: int,
    eta: float,
    mode: str
) -> Union[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Train a network by:
    1. forward propagating an input feature through the network
    2. Calculate the error between the prediction the network
    made and the actual target
    3. Backpropagating the error through the network to adjust
    the weights.
    '''
    W1tr = np.zeros((W1.shape[0], W1.shape[1]))
    W2tr = np.zeros((W2.shape[0], W2.shape[1]))
    E_total = []
    misclassification_rate = []
    

    for it in range(iterations):
        dE1_total = np.zeros((W1.shape[0], W1.shape[1]))
        dE2_total = np.zeros((W2.shape[0], W2.shape[1]))
        guesses = []
        error = 0

        for point in range(X_train.shape[0]):
            #Put target in vector so it is handled right
            target = np.zeros(K)
            target[t_train[point]] = 1.0
            #Backpropogate
            y, dE1, dE2 = backprop_mode(X_train[point], target, M, K, W1, W2, mode)
            guess_index = np.argmax(y)
            guesses.append(guess_index)
            dE1_total += dE1
            dE2_total += dE2
            error -= (target * np.log(y)).sum() + ((1 - target) * np.log(1 - y)).sum()

        W1 = W1 - eta * dE1_total / X_train.shape[0]
        W2 = W2 - eta * dE2_total / X_train.shape[0]
        misclassification_rate.append((t_train != guesses).sum() /  t_train.shape[0])
        E_total.append(error / X_train.shape[0])

    W1tr = W1
    W2tr = W2
    
    return np.array(W1tr), np.array(W2tr), np.array(E_total), np.array(misclassification_rate), np.array(guesses)

def test_nn_mode(
    X: np.ndarray,
    M: int,
    K: int,
    W1: np.ndarray,
    W2: np.ndarray,
    mode: str
) -> np.ndarray:
    '''
    Return the predictions made by a network for all features
    in the test set X.
    '''
    guesses = np.zeros(X.shape[0],dtype=int)
    
    for feature in range(X.shape[0]):
       y, z0, z1, a1, a2 = ffnn_mode(X[feature], M, K, W1, W2, mode) 
       guesses[feature] = np.argmax(y)

    return guesses

def train_test_plot_mode():
    modes = ['sigmoid', 'relu', 'US', 'tanh']

    features, targets, classes = tools.load_iris()
    (train_features, train_targets), (test_features, test_targets) = tools.split_train_test(features, targets,train_ratio=0.8)
   
    K = len(classes)  # number of classes
    M = 6
    D = train_features.shape[1]

    W1 = 2 * np.random.rand(D + 1, M) - 1
    W2 = 2 * np.random.rand(M + 1, K) - 1

    for mode in modes:

        W1_use = W1
        W2_use = W2

        if mode in ['relu', 'US']:
            W1_use = (W1 + 1) / 2
            W2_use = (W2 + 1) / 2

        W1tr, W2tr, Etotal, misclassification_rate, last_guesses = train_nn_mode(train_features, train_targets, M, K, W1_use, W2_use, 500, 0.1, mode)
        guesses = test_nn_mode(test_features, M, K, W1tr, W2tr, mode)

        train_accuracy = accuracy(train_targets, last_guesses)
        test_accuracy = accuracy(test_targets, guesses)

        train_confusion = confusion_matrix(train_targets, last_guesses, classes)
        test_confusion = confusion_matrix(test_targets, guesses, classes)

        print('Train Accuracy', mode, ':', train_accuracy)
        print('Test Accuracy', mode, ':', test_accuracy)
        print('Train Confusion Matrix:', mode, '\n', train_confusion)
        print('Test Confusion Matrix:', mode, '\n', test_confusion)

        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot(Etotal)    
        plt.title('Error per training itertation:: ' + mode)
        plt.grid()
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Total Error')
        plt.show()

        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot(misclassification_rate)    
        plt.title('Misclassification per training itertation: ' + mode)
        plt.grid()
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Misclassification rate')
        plt.show()





    
