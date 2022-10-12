# Author: Sigurður Ágúst Jakobsson
# Date: 09.10.22
# Project: Support Vector Machines
# Acknowledgements: 
# Documentation of sklearn modules

# NOTE: Your code should NOT contain any main functions or code that is executed
# automatically.  We ONLY want the functions as stated in the README.md.
# Make sure to comment out or remove all unnecessary code before submitting.


from tools import plot_svm_margin, load_cancer
from sklearn import svm
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score, precision_score, recall_score

import numpy as np
import matplotlib.pyplot as plt


def _plot_linear_kernel(print_stats=False):
    X, t = make_blobs(n_samples=40, centers=2)
    clf = svm.SVC(kernel='linear', C=1000)
    clf.fit(X, t)    

    if print_stats:
        print('Number of SVs:', clf.n_support_)

    plot_svm_margin(clf, X, t)

def _subplot_svm_margin(
    svc,
    X: np.ndarray,
    t: np.ndarray,
    num_plots: int,
    index: int
):
    '''
    Plots the decision boundary and decision margins
    for a dataset of features X and labels t and a support
    vector machine svc.

    Input arguments:
    * svc: An instance of sklearn.svm.SVC: a C-support Vector
    classification model
    * X: [N x f] array of features
    * t: [N] array of target labels
    '''
    # similar to tools.plot_svm_margin but added num_plots and
    # index where num_plots should be the total number of plots
    # and index is the index of the current plot being generated
    
    ax = plt.subplot(1, num_plots, index)

    ax.scatter(X[:, 0], X[:, 1], c=t, s=30, cmap=plt.cm.Paired)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T

    Z = svc.decision_function(xy).reshape(XX.shape)

    # plot decision boundary and margins
    ax.contour(
        XX, YY, Z,
        colors='k', levels=[-1, 0, 1],
        alpha=0.5, linestyles=['--', '-', '--'])

    # plot support vectors
    ax.scatter(
        svc.support_vectors_[:, 0],
        svc.support_vectors_[:, 1],
        s=100, linewidth=1, facecolors='none', edgecolors='k')


def _compare_gamma(print_stats=False):

    X, t = make_blobs(n_samples=40, centers=2, random_state=6)

    clf = svm.SVC(kernel='rbf', C=1000)
    clf.fit(X, t)
    _subplot_svm_margin(clf, X, t, 3, 1)

    if print_stats:
        print('Number of SVs Gamma-standard:', clf.n_support_)

    clf = svm.SVC(kernel='rbf', C=1000, gamma=0.2)
    clf.fit(X, t)
    _subplot_svm_margin(clf, X, t, 3, 2)

    if print_stats:
        print('Number of SVs Gamma-0.2:', clf.n_support_)

    clf = svm.SVC(kernel='rbf', C=1000, gamma=2.0)
    clf.fit(X, t)
    _subplot_svm_margin(clf, X, t, 3, 3)

    if print_stats:
        print('Number of SVs Gamma-2.0:', clf.n_support_)

    plt.show()


def _compare_C(print_stats=False):
    X, t = make_blobs(n_samples=40, centers=2)
    
    clf = svm.SVC(kernel='linear', C=1000)
    clf.fit(X, t)
    _subplot_svm_margin(clf, X, t, 5, 1)

    if print_stats:
        print('Number of SVs C-1000:', clf.n_support_)

    clf = svm.SVC(kernel='linear', C=0.5)
    clf.fit(X, t)
    _subplot_svm_margin(clf, X, t, 5, 2)

    if print_stats:
        print('Number of SVs C-0.5:', clf.n_support_)

    clf = svm.SVC(kernel='linear', C=0.3)
    clf.fit(X, t)
    _subplot_svm_margin(clf, X, t, 5, 3)

    if print_stats:
        print('Number of SVs C-0.3:', clf.n_support_)

    clf = svm.SVC(kernel='linear', C=0.05)
    clf.fit(X, t)
    _subplot_svm_margin(clf, X, t, 5, 4)

    if print_stats:
        print('Number of SVs C-0.05:', clf.n_support_)

    clf = svm.SVC(kernel='linear', C=0.0001)
    clf.fit(X, t)
    _subplot_svm_margin(clf, X, t, 5, 5)

    if print_stats:
        print('Number of SVs C-0.0001:', clf.n_support_)

    plt.show()

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

def train_test_SVM(
    svc,
    X_train: np.ndarray,
    t_train: np.ndarray,
    X_test: np.ndarray,
    t_test: np.ndarray,
):
    '''
    Train a configured SVM on <X_train> and <t_train>
    and then measure accuracy, precision and recall on
    the test set

    This function should return (accuracy, precision, recall)
    '''
    svc.fit(X_train, t_train)
    predictions = svc.predict(X_test)

    acc = accuracy(t_test, predictions)
    conf_matrix = confusion_matrix(t_test, predictions, [0, 1])

    precision = conf_matrix[1,1] / (conf_matrix[1,1] + conf_matrix[0,1])
    recall = conf_matrix[1,1] / (conf_matrix[1,1] + conf_matrix[1,0])

    return (acc, precision, recall)

def SVM_compare(prt=False):
    (X_train, t_train), (X_test, t_test) = load_cancer()

    svc = svm.SVC(kernel='linear', C=1000)
    linstats = train_test_SVM(svc, X_train, t_train, X_test, t_test)
    if prt:
        print('Linear stats', linstats)

    svc = svm.SVC(kernel='rbf', C=1000)
    rbfstats = train_test_SVM(svc, X_train, t_train, X_test, t_test)
    if prt:
        print('Radial stats', rbfstats)

    svc = svm.SVC(kernel='poly', C=1000)
    polystats = train_test_SVM(svc, X_train, t_train, X_test, t_test)
    if prt:
        print('Poly stats', polystats)

    return linstats, rbfstats, polystats

def mult_SVM_compare():
    N = 100

    #Data containers
    lin_acc = []
    lin_prec = []
    lin_rec = []
    rbf_acc = []
    rbf_prec = []
    rbf_rec = []
    poly_acc = []
    poly_prec = []
    poly_rec = []

    #Multiple tests
    for i in range(N):
        linstats, rbfstats, polystats = SVM_compare()
        lin_acc.append(linstats[0])
        lin_prec.append(linstats[1])
        lin_rec.append(linstats[2])
        rbf_acc.append(rbfstats[0])
        rbf_prec.append(rbfstats[1])
        rbf_rec.append(rbfstats[2])
        poly_acc.append(polystats[0])
        poly_prec.append(polystats[1])
        poly_rec.append(polystats[2])
        if i % 5 == 0:
            print(i+1)

    #Statistical analysis
    lin_acc_mean = np.mean(lin_acc)
    lin_acc_sd = np.var(lin_acc) ** 0.5
    lin_prec_mean = np.mean(lin_prec)
    lin_prec_sd = np.var(lin_prec) ** 0.5
    lin_rec_mean = np.mean(lin_rec)
    lin_rec_sd = np.var(lin_rec) ** 0.5

    rbf_acc_mean = np.mean(rbf_acc)
    rbf_acc_sd = np.var(rbf_acc) ** 0.5
    rbf_prec_mean = np.mean(rbf_prec)
    rbf_prec_sd = np.var(rbf_prec) ** 0.5
    rbf_rec_mean = np.mean(rbf_rec)
    rbf_rec_sd = np.var(rbf_rec) ** 0.5

    poly_acc_mean = np.mean(poly_acc)
    poly_acc_sd = np.var(poly_acc) ** 0.5
    poly_prec_mean = np.mean(poly_prec)
    poly_prec_sd = np.var(poly_prec) ** 0.5
    poly_rec_mean = np.mean(poly_rec)
    poly_rec_sd = np.var(poly_rec) ** 0.5

    #2 tailed tests 95%
    print('Linear accuracy 95%', [lin_acc_mean-1.96*lin_acc_sd, lin_acc_mean, lin_acc_mean+1.96*lin_acc_sd])
    print('Linear precision 95%', [lin_prec_mean-1.96*lin_prec_sd, lin_prec_mean, lin_prec_mean+1.96*lin_prec_sd])
    print('Linear recall 95%', [lin_rec_mean-1.96*lin_rec_sd, lin_rec_mean, lin_rec_mean+1.96*lin_rec_sd])
    print('RBF accuracy 95%', [rbf_acc_mean-1.96*rbf_acc_sd, rbf_acc_mean, rbf_acc_mean+1.96*rbf_acc_sd])
    print('RBF precision 95%', [rbf_prec_mean-1.96*rbf_prec_sd, rbf_prec_mean, rbf_prec_mean+1.96*rbf_prec_sd])
    print('RBF recall 95%', [rbf_rec_mean-1.96*rbf_rec_sd, rbf_rec_mean, rbf_rec_mean+1.96*rbf_rec_sd])
    print('Poly accuracy 95%', [poly_acc_mean-1.96*poly_acc_sd, poly_acc_mean, poly_acc_mean+1.96*poly_acc_sd])
    print('Poly precision 95%', [poly_prec_mean-1.96*poly_prec_sd, poly_prec_mean, poly_prec_mean+1.96*poly_prec_sd])
    print('Poly recall 95%', [poly_rec_mean-1.96*poly_rec_sd, poly_rec_mean, poly_rec_mean+1.96*poly_rec_sd])

def bball_data_analysis():
    pass