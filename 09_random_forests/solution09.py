# Author: Sigurður Ágúst Jakobsson
# Date: 23.10.22
# Project: Random Forests
# Acknowledgements: 
# Sklearn, matplotlib and other library documentation online
# Discussed solution strategies with Gylfi Andrésson
# Used split_train_test from previous assignments tools as help for independent section.

# NOTE: Your code should NOT contain any main functions or code that is executed
# automatically.  We ONLY want the functions as stated in the README.md.
# Make sure to comment out or remove all unnecessary code before submitting.


from audioop import cross
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (confusion_matrix, accuracy_score, recall_score, precision_score)

from collections import OrderedDict


class CancerClassifier:
    '''
    A general class to try out different sklearn classifiers
    on the cancer dataset
    '''
    def __init__(self, classifier, train_ratio: float = 0.7):
        self.classifier = classifier
        cancer = load_breast_cancer() #D=30
        self.X = cancer.data  # all feature vectors
        self.t = cancer.target  # all corresponding labels
        self.X_train, self.X_test, self.t_train, self.t_test =\
            train_test_split(
                cancer.data, cancer.target,
                test_size=1-train_ratio, random_state=109)

        # Fit the classifier to the training data here
        self.classifier.fit(self.X_train, self.t_train)
        self.predictions = self.classifier.predict(self.X_test)


    def confusion_matrix(self) -> np.ndarray:
        '''Returns the confusion matrix on the test data
        '''
        return confusion_matrix(self.t_test, self.predictions)

    def accuracy(self) -> float:
        '''Returns the accuracy on the test data
        '''
        return accuracy_score(self.t_test, self.predictions)

    def precision(self) -> float:
        '''Returns the precision on the test data
        '''
        return precision_score(self.t_test, self.predictions)

    def recall(self) -> float:
        '''Returns the recall on the test data
        '''
        return recall_score(self.t_test, self.predictions)

    def cross_validation_accuracy(self) -> float:
        '''Returns the average 10-fold cross validation
        accuracy on the entire dataset.
        '''
        score_array = cross_val_score(self.classifier, self.X, self.t, cv=10)
        return score_array.mean()

    def feature_importance(self) -> list:
        '''
        Draw and show a barplot of feature importances
        for the current classifier and return a list of
        indices, sorted by feature importance (high to low).
        '''
        feature_importance = self.classifier.feature_importances_
        index = range(feature_importance.shape[0])

        fig = plt.figure()
        ax = fig.add_subplot()

        plt.bar(index, feature_importance)
        plt.grid()
        ax.set_xlabel('Feature Index')
        ax.set_ylabel('Feature Importance')   
        plt.title('Feature importance plot')
        plt.show()

        #Negate the numbers so smallest is largest and we get desceding order
        #Most important first.  Default is ascending order.
        return np.argsort(-feature_importance)


def _plot_oob_error():
    RANDOM_STATE = 1337

    cancer = load_breast_cancer() #D=30
    X = cancer.data  # all feature vectors
    t = cancer.target  # all corresponding labels

    ensemble_clfs = [
        ("RandomForestClassifier, max_features='sqrt'",
            RandomForestClassifier(
                n_estimators=100,
                warm_start=True,
                oob_score=True,
                max_features="sqrt",
                random_state=RANDOM_STATE)),
        ("RandomForestClassifier, max_features='log2'",
            RandomForestClassifier(
                n_estimators=100,
                warm_start=True,
                max_features='log2',
                oob_score=True,
                random_state=RANDOM_STATE)),
        ("RandomForestClassifier, max_features=None",
            RandomForestClassifier(
                n_estimators=100,
                warm_start=True,
                max_features=None,
                oob_score=True,
                random_state=RANDOM_STATE))]

    # Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
    error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)

    min_estimators = 30
    max_estimators = 175

    for label, clf in ensemble_clfs:
        for i in range(min_estimators, max_estimators + 1):
            clf.set_params(n_estimators=i)
            #Fit to whole data set since random forest calculates 
            #bootstrap combinations of data for each internal tree
            #Thats what demo code seems to do
            clf.fit(X,  t)  # Use cancer data here
            oob_error = 1 - clf.oob_score_
            error_rate[label].append((i, oob_error))

    # Generate the "OOB error rate" vs. "n_estimators" plot.
    for label, clf_err in error_rate.items():
        xs, ys = zip(*clf_err)
        plt.plot(xs, ys, label=label)

    plt.xlim(min_estimators, max_estimators)
    plt.xlabel("n_estimators")
    plt.ylabel("OOB error rate")
    plt.legend(loc="upper right")
    plt.show()


def _plot_extreme_oob_error():
    
    #Assume its useful to keep this for grading
    RANDOM_STATE = 1337
    
    cancer = load_breast_cancer() #D=30
    X = cancer.data  # all feature vectors
    t = cancer.target  # all corresponding labels

    ensemble_clfs = [
        ("ExtraTreesClassifier, max_features='sqrt'",
            ExtraTreesClassifier(
                n_estimators=100,
                warm_start=True,
                oob_score=True,
                bootstrap=True,
                max_features="sqrt",
                random_state=RANDOM_STATE)),
        ("ExtraTreesClassifier, max_features='log2'",
            ExtraTreesClassifier(
                n_estimators=100,
                warm_start=True,
                max_features='log2',
                oob_score=True,
                bootstrap=True,
                random_state=RANDOM_STATE)),
        ("ExtraTreesClassifier, max_features=None",
            ExtraTreesClassifier(
                n_estimators=100,
                warm_start=True,
                max_features=None,
                oob_score=True,
                bootstrap=True,
                random_state=RANDOM_STATE))]

    # Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
    error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)

    min_estimators = 30
    max_estimators = 175

    for label, clf in ensemble_clfs:
        for i in range(min_estimators, max_estimators + 1):
            clf.set_params(n_estimators=i)
            #Fit to whole data set since random forest calculates 
            #bootstrap combinations of data for each internal tree
            #Thats what demo code seems to do
            clf.fit(X,  t)  # Use cancer data here
            oob_error = 1 - clf.oob_score_
            error_rate[label].append((i, oob_error))

    # Generate the "OOB error rate" vs. "n_estimators" plot.
    for label, clf_err in error_rate.items():
        xs, ys = zip(*clf_err)
        plt.plot(xs, ys, label=label)

    plt.xlim(min_estimators, max_estimators)
    plt.xlabel("n_estimators")
    plt.ylabel("OOB error rate")
    plt.legend(loc="upper right")
    plt.show()

def _test_2_1():
    '''Function to run tests for problem 2.1'''
    #Sqrt(30) = 5,48
    #log2(30) = 4,91
    max_feature_array = [3, 4, "sqrt", "log2", 6, 7, 8, 9, 10] #Default = sqrt
    n_estimator_array = [10, 20, 40, 80, 100, 160, 320] #Default = 100

    best_acc = 0
    best_acc_features = 0
    best_acc_n = 0
    best_rec = 0
    best_rec_features = 0
    best_rec_n = 0
    best_pre = 0
    best_pre_features = 0
    best_pre_n = 0
    best_cva = 0
    best_cva_features = 0
    best_cva_n = 0

    for feature in range(len(max_feature_array)):
        for estimator in range(len(n_estimator_array)):
            
            print('Testing:')
            print('Max Features: ', max_feature_array[feature])
            print('N Estimators: ', n_estimator_array[estimator])

            classifier_type = RandomForestClassifier(n_estimators=n_estimator_array[estimator], max_features=max_feature_array[feature])
            cc = CancerClassifier(classifier_type)
            
            cm = cc.confusion_matrix()           
            
            acc = cc.accuracy()  
            if acc > best_acc:
                best_acc = acc
                best_acc_features = max_feature_array[feature] 
                best_acc_n = n_estimator_array[estimator]     
            
            pre = cc.precision()
            if pre > best_pre:
                best_pre = pre
                best_pre_features = max_feature_array[feature] 
                best_pre_n = n_estimator_array[estimator]  
            
            rec = cc.recall()
            if rec > best_rec:
                best_rec = rec
                best_rec_features = max_feature_array[feature] 
                best_rec_n = n_estimator_array[estimator]  

            cva = cc.cross_validation_accuracy()
            if cva > best_cva:
                best_cva = cva
                best_cva_features = max_feature_array[feature] 
                best_cva_n = n_estimator_array[estimator]  
            
            print('Confusion Matrix:')
            print(cm)
            print('Accuracy: ', acc)
            print('Precision: ', pre)
            print('Recall: ', rec)
            print('Cross Validation Accuracy: ', cva)
            print('')

    print('Best results:')

    print('Accuracy:')
    print('Value:', best_acc)
    print('Max Features:', best_acc_features)
    print('N Estimators:', best_acc_n)

    print('Precision:')
    print('Value:', best_pre)
    print('Max Features:', best_pre_features)
    print('N Estimators:', best_pre_n)

    print('Recall:')
    print('Value:', best_rec)
    print('Max Features:', best_rec_features)
    print('N Estimators:', best_rec_n)

    print('Cross Validation Accuracy:')
    print('Value:', best_cva)
    print('Max Features:', best_cva_features)
    print('N Estimators:', best_cva_n)


def x_fold_split(features: np.ndarray, targets: np.ndarray, x: int):
    '''Shuffle the features and targets in unison. Split into x sections for cross validation.'''

    #Standard code from split_train_test
    p = np.random.permutation(features.shape[0])
    features = features[p]
    targets = targets[p]

    section_len = features.shape[0] // x

    feature_data_splits = []
    target_data_splits = []

    for section in range(x):
        if section == x-1:            
            features_sec, targets_sec = features[section_len*section:, :], targets[section_len*section:]
        else: 
            features_sec, targets_sec = features[section_len*section:section_len*(section+1), :], targets[section_len*section:section_len*(section+1)]

        feature_data_splits.append(features_sec)
        target_data_splits.append(targets_sec)

    return feature_data_splits, target_data_splits

def _indep():
    cancer = load_breast_cancer() #D=30
    X = cancer.data  # all feature vectors
    t = cancer.target  # all corresponding labels

    validation_fold = 10

    feature_data_splits, target_data_splits = x_fold_split(X, t, validation_fold)

    #Parameters to test
    runs = [(100, 4), (80, 6)]

    for i in runs:
        
        accuracies = []
        precisions = []
        recalls = []
        cms = []

        my_classifier = RandomForestClassifier(n_estimators=i[0], max_features=i[1])

        #Calculate stats for each fold
        for fold in range(validation_fold):
            test_X = feature_data_splits[fold]
            test_t = target_data_splits[fold]

            #Make arrays
            if fold == 0:
                train_x = feature_data_splits[1]
                train_t = target_data_splits[1]
                for others in range(2, validation_fold):
                    train_x = np.append(train_x, feature_data_splits[others], axis=0)
                    train_t = np.append(train_t, target_data_splits[others], axis=0)
            else:
                train_x = feature_data_splits[0]
                train_t = target_data_splits[0]
                for others in range(1, validation_fold):
                    if fold != others:
                        train_x = np.append(train_x, feature_data_splits[others], axis=0)
                        train_t = np.append(train_t, target_data_splits[others], axis=0)            

            #Train, predict and analyze for fold
            my_classifier.fit(train_x, train_t)
            predictions = my_classifier.predict(test_X)
            
            accuracies.append(accuracy_score(test_t, predictions))
            precisions.append(precision_score(test_t, predictions))
            recalls.append(recall_score(test_t, predictions))
            cms.append(confusion_matrix(test_t, predictions))

        #Calculate mean for run
        mean_accuracy = 0
        mean_precision = 0
        mean_recall = 0
        mean_cm = 0

        for fold in range(validation_fold):
            mean_accuracy += accuracies[fold] / validation_fold
            mean_precision += precisions[fold] / validation_fold
            mean_recall += recalls[fold] / validation_fold
            mean_cm += cms[fold] / validation_fold

        #Print run stats
        print('\nRun - n_estimators:', i[0], ', max_features:', i[1])
        print('Cross Validation Accuracy:', mean_accuracy)
        print('Cross Validation Precision:', mean_precision)
        print('Cross Validation Recall:', mean_recall)
        print('Cross Validation CM:', mean_cm)




