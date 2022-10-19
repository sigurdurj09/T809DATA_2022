import solution09 as sol
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

print('Test 1.1')
classifier_type = DecisionTreeClassifier()
cc = sol.CancerClassifier(classifier_type)
print('Confusion Matrix:')
print(cc.confusion_matrix())
print('Accuracy:')
print(cc.accuracy())
print('Precision:')
print(cc.precision())
print('Recall:')
print(cc.recall())
print('Cross Validation Accuracy:')
print(cc.cross_validation_accuracy())

print('Test Targets:')
print(cc.t_test)
print('Predictions:')
print(cc.predictions)

print('Test 1.2')


print('Test 2.1')
#sol._test_2_1()

print('Test 2.2')

my_classifier = RandomForestClassifier(n_estimators=10, max_features=3)
cc = sol.CancerClassifier(my_classifier)
feature_idx = cc.feature_importance()
print('Feature Importance:')
print(feature_idx)

print('\nTest 2.3')


print('\nTest 2.4')
sol._plot_oob_error()

print('\nTest 2.5')


print('\nTest 3.1')
my_classifier = ExtraTreesClassifier()
cc = sol.CancerClassifier(my_classifier)
feature_idx = cc.feature_importance()

print('Feature Importance:')
print(feature_idx)

print('Confusion Matrix:')
print(cc.confusion_matrix())
print('Accuracy:')
print(cc.accuracy())
print('Precision:')
print(cc.precision())
print('Recall:')
print(cc.recall())
print('Cross Validation Accuracy:')
print(cc.cross_validation_accuracy())

print('Test 3.2')
sol._plot_extreme_oob_error()

print('Test Independent')






