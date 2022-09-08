import solution04 as sol
import numpy as np
import tools
import help

#help.estimate_covariance()
#help.pdf()

features, targets, classes = tools.load_iris()
(train_features, train_targets), (test_features, test_targets) = tools.split_train_test(features, targets, train_ratio=0.6)

print('Test 1.1')
print(sol.mean_of_class(train_features, train_targets, 0))
print(sol.mean_of_class(train_features, train_targets, 1))
print(sol.mean_of_class(train_features, train_targets, 2))

print('Test 1.2')
print(sol.covar_of_class(train_features, train_targets, 0))
print(sol.covar_of_class(train_features, train_targets, 1))
print(sol.covar_of_class(train_features, train_targets, 2))

print('Test 1.3')
class_mean = sol.mean_of_class(train_features, train_targets, 0)
class_cov = sol.covar_of_class(train_features, train_targets, 0)

print(sol.likelihood_of_class(test_features[0, :], class_mean, class_cov))

print('Test 1.4')
print(sol.maximum_likelihood(train_features, train_targets, test_features, classes))

print('Test 1.5')
likelihoods = sol.maximum_likelihood(train_features, train_targets, test_features, classes)
print(sol.predict(likelihoods))

print('Test 2.1')
likelihoods2 = sol.maximum_aposteriori(train_features, train_targets, test_features, classes)
print(sol.predict(likelihoods2))

print('Test 2.2')
predict_ml = sol.predict(likelihoods)
predict_ap = sol.predict(likelihoods2)

print(sol.accuracy(test_targets, predict_ml))
print(sol.accuracy(test_targets, predict_ap))

print(sol.confusion_matrix(test_targets, predict_ml, classes))
print(sol.confusion_matrix(test_targets, predict_ap, classes))

print('Test Independent')
sol.multiple_alien_compare()



