import solution12 as sol
import numpy as np
import tools
from sklearn import svm
import matplotlib.pyplot as plt

print('Test 1.1')
print(sol.standardize(np.array([[0, 0], [0, 0], [1, 1], [1, 1]])))

print('Test 1.2')

X = np.array([
    [1, 2, 3, 4],
    [0, 0, 0, 0],
    [4, 5, 5, 4],
    [2, 2, 2, 2],
    [8, 6, 4, 2]])
sol.scatter_standardized_dims(X, 0, 2)
plt.show()

print('Test 1.3')

sol._scatter_cancer()

print('Test 1.4')

print('Test 2.1')
sol._plot_pca_components()

print('Test 3.1')
sol._plot_eigen_values()

print('Test 3.2')
sol._plot_log_eigen_values()

print('Test 3.3')
sol._plot_cum_variance()


print('Test 3.4')


print('Test Independent')






