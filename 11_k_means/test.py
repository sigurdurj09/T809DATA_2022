import solution11 as sol
import numpy as np
import tools
from sklearn import svm


print('Test 1.1')

a = np.array([
    [1, 0, 0],
    [4, 4, 4],
    [2, 2, 2]])

b = np.array([
    [0, 0, 0],
    [4, 4, 4]])

print(sol.distance_matrix(a, b))

print('Test 1.2')

dist = np.array([
        [  1,   2,   3],
        [0.3, 0.1, 0.2],
        [  7,  18,   2],
        [  2, 0.5,   7]])

print(sol.determine_r(dist))


print('Test 1.3')

dist = np.array([
        [  1,   2,   3],
        [0.3, 0.1, 0.2],
        [  7,  18,   2],
        [  2, 0.5,   7]])

R = sol.determine_r(dist)

print(sol.determine_j(R, dist))


print('Test 1.4')

X = np.array([
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, 0]])

Mu = np.array([
    [0.0, 0.5, 0.1],
    [0.8, 0.2, 0.3]])

R = np.array([
    [1, 0],
    [0, 1],
    [1, 0]])

print(sol.update_Mu(Mu, X, R))

print('Test 1.5')

X, y, c = tools.load_iris()

print(sol.k_means(X, 4, 10))


print('Test 1.6')

#sol._plot_j()

print('Test 1.7')

#sol._plot_multi_j()

print('Test 1.8')


print('Test 1.9')

X, y, c = sol.load_iris()

print(sol.k_means_predict(X, y, c, 5))

print('Test 1.10')

print(sol._iris_kmeans_accuracy())

print('Test 2.1')

#print(sol._my_kmeans_on_image())

# num_clusters = [2, 5, 10, 20]

# for num in num_clusters:
#     sol.plot_image_clusters(num)

print('Test 3.1')
sol._gmm_info()

print('Test 3.2')
sol._plot_gmm()

print('Test Independent')






