import tools as tools
import solution02 as sol
import numpy as np

d, t, classes = tools.load_iris()
#tools.plot_points(d, t)

x, points = d[0,:], d[1:, :]
x_target, point_targets = t[0], t[1:]
print('x: ', x)
print('x_target: ', x_target)

print('Test 1.1')
print(sol.euclidian_distance(x, points[0]))
print(sol.euclidian_distance(x, points[50]))

print('Test 1.2')
print(sol.euclidian_distances(x, points))

print('Test 1.3')
print(sol.k_nearest(x, points, 1))
print(sol.k_nearest(x, points, 3))

print('Test 1.4')
print(sol.vote(np.array([0,0,1,2]), np.array([0,1,2])))
print(sol.vote(np.array([1,1,1,1]), np.array([0,1])))

print('Test 1.5')
print(sol.knn(x, points, point_targets, classes, 1))
print(sol.knn(x, points, point_targets, classes, 5))
print(sol.knn(x, points, point_targets, classes, 150))

print('Test 2.1')
(d_train, t_train), (d_test, t_test) = tools.split_train_test(d, t, train_ratio=0.8)
print(sol.knn_predict(d_test, t_test, classes, 10))
print(sol.knn_predict(d_test, t_test, classes, 5))

print('Test 2.2')
print(sol.knn_accuracy(d_test, t_test, classes, 10))
print(sol.knn_accuracy(d_test, t_test, classes, 5))

print('Test 2.3')
print(sol.knn_confusion_matrix(d_test, t_test, classes, 10))
print(sol.knn_confusion_matrix(d_test, t_test, classes, 20))

print('Test 2.4')
print(sol.best_k(d_train, t_train, classes))

print('Test 2.5')
sol.knn_plot_points(d, t, classes, 3)

print('Independent Tests')

print('Test B.1')
neighbors = sol.k_nearest(x, points, 9)
targets = point_targets[neighbors]
distances = sol.euclidian_distances(x, points[neighbors])
print(sol.vote(targets, classes))
print(sol.weighted_vote(targets, distances, classes))

print('Test B.2')
print(sol.wknn(x, points, point_targets, classes, 1))
print(sol.wknn(x, points, point_targets, classes, 5))
print(sol.wknn(x, points, point_targets, classes, 150))

print('Test B.3')
print(sol.wknn_predict(d_test, t_test, classes, 10))
print(sol.wknn_predict(d_test, t_test, classes, 5))

print('Test B.4')
sol.compare_knns(d_test, t_test, classes)
"""
print('Extra Tests')
d = np.array([[1,1],[3,2],[2,4],[4,5],[5,8],[6,3]])
t = np.array([0,0,0,1,1,1])
classes = [0,1]
#tools.plot_points(d, t)

x, points = d[0,:], d[1:, :]
x_target, point_targets = t[0], t[1:]
print('x: ', x)
print('x_target: ', x_target)

print('Extra Test 1.1')
print(sol.euclidian_distance(x, points[0]))
print(sol.euclidian_distance(x, points[2]))

print('Extra Test 1.2')
print(sol.euclidian_distances(x, points))

print('Extra Test 1.3')
print(sol.k_nearest(x, points, 1))
print(sol.k_nearest(x, points, 3))

print('Extra Test 1.4')
print(sol.vote(np.array(t), np.array(classes)))

print('Extra Test 1.5')
print(sol.knn(x, points, point_targets, classes, 1))
print(sol.knn(x, points, point_targets, classes, 4))


print('Extra Test 2.4')
print(sol.best_k(d_train, t_train, classes))

print('Extra Test 2.5')
sol.knn_plot_points(d, t, classes, 3)

print('Extra Test B.1')
neighbors = sol.k_nearest(x, points, 4)
targets = point_targets[neighbors]
distances = sol.euclidian_distances(x, points[neighbors])
print(sol.vote(targets, classes))
print(sol.weighted_vote(targets, distances, classes))

print('Extra Test B.2')
print(sol.wknn(x, points, point_targets, classes, 1))
print(sol.wknn(x, points, point_targets, classes, 2))
"""