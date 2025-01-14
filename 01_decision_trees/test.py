import tools as tools
import solution01 as sol
import numpy as np

print('Test 1.1')
print(sol.prior([0, 0, 1], [0, 1]))
print(sol.prior([0, 2, 3, 3], [0, 1, 2, 3])) 

print('Test 1.2')
features, targets, classes = tools.load_iris()
(f_1, t_1), (f_2, t_2) = sol.split_data(features, targets, 2, 4.65)
print(f_1.shape)
print(t_1.shape)
print(f_2.shape)
print(t_2.shape)

print('Test 1.3')
print(sol.gini_impurity(t_1, classes))
print(sol.gini_impurity(t_2, classes))

print('Test 1.4')
print(sol.weighted_impurity(t_1, t_2, classes))

print('Test 1.5')
print(sol.total_gini_impurity(features, targets, classes, 2, 4.65))

print('Test 1.6')
print(sol.brute_best_split(features, targets, classes, 30))

print('Part2 Test')
features, targets, classes = tools.load_iris()
dt = sol.IrisTreeTrainer(features, targets, classes=classes)
dt.train()
print(f'The accuracy is: {dt.accuracy()}')
dt.plot()
print(f'I guessed: {dt.guess()}')
print(f'The true targets are: {dt.test_targets}')
print(dt.confusion_matrix())

features, targets, classes = tools.load_iris()
dt = sol.IrisTreeTrainer(features, targets, classes=classes, train_ratio=0.6)
dt.plot_progress()

features = np.array([[1,1],[3,2],[2,4],[4,5],[5,8],[6,3]])
targets = np.array([0,0,0,1,1,1])
classes = [0,1]

print('Extra Test 1.1')
print(sol.prior(targets, classes))

print('Extra Test 1.2')
(f_1, t_1), (f_2, t_2) = sol.split_data(features, targets, 0, 2.5)
print(f_1.shape)
print(t_1.shape)
print(f_2.shape)
print(t_2.shape)

print('Extra Test 1.3')
print(sol.gini_impurity(t_1, classes))
print(sol.gini_impurity(t_2, classes))
print(sol.weighted_impurity(t_1, t_2, classes))

print('Extra Test 1.4')
(f_1, t_1), (f_2, t_2) = sol.split_data(features, targets, 0, 4.5)
print(sol.gini_impurity(t_1, classes))
print(sol.gini_impurity(t_2, classes))
print(sol.weighted_impurity(t_1, t_2, classes))

print('Extra Test 1.5')
print(sol.total_gini_impurity(features, targets, classes, 1, 4.65))

print('Extra Test 1.6')
print(sol.brute_best_split(features, targets, classes, 5))

print('Extra Part2 Test')
dt = sol.IrisTreeTrainer(features, targets, classes=classes)
dt.train()
print(f'The accuracy is: {dt.accuracy()}')
dt.plot()
print(f'I guessed: {dt.guess()}')
print(f'The true targets are: {dt.test_targets}')
print(dt.confusion_matrix())

dt = sol.IrisTreeTrainer(features, targets, classes=classes, train_ratio=0.6)
dt.plot_progress()