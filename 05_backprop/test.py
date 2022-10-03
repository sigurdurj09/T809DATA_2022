import solution05 as sol
import numpy as np
import tools


print('Test 1.1')

print(sol.sigmoid(0.5))
print(sol.d_sigmoid(0.2))
print(sol.sigmoid(-101))

print('Test 1.2')

print(sol.perceptron(np.array([1.0, 2.3, 1.9]),np.array([0.2,0.3,0.1])))
print(sol.perceptron(np.array([0.2,0.4]),np.array([0.1,0.4])))

print('Test 1.3')
np.random.seed(1234)
features, targets, classes = tools.load_iris()
(train_features, train_targets), (test_features, test_targets) = \
    tools.split_train_test(features, targets)

# initialize the random generator to get repeatable results
np.random.seed(1234)

# Take one point:
x = train_features[0, :]
K = 3 # number of classes
M = 10
D = 4
# Initialize two random weight matrices
W1 = 2 * np.random.rand(D + 1, M) - 1
W2 = 2 * np.random.rand(M + 1, K) - 1
y, z0, z1, a1, a2 = sol.ffnn(x, M, K, W1, W2)

print('y:',y)
print('z0:',z0)
print('z1:',z1)
print('a1:',a1)
print('a2:',a2)

print('Hardcoded test')
x = np.array([6.3, 2.5, 4.9, 1.5])
y, z0, z1, a1, a2 = sol.ffnn(x, M, K, W1, W2)
print('y:',y)
print('z0:',z0)
print('z1:',z1)
print('a1:',a1)
print('a2:',a2)

print('Test 1.4')
# initialize random generator to get predictable results
np.random.seed(42)

K = 3  # number of classes
M = 6
D = train_features.shape[1]

x = features[0, :]

# create one-hot target for the feature
target_y = np.zeros(K)
target_y[targets[0]] = 1.0

# Initialize two random weight matrices
W1 = 2 * np.random.rand(D + 1, M) - 1
W2 = 2 * np.random.rand(M + 1, K) - 1

y, dE1, dE2 = sol.backprop(x, target_y, M, K, W1, W2)

print('y:',y)
print('dE1:',dE1)
print('dE2:',dE2)

print('Test 2.1')
print('EX1')

# initialize the random seed to get predictable results
np.random.seed(23)
features, targets, classes = tools.load_iris()
(train_features, train_targets), (test_features, test_targets) = tools.split_train_test(features, targets)

x = train_features[0, :]
K = 3  # number of classes
M = 6
D = train_features.shape[1]

# Initialize two random weight matrices
W1 = 2 * np.random.rand(D + 1, M) - 1
W2 = 2 * np.random.rand(M + 1, K) - 1

y, z0, z1, a1, a2 = sol.ffnn(x, M, K, W1, W2)
target_y = np.zeros(K)
target_y[targets[0]] = 1.0

W1tr, W2tr, Etotal, misclassification_rate, last_guesses = sol.train_nn(train_features[:20, :], train_targets[:20], M, K, W1, W2, 500, 0.1)

print('W1tr:',W1tr)
print('W2tr:',W2tr)
print('Etotal:',Etotal)
print('missclassification_rate:',misclassification_rate)
print('last_guesses:',last_guesses)
print('targets:     ',train_targets[:20])

print('EX2')
np.random.seed(90210)
features, targets, classes = tools.load_iris()
(train_features, train_targets), (test_features, test_targets) = tools.split_train_test(features, targets)

x = train_features[0, :]
K = 3  # number of classes
M = 8
D = train_features.shape[1]

# Initialize two random weight matrices
W1 = 2 * np.random.rand(D + 1, M) - 1
W2 = 2 * np.random.rand(M + 1, K) - 1

y, z0, z1, a1, a2 = sol.ffnn(x, M, K, W1, W2)
target_y = np.zeros(K)
target_y[targets[0]] = 1.0

W1tr, W2tr, Etotal, misclassification_rate, last_guesses = sol.train_nn(train_features[:20, :], train_targets[:20], M, K, W1, W2, 500, 0.1)

print('W1tr:',W1tr)
print('W2tr:',W2tr)
print('Etotal:',Etotal)
print('missclassification_rate:',misclassification_rate)
print('last_guesses:',last_guesses)
print('targets:     ',train_targets[:20])


print('Test 2.2')
guesses = sol.test_nn(test_features, M, K, W1tr, W2tr)
print('Trained guesses:', guesses)
print('Actual:         ', test_targets)

print('Test 2.3')
np.random.seed()
sol.train_test_plot()

print('Test Independent')
#sol.train_test_plot_mode()




