import solution03 as sol
import numpy as np
import tools

np.random.seed(1234)
print('Test 1.1')
np.random.seed(1234)
test_1 = sol.gen_data(2, 3, np.array([0, 1, -1]), 1.3)
np.random.seed(1234)
test_2 = sol.gen_data(5, 1, np.array([0.5]), 0.5)

print(test_1)
print(test_2)

print('Test 1.2')

np.random.seed(1234)
X = sol.gen_data(300, 3, np.array([0, 1, -1]), 3**0.25)
tools.bar_per_axis(X)
test_3_m = X.mean(0)

np.random.seed(1234)
test_3_high_n = sol.gen_data(3000, 3, np.array([0, 1, -1]), 3**0.25)
tools.bar_per_axis(test_3_high_n)
test_3_m_high_n = test_3_high_n.mean(0)

np.random.seed(1234)
test_3_low_n = sol.gen_data(30, 3, np.array([0, 1, -1]), 3**0.25)
tools.bar_per_axis(test_3_low_n)
test_3_m_low_n = test_3_low_n.mean(0)

np.random.seed(1234)
test_3_high_cov = sol.gen_data(300, 3, np.array([0, 1, -1]), 3**1)
tools.bar_per_axis(test_3_high_cov)
test_3_m_high_cov = test_3_high_cov.mean(0)

np.random.seed(1234)
test_3_low_cov = sol.gen_data(300, 3, np.array([0, 1, -1]), 3**0.1)
tools.bar_per_axis(test_3_low_cov)
test_3_m_low_cov = test_3_low_cov.mean(0)

print('Means')
print(test_3_m)
print(test_3_m_high_n)
print(test_3_m_low_n)
print(test_3_m_high_cov)
print(test_3_m_low_cov)

print('Deviation')
print(test_3_m - np.array([0, 1, -1]))
print(test_3_m_high_n - np.array([0, 1, -1]))
print(test_3_m_low_n - np.array([0, 1, -1]))
print(test_3_m_high_cov - np.array([0, 1, -1]))
print(test_3_m_low_cov - np.array([0, 1, -1]))

print('Test 1.3')

print('Test 1.4')

mean = np.mean(X, 0)
new_x = sol.gen_data(1, 3, np.array([0, 0, 0]), 1)
print(sol.update_sequence_mean(mean, new_x, X.shape[0]))

print('Test 1.5')

sol._plot_sequence_estimate()

print('Test 1.6')

sol._plot_mean_square_error()

print('Independent Section')
sol._plot_changing_sequence_estimate()