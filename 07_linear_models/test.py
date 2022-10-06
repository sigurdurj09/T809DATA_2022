import solution07 as sol
import numpy as np
import tools


print('Test 1.1')

X, t = tools.load_regression_iris()
N, D = X.shape

M, sigma = 10, 10
mu = np.zeros((M, D))
for i in range(D):
    mmin = np.min(X[i, :])
    mmax = np.max(X[i, :])
    mu[:, i] = np.linspace(mmin, mmax, M)

print('X:', X)
print('mu:', mu)
print('sigma:', sigma)
print('t:', t)

fi = sol.mvn_basis(X, mu, sigma)
print('fi:', fi)

sol._plot_mu(mu)

print('Test 1.2')

#sol._plot_mvn()

print('Test 1.3')

fi = sol.mvn_basis(X, mu, sigma) # same as before
lamda = 0.001
wml = sol.max_likelihood_linreg(fi, t, lamda)
print('wml:', wml)

print('Test 1.4')

wml = sol.max_likelihood_linreg(fi, t, lamda) # as before
prediction = sol.linear_model(X, mu, sigma, wml)
print('prediction', prediction)

print('Test 1.5')

sol._prediction_accuracy()

print('Test Independent')
sol._prediction_accuracy_indep1()
sol._prediction_accuracy_indep2()





