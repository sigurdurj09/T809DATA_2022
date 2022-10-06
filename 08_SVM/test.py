import solution08 as sol
import numpy as np
import tools
from sklearn import svm

print('Test 1.1')
sol._plot_linear_kernel(True)

print('Test 1.2')



print('Test 1.3')
sol._compare_gamma(True)

print('Test 1.4')



print('Test 1.5')

sol._compare_C(True)


print('Test 1.6')

print('Test 2.1')
(X_train, t_train), (X_test, t_test) = tools.load_cancer()
svc = svm.SVC(C=1000)

stats = sol.train_test_SVM(svc, X_train, t_train, X_test, t_test)
print(stats)

print('Test 2.2')
sol.SVM_compare(True)
#sol.mult_SVM_compare()


print('Test Independent')
sol.bball_data_analysis()





