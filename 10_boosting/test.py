import solution10 as sol
import numpy as np
import tools
from sklearn import svm
from tools import get_titanic, build_kaggle_submission

print('Test 1.1')
(tr_X, tr_y), (tst_X, tst_y), submission_X = get_titanic()
# get the first 1 row in the training features
print(tr_X[:1])

(tr_X, tr_y), (tst_X, tst_y), submission_X = sol.get_better_titanic()
# get the first 1 row in the training features
print(tr_X[:1])

print('Test 1.2')



print('Test 2.1')

stats = sol.rfc_train_test(tr_X, tr_y, tst_X, tst_y)
print(stats)

print('Test 2.2')

print('Test 2.3')

stats = sol.gb_train_test(tr_X, tr_y, tst_X, tst_y)
print(stats)


print('Test 2.4')


print('Test 2.5')
#params = sol.param_search(tr_X, tr_y)
#print(params)

print('Test 2.6')
#stats = sol.gb_optimized_train_test(tr_X, tr_y, tst_X, tst_y)
#print(stats)

#params = sol.param_search_upd(tr_X, tr_y)
#print(params)

#stats = sol.gb_optimized_train_test_upd(tr_X, tr_y, tst_X, tst_y)
#print('Updated GB opt:', stats)

print('Test 3.1')
#sol._create_submission()
sol._create_submission_upd()

print('Test 3.2')

print('Test Independent')






