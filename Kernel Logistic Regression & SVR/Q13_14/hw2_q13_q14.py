import numpy as np
from sklearn.svm import SVR
import sys
def read_data(path):
	lines = open(path, 'r').readlines()
	datas = np.array([np.fromstring(line, dtype=float, sep=' ') for line in lines])
	all_x = datas[:, :-1]
	all_y = datas[:, -1]
	return (all_x, all_y)
def get_error(predict, origin):
	error = np.sum(predict != origin)
	return  float(error/len(predict))
all_x, all_y = read_data('hw2_lssvm_all.dat')
train_x, train_y = (all_x[:400], all_y[:400])
test_x, test_y = (all_x[400:], all_y[400:])

for gamma in [32, 2, 0.125]:
	for c in [0.001, 1, 1000]:
		clf = SVR(gamma=gamma, epsilon=0.5, C=c, shrinking=False)
		clf.fit(train_x, train_y)

		predict_train = clf.predict(train_x)
		predict_train[predict_train > 0] = 1
		predict_train[predict_train < 0] = -1
		Ein = get_error(predict_train, train_y)
		print('Ein = ',Ein ,' when gamma = ' ,gamma ,' C = ', c)

		predict_test = clf.predict(test_x)
		predict_test[predict_test > 0] = 1
		predict_test[predict_test < 0] = -1
		Eout = get_error(predict_test, test_y)
		print('Eout = ', Eout , ' when gamma = ' , gamma ,' C = ' ,c)