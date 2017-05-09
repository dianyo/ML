import numpy as np
import sys
import csv
def read_data(path):
	lines = open(path, 'r').readlines()
	datas = np.array([np.fromstring(line, dtype=float, sep=' ') for line in lines])
	all_x = datas[:, :-1]
	all_y = datas[:, -1]
	return (all_x, all_y)
def kernel(x1, x2, gamma):
	return np.exp(-1 * gamma * np.dot((x1 - x2), (x1 - x2)))
def get_kernel_matrix(X, gamma):
	K = np.zeros((X.shape[0], X.shape[0]), dtype=float)
	for i in range(X.shape[0]):
		for j in range(i+1):
			K[i, j] = kernel(X[i], X[j], gamma)
	for i in range(X.shape[0]):
		for j in range(i+1, X.shape[0]):
			K[i, j] = K[j, i]
	return K

def solve(K, l, y):
	return np.linalg.solve(K + np.eye(K.shape[0]) * l, y)

def predict(x, X, Beta, gamma):
	kernel_value = np.array([kernel(x, v, gamma) for v in X])
	y = np.sum(Beta * kernel_value)
	if y > 0 : return 1
	return -1

def get_error(predict, origin):
	error = np.sum(predict != origin)
	return  float(error/len(predict))
all_x, all_y = read_data('hw2_lssvm_all.dat')
train_x, train_y = (all_x[:400], all_y[:400])
test_x, test_y = (all_x[400:], all_y[400:])

for gamma in [32, 2, 0.125]:
	for lamb in [0.001, 1, 1000]:
		K = get_kernel_matrix(train_x, gamma)

		Beta = solve(K, lamb, train_y)

		predict_train = []
		for x in train_x:
			predict_train.append(predict(x, train_x, Beta, gamma))
		predict_train = np.array(predict_train)
		Ein = get_error(predict_train, train_y)
		print('Ein = ',Ein ,' when gamma = ' ,gamma ,' lambda = ', lamb)

		predict_test = []
		for x in test_x:
			predict_test.append(predict(x, train_x, Beta, gamma))
		predict_test = np.array(predict_test)
		Eout = get_error(predict_test, test_y)
		print('Eout = ', Eout , ' when gamma = ' , gamma ,' lambda = ' ,lamb)