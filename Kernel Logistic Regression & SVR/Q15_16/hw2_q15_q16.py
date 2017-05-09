import numpy as np
def read_data(path):
	lines = open(path, 'r').readlines()
	datas = np.array([np.fromstring(line, dtype=float, sep=' ') for line in lines])
	all_x = datas[:, :-1]
	all_y = datas[:, -1]
	return (all_x, all_y)
def solve(K, l, y):
	return np.linalg.solve(K + np.eye(K.shape[0]) * l, y)
def get_part_train(X, y):
	choice = np.random.choice(400, 400)
	return (X[choice], y[choice])

def kernel(x1, x2):
	return np.dot(x1.T ,x2)
def predict(x, X, Beta):
	kernel_value = np.array([kernel(x, v) for v in X])
	y = np.sum(Beta * kernel_value)
	if y > 0 : return 1
	return -1
def get_error(predict, origin):
	error = np.sum(predict != origin)
	return  float(error/len(predict))
def get_kernel_matrix(X):
	K = np.zeros((X.shape[0], X.shape[0]), dtype=float)
	for i in range(X.shape[0]):
		for j in range(i+1):
			K[i, j] = kernel(X[i], X[j])
	for i in range(X.shape[0]):
		for j in range(i+1, X.shape[0]):
			K[i, j] = K[j, i]
	return K
all_x, all_y = read_data('hw2_lssvm_all.dat')
train_x, train_y = (all_x[:400], all_y[:400])
test_x, test_y = (all_x[400:], all_y[400:])
np.random.seed = 1

for lamb in [0.001, 0.1, 1, 10,100]:
	predict_test = np.zeros(test_y.shape)
	predict_train = np.zeros(train_y.shape)
	for i in range(200):
		part_train_x, part_train_y = get_part_train(train_x, train_y)
		K = get_kernel_matrix(part_train_x)

		Beta = solve(K, lamb, part_train_y)
		k = 0
		for x in train_x:
			predict_train[k] = predict_train[k] + (predict(x, part_train_x, Beta))
			k = k + 1

		k = 0
		for x in test_x:
			predict_test[k] = predict_test[k] + (predict(x, part_train_x, Beta))
			k = k + 1
	predict_train[predict_train > 0] = 1
	predict_train[predict_train < 0] = -1
	Ein = get_error(predict_train, train_y)
	print('Ein = ',Ein ,' lambda = ', lamb)

	predict_test[predict_test > 0] = 1
	predict_test[predict_test < 0] = -1
	Eout = get_error(predict_test, test_y)
	print('Eout = ', Eout , ' lambda = ' ,lamb)