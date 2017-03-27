import numpy as np
import csv
import pandas
import sys
def get_data():
	origin_data = None
	with open('train.csv', encoding='big5') as f:
		reader = csv.reader(f)
		origin_data = list(reader)

	#remove garbage
	origin_data.pop(0)
	for i in range(len(origin_data)):
		tmp = 3
		while tmp:
			origin_data[i].pop(0)
			tmp = tmp - 1
		if (i - 10) % 18 == 0:
			for j in range(len(origin_data[i])):
				origin_data[i][j] = '1e-10'

	#get each X vector
	data_x = np.empty((5751, 9*18), dtype=float)
	data_y = []
	tmp_data = []
	for i in range(18):
		tmp_data.append([])
	for i in range(len(origin_data)):
		tmp_data[i%18].extend(origin_data[i])
	tmp_data = np.array(tmp_data, dtype=float)
	for i in range(5751):
		data_x[i] = tmp_data[0:18, i:i+9].ravel()
		data_y.append(tmp_data[9, i+9])
	return(data_x, data_y)
def normalize(X):
	tmp = np.amax(X, axis=1) - np.amin(X, axis=1)
	tmp_X = np.copy(X)
	for i in range(len(tmp_X)):
		tmp_X[i] = X[i] / (tmp[i] + 1e-8)
	return tmp_X
def run_adagrad(X, y):
	b = 1
	w = np.full(X.shape[1]*2, 1)
	iteration = 10000
	eta = 0.9
	_lambda = 2
	gb =10
	G = np.full(X.shape[1]*2, 0.0)

	for i in range(iteration):
		b_grad = 0
		w_grad = np.full(X.shape[1]*2, 0.0)
		arr = np.arange(len(y))
		# np.random.shuffle(arr)
		for j in range(len(arr)):
			if j%50 == 0:
				#new b, w
				gb = gb + b_grad**2
				G = G + np.square(w_grad)
				b = b - eta/np.sqrt(gb + 1e-8) * b_grad
				w = w - eta/np.sqrt(G + 1e-8) *w_grad
				b_grad = 0
				w_grad = np.full(X.shape[1]*2, 0.0)
			tmp = 2*(y[j] - w[:X.shape[1]].dot(X[arr[j]]) - w[X.shape[1]:].dot(X[arr[j]]*X[arr[j]]) - b)
			# tmp = 2*(y[j] - w.dot(X[arr[j]]) - b)
			# b_grad = b_grad - tmp
			w_grad[:X.shape[1]] = w_grad[:X.shape[1]] - tmp * X[arr[j]] + 2*_lambda*w[:X.shape[1]]
			w_grad[X.shape[1]:] = w_grad[X.shape[1]:] - tmp * X[arr[j]] * X[arr[j]] + 2*_lambda*w[X.shape[1]:]
			# w_grad = w_grad - tmp*X[arr[j]] #+ 2*_lambda*w
		if i % 300 == 0:
			E_in = lost(b,w,X,y)
			print("global error = " + str(E_in))
	return (b, w)
def remove_garbage(X, garbage_feature):
	new_X = np.delete(X, garbage_feature, 1)
	return new_X
def write_data(b, w):
	tmp_file = open("param1.txt", "w+")
	tmp_file.write(str(b))
	tmp_file.write("\n")
	tmp_file.write(np.array_str(w) + "\n")
	tmp_file.close()
def lost(b, w, X, y):
	lost = 0.0
	for i in range(len(y)):
		# tmp = (y[i] - w.dot(X[i]) - b)
		tmp = (y[i] - w.dot(X[i][1:]) - b)
		lost = lost + tmp**2
	lost = lost / len(y)
	return  np.sqrt(lost)
def find_high_corr(X,y):
	correlation = find_corrrelation(X, y)
	correlation = np.abs(correlation)
	feature = []
	with open('delete_weights', 'w+') as f:
		for i in range(len(correlation)):
			if not correlation[i] > 0.5:
				for j in range(9):
					feature.append(i*9+j)
					f.write(str(i*9 + j) + '\n')
			else:
				for j in range(2):
					feature.append(i*9+j)
					f.write(str(i*9 + j) + '\n')
				print(i)
	return feature
def find_corrrelation(X, y):
	correlation = []
	for i in range(0, X.shape[1], 9):
		tmp_y = np.array(y)
		correlation.append(np.corrcoef(X[:, i], X[:,81])[1,0])
	return correlation
def add_test(X, y):
	with open("test_X.csv", "r+") as f:
		reader = csv.reader(f)
		test_data = list(reader)
	for i in range(len(test_data)):
		test_data[i].pop(0)
		test_data[i].pop(0)
		if (i - 10) % 18 == 0:
			for j in range(0, len(test_data[i])):
				if test_data[i][j] == 'NR':
					test_data[i][j] = '0'
	for i in range(0, 4320, 18):
		tmp = test_data[i:i+18]
		for j in range(2):
			tmp_X = np.array(tmp[8][j:j+7] + tmp[9][j:j+7], dtype=float) #+ tmp[7][j:j+7] + tmp[8][j:j+7] + tmp[9][j:j+7] + tmp[12][j:j+7] + tmp[13][j:j+7], dtype=float)
			tmp_y = float(tmp[9][j+7])
			X = np.vstack((X, tmp_X))
			y.append(tmp_y)
	return (X, y)
def validate(X, y):
	part = int(len(X)/10)
	start = part*9
	end = len(X)
	new_X = np.delete(X, range(start, end), axis=0)
	val_X = X[start:end]
	new_y = np.delete(y, range(start, end), axis=0)
	val_y = y[start:end]
	return (new_X, new_y, val_X, val_y)
X, y = get_data()
np.random.seed(0)
high_correlation_weight = find_high_corr(X, y)
# normal_X = normalize(X)
no_garbage_X = remove_garbage(X, high_correlation_weight)
all_X, all_y = add_test(no_garbage_X, y)
valid_error = 0
# for i in range(5):
# X, y, val_X, val_y = validate(all_X, all_y)
print(all_X.shape)
b = np.full((all_X.shape[0],1), 0.0)
# all_X = np.concatenate((all_X, all_X*all_X), axis=1)
all_X = np.concatenate((b, all_X), axis=1)
print(len(y))
w = np.dot(np.linalg.pinv(all_X), all_y)
b = w[0]
w = w[1:]
print(w.shape)
# valid_error = valid_error + lost(b,w, val_X, val_y)
# 	# print("tmp valid error = " + str(valid_error))
print("Ein error = " + str(lost(b,w,all_X,all_y)))
# print("Validation error (quadratic & all coef > 0.4)=  " + str(valid_error))
write_data(b,w)