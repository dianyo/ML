import numpy as np
import csv
import pandas
import sys
#read data
train_file = sys.argv[1]
test_file = sys.argv[2]
def get_data():
	global train_file
	origin_data = None
	with open(train_file, encoding='big5') as f:
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
	b = 3
	w = np.full(X.shape[1], 3)
	iteration = 5000
	eta = 0.9
	gb =10
	G = np.full(X.shape[1], 0.0)

	for i in range(iteration):
		b_grad = 0
		w_grad = np.full(X.shape[1], 0.0)
		arr = np.arange(len(y))
		#np.random.shuffle(arr)
		for j in range(len(arr)):
			if j%50 == 0:
				#new b, w
				gb = gb + b_grad**2
				G = G + np.square(w_grad)
				b = b - eta/np.sqrt(gb + 1e-8) * b_grad
				w = w - eta/np.sqrt(G + 1e-8) *w_grad
				b_grad = 0
				w_grad = np.full(X.shape[1], 0.0)
			tmp = 2*(y[j] - w.dot(X[arr[j]]) - b)
			b_grad = b_grad - tmp
			w_grad = w_grad - tmp*X[arr[j]]
		if i % 100 == 0:
			E_in = np.sqrt(lost(b, w, X, y)/5751)
			print(E_in)
	return (b, w)
def remove_garbage(X, garbage_feature):
	# garbage_feature = None
	# with open("garbage_feature", "r+") as f:
	# 	garbage_feature = f.readline()
	# garbage_feature = garbage_feature.replace('[', '')
	# garbage_feature = garbage_feature.replace(']', '')
	# garbage_feature = garbage_feature.replace(',', '')
	# garbage_feature = [int((i)) for i in garbage_feature.split(' ')]
	new_X = np.delete(X, garbage_feature, 1)
	return new_X
def write_data(b, w):
	tmp_file = open("param.txt", "w+")
	tmp_file.write(str(b))
	tmp_file.write("\n")
	tmp_file.write(np.array_str(w) + "\n")
	tmp_file.close()
def lost(b, w, X, y):
	lost = 0.0
	for i in range(len(y)):
		tmp = (y[i] - w.dot(X[i]) - b)
		lost = lost + tmp**2
	return lost
def find_high_corr(X,y):
	correlation = find_corrrelation(X, y)
	correlation = np.abs(correlation)
	feature = []
	with open('delete_weights', 'w+') as f:
		for i in range(len(correlation)):
			if not (i >= 83 and i < 90):
				feature.append(i)
				f.write(str(i) + '\n')
	return feature
def find_corrrelation(X, y):
	correlation = []
	for i in range(X.shape[1]):
		tmp_y = np.array(y)
		correlation.append(np.corrcoef(X[:, i], tmp_y)[1,0])
	return correlation
def add_test(X, y):
	global test_file
	with open(test_file, "r+") as f:
		reader = csv.reader(f)
		test_data = list(reader)
	for i in range(len(test_data)):
		test_data[i].pop(0)
		test_data[i].pop(0)
	for i in range(0, 4320, 18):
		tmp = test_data[i+9]
		for j in range(2):
			tmp_X = np.array(tmp[j:j+7], dtype=float)
			tmp_y = float(tmp[j+7])
			X = np.vstack((X, tmp_X))
			y.append(tmp_y)
	return (X, y)
X, y = get_data()
high_correlation_weight = find_high_corr(X, y)
#normal_X = normalize(X)
no_garbage_X = remove_garbage(X, high_correlation_weight)
add_test_X, add_test_y  = add_test(no_garbage_X, y)
# add_test_X[add_test_X < 0] = 0
b,w = run_adagrad(add_test_X, add_test_y)
write_data(b,w)