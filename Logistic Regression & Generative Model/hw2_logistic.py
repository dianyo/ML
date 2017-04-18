import sys
import numpy as np
import csv

def read_data(X_path, y_path):
	# get X data
	X = None
	with open(X_path) as f:
		reader = csv.reader(f)
		X = list(reader)
	X.pop(0)
	X = np.array(X, dtype=float)
	# get y data 
	y = None
	with open(y_path) as f:
		reader = csv.reader(f)
		y = list(reader)
	y = np.array(y, dtype=float)
	return (X, y)
def read_W(path):
	weight = None
	with open(path) as f:
		reader = csv.reader(f)
		weight = list(reader)
	weight = np.array(weight, dtype=float)
	weight = np.delete(weight, 0, axis=1)
	high_weight = []
	for i in range(len(weight)):
		if np.abs(weight[i]) > 20:
			high_weight.extend([i])
	return high_weight
def sigmoid(z):
	return 1/(1 + np.exp(-1*z))
def f(W, x):
	return sigmoid(W.dot(x))
def gradient(y, W, x):
	return -1*(y - f(W, x))*x
def logistic_regression(X, y):
	# init variables
	X = np.insert(X, 0, 1, axis=1) 
	w_size = X.shape[1]
	W = np.random.random(w_size)
	G = np.full(w_size, 0.0)
	order = np.arange(len(y))
	np.random.shuffle(order)
	iteration = 500
	eta = 0.9
	l = 0.01
	# start gradient descent
	for i in range(iteration):
		w_grad = np.full(w_size, 0.0)
		for j in range(len(y)):
			if j % (int(len(y)/10)) == 0:
				# new W
				G = G + np.square(w_grad)
				W = W - eta/np.sqrt(G + 1e-8) * w_grad
				w_grad = np.full(w_size, 0.0)
			w_grad = w_grad + gradient(y[order[j]], W, X[order[j]])
		if i % 25 == 0:
			print(str(i) + "th iteration")
	return W
def read_test_data(X_path):
	test_X = None
	with open(X_path) as f:
		reader = csv.reader(f)
		test_X  = list(reader)
	test_X.pop(0)
	test_X = np.array(test_X, dtype=float)
	return test_X
def get_answer(X, W):
	y = np.full(X.shape[0], 0, dtype=int)
	X = np.insert(X, 0, 1, axis=1) 
	for i in range(len(X)):
		tmp_y = f(W, X[i])
		if tmp_y >= 0.5:
			y[i] = 1
	return y
def write_submission(y, output_path):
	with open(output_path, 'w', newline='\n') as f:
		writer = csv.writer(f)
		writer.writerow(['id','label'])
		for i in range(len(y)):
			writer.writerow([str(i+1), y[i]])
def write_W(W):
	with open('weight.csv', 'w', newline='\n') as f:
		writer = csv.writer(f)
		for i in range(len(W)):
			writer.writerow([str(i+1), W[i]])
def add_high_level(X, high_weight, level):
	for i in range(1, level):
		for j in range(len(high_weight)):
			new_X = X[:, high_weight[j] - 1]**(i+1)
			X = np.append(X, new_X.reshape((new_X.shape[0], 1)), axis=1)
	return X
def accuracy(y, X, W):
	test_y = get_answer(X, W)
	count = 0
	for i in range(len(y)):
		if y[i] != test_y[i]:
			count = count + 1
	return 1.0 - (count/len(y))
def validation(X, y, part):
	start = int(len(X)/5) * part
	end = None
	if part == 4:
		end = len(X)
	else:
		end = int(len(X)/5) * (part + 1)

	test_X = X[start:end]
	test_y = y[start:end]
	X = np.delete(X, list(range(start, end)),0)
	y = np.delete(y, list(range(start, end)),0)
	return  (X, test_X, y, test_y)

trainX_file = sys.argv[1]
trainy_file = sys.argv[2]
testX_file = sys.argv[3]
output = sys.argv[4]

np.random.seed(0)
all_X, y = read_data(trainX_file, trainy_file)
high_weight = [1,2,4,5,6]
X = add_high_level(all_X, high_weight, 2)
# do cross-validation
# validation_accuracy = 0
# train_accuracy = 0
# for i in range(5):
# 	print("validation " + str(i))
# 	X, test_X, y, test_y = validation(all_X, all_y, i)
# 	X_max = X.max(axis=0)
# 	X_min = X.min(axis=0)
# 	norm_X = (X - X_min) / (X_max - X_min + 1e-8) #normalize X
# 	W = logistic_regression(norm_X, y)
# 	norm_test_X = (test_X - X_min) / (X_max - X_min + 1e-8)
# 	validation_accuracy = validation_accuracy + accuracy(test_y, norm_test_X, W)
# 	train_accuracy = train_accuracy + accuracy(y, norm_X, W)
# print("norm train error: " + str(train_accuracy/5))
# print("norm validation test error: " + str(validation_accuracy/5))
X_max = X.max(axis=0)
X_min = X.min(axis=0)
norm_X = (X - X_min) / (X_max - X_min + 1e-8) #normalize X
W = logistic_regression(norm_X, y)
test_X = read_test_data(testX_file)
test_X = add_high_level(test_X, high_weight, 2)
norm_test_X = (test_X - X_min) / (X_max - X_min)
test_y = get_answer(norm_test_X, W)
write_submission(test_y, output)
write_W(W)
