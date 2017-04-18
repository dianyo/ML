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
def sigmoid(z):
	return 1/(1 + np.exp(-1*z))
def f(W, x, b):
	return sigmoid((W.T).dot(x) + b)
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
def generative_model(X, y):
	N1 = len(y[y==0])
	N2 = len(y[y==1])
	# get the mean
	sum1 = np.zeros((X.shape[1],))
	sum2 = np.zeros((X.shape[1],))
	count = 0
	for i in range(X.shape[0]):
		if y[i] == 0:
			sum1 = sum1 + X[i]
		else:
			sum2 = sum2 + X[i]
	mu1 = sum1 / N1
	mu2 = sum2 / N2
	# get the variance
	sum1 = np.zeros((X.shape[1], X.shape[1]))
	sum2 = np.zeros((X.shape[1], X.shape[1]))
	count = 0
	for i in range(X.shape[0]):
		if y[i] == 0:
			count = count + 1
			tmp = (X[i] - mu1).reshape((106, 1))
			sum1 = sum1 + tmp.dot(tmp.T)
		else:
			tmp = (X[i] - mu2).reshape((106, 1))
			sum2 = sum2 + tmp.dot(tmp.T)
	sigma1 = sum1 / N1
	sigma2 = sum2 / N2	
	sigma = (N1 * sigma1 + N2 * sigma2) / (N1 + N2)
	mu1 = mu1.reshape((106, 1))
	mu2 = mu2.reshape((106, 1))
	W = (((mu1 - mu2).T).dot(np.linalg.inv(sigma))).T
	b = (-1*(mu1.T).dot(np.linalg.inv(sigma)).dot(mu1) + (mu2.T).dot(np.linalg.inv(sigma)).dot(mu2) + 2*np.log(N1/N2))/2
	return  (W, b)
def read_test_data(X_path):
	test_X = None
	with open(X_path) as f:
		reader = csv.reader(f)
		test_X  = list(reader)
	test_X.pop(0)
	test_X = np.array(test_X, dtype=float)
	return test_X
def get_answer(X, W, b):
	y = np.full(X.shape[0], 0, dtype=int)
	for i in range(len(X)):
		tmp_y = f(W, X[i], b)
		if tmp_y < 0.5:
			y[i] = 1
	return y
def write_submission(y, ouput):
	with open(ouput, 'w', newline='\n') as f:
		writer = csv.writer(f)
		writer.writerow(['id','label'])
		for i in range(len(y)):
			writer.writerow([str(i+1), y[i]])

trainX_file = sys.argv[1]
trainy_file = sys.argv[2]
testX_file = sys.argv[3]
ouput = sys.argv[4]

np.random.seed(0)
X, y = read_data(trainX_file, trainy_file)
X_max = X.max(axis=0)
X_min = X.min(axis=0)
norm_X = (X - X_min) / (X_max - X_min) #normalize X
W,b = generative_model(norm_X, y)
test_X = read_test_data(testX_file)
norm_test_X = (test_X - X_min) / (X_max - X_min)
test_y = get_answer(norm_test_X, W, b)
print(test_y)
write_submission(test_y, ouput)
