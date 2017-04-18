import numpy as np
from sklearn import svm
import sys
import csv
import matplotlib.pyplot as plt
def length(x):
	return np.sqrt(x.dot(x.T))
def read_data():
	testX = None
	testY = None
	trainX = None
	trainY = None
	with open("features.train", "r+") as train_file:
		data = np.array([[float(i.strip()) for i in line.split()] for line in train_file])
		trainX = data[:, 1:]
		trainY = data[:, 0]
	with open("features.test", "r+") as test_file:
		data = np.array([[float(i.strip()) for i in line.split()] for line in test_file])
		testX = data[:, 1:]
		testY = data[:, 0]
	return (trainX, trainY, testX, testY)
def problem_11(X, y):
	y[y!=0] = -1
	y[y==0] = 1
	logC = np.array([-5, -3, -1, 1,3], dtype=float)
	C = np.power(10, logC)
	W = np.zeros(5, dtype=float)
	for i in range(5):
		clf = svm.SVC(C=C[i], kernel='linear',shrinking=False)
		clf.fit(X, y)
		W[i] = length(clf.coef_)
	plot_answer('prob11',logC, W, 'logC', '|W|')
def problem_12_13(X, y):
	y[y!=8] = -1
	y[y==8] = 1
	print(np.unique(y))
	logC = np.array([-5, -3, -1, 1, 3], dtype=float)
	C = np.power(10, logC)
	Ein = np.zeros(5, dtype=float)
	SV_num = np.zeros(5, dtype=int)
	for i in range(5):
		clf = svm.SVC(C=C[i], kernel='poly', degree=2, gamma=1, coef0=1, shrinking=False)
		clf.fit(X, y)
		Ein[i] = 1 - clf.score(X,y)
		SV_num[i] = clf.support_.shape[0]
	plot_answer("prob12", logC, Ein, "logC", "Ein")
	plot_answer("prob13", logC, SV_num, "logC", "SV number(s)")
def problem_14(X, y):
	y[y!=0] = -1
	y[y==0] = 1
	logC = np.array([-3, -2, -1, 0, 1], dtype=float)
	D = np.zeros(5, dtype=float)
	C = np.power(10, logC)
	for i in range(5):
		clf = svm.SVC(C=C[i], kernel='rbf', gamma=80, shrinking=False)
		clf.fit(X, y)
		SV = clf.support_vectors_[0]
		D[i] = np.abs(clf.decision_function(SV.reshape(1, -1)))
	plot_answer("prob14", logC, D, "logC", "Distance")
def problem_15(X, y, testX, testY):
	y[y!=0] = -1
	y[y==0] = 1
	testY[testY!=0] = -1
	testY[testY==0] = 1
	Eout = np.zeros(5, dtype=float)
	logGamma = np.array([0,1,2,3,4], dtype=float)
	Gamma = np.power(10, logGamma)
	for i in range(5):
		clf = svm.SVC(C=0.1, kernel='rbf', gamma=Gamma[i], shrinking=False)
		clf.fit(X,y)
		Eout[i] = 1 - clf.score(testX, testY)
		print(Eout[i])
	plot_answer("prob15",logGamma, Eout, "logGamma", "Eout")
def problem_16(X, y):
	y[y!=0] = -1
	y[y==0] = 1
	logGamma = np.array([-1,0,1,2,3], dtype=float)
	Gamma = np.power(10, logGamma)
	select = np.zeros(5, dtype=int)
	# set validation
	for i in range(100):
		print(i)
		index = np.arange(len(X))
		np.random.shuffle(index)
		val_X = X[index[:1000]]
		val_y = y[index[:1000]]
		trainX = X[index[1000:]]
		trainY = y[index[1000:]]
		Eval = np.zeros(5, dtype=float)
		for j in range(5):
			clf = svm.SVC(C=0.1, kernel='rbf', gamma=Gamma[j], shrinking=False)
			clf.fit(trainX, trainY)
			Eval[j] = 1 - clf.score(val_X, val_y)
		select[np.argmin(Eval)] = select[np.argmin(Eval)] + 1
	plot_answer("prob16",logGamma, select, "logGamma", "select(s)", form='Histogram')		
def plot_answer(problem,X, Y, X_label, Y_label, form='line'):
	if form == 'Histogram':
		plt.bar(X, Y, align='center')
		plt.xlabel(X_label)
		plt.ylabel(Y_label)
		plt.savefig(problem + '.png')
	else:
		plt.plot(X, Y, 'ro')
		plt.plot(X, Y)
		plt.xlabel(X_label)
		plt.ylabel(Y_label)
		plt.savefig(problem + '.png')
	plt.close()
# read data
np.random.seed(1)
trainX, trainY, testX, testY = read_data()
# solve problem 11
problem_11(trainX, np.copy(trainY))
print("solved 11")
# solve problem 12 and 13
problem_12_13(trainX, np.copy(trainY))
print("solved 12, 13")
# solve problem 14
problem_14(trainX, np.copy(trainY))
print("solved 14")
problem_15(trainX, np.copy(trainY), testX, testY)
print("solved 15")
problem_16(trainX, np.copy(trainY))
print("solved 16")