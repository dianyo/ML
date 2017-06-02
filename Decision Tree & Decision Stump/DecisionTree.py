import numpy as np
import sys
train_path = sys.argv[1]
test_path = sys.argv[2]

class Node():
	"""docstring for Node"""
	def __init__(self, h, child, parent):
		self.parent = parent
		self.left = None
		self.right = None

		self.feature = None
		self.theta = None

		self.gt = None
		self.h = h
		self.child = child

	def train(self, X, y):
		
		if len(np.unique(y)) == 1:
			self.gt = y[0]
		else:
			min_impurity = np.inf
			
			for feature in range(2):
		
				# sort
				order = np.argsort(X[:,feature])
				X = X[order]
				y = y[order]

				# decision stump
				for i in range(len(X)):
					# find impurity
					impurity = (i + 1) * self.Gini(y[:i+1]) + (len(X) - i - 1) * self.Gini(y[i+1:])
					if impurity < min_impurity:
						min_impurity = impurity
						self.theta = (X[i][feature] + X[i+1][feature]) / 2
						self.feature = feature
						data_left = (np.copy(X[:i+1]), np.copy(y[:i+1]))
						data_right = (np.copy(X[i+1:]), np.copy(y[i+1:]))
			self.left = Node(self.h + 1, 'left', self)
			self.left.train(data_left[0], data_left[1])

			self.right = Node(self.h + 1, 'right', self)
			self.right.train(data_right[0], data_right[1])

	def Gini(self, y):
		count = np.zeros(2, dtype= int)
		for i in range(len(y)) :
	 		count[int((y[i] + 1) / 2)] += 1
		return 1 - np.sum((count / len(y)) ** 2)
	def getRoot(self):
		Tree = self
		while Tree.parent.h != 0:
			Tree = Tree.parent
		return Tree.parent
	def printNode(self):
		Tree = self
		print("h = " + str(self.h))
		print("d = " + str(self.feature))
		print("theta = " + str(self.theta))
		print("child = " + str(self.child))
		print("g = " + str(self.gt))
		print("\n")
		if Tree.gt != None:
			return
		else:
			Tree.left.printNode()
			Tree.right.printNode()

	def predict(self, X):
		predict_y = np.zeros(len(X))

		for i in range(len(X)):
			Tree = self

			while Tree.gt == None:
				if X[i][Tree.feature] < Tree.theta:
					Tree = Tree.left
				else:
					Tree = Tree.right
			predict_y[i] = Tree.gt
		return predict_y

	def prune(self, X, y):
		if self.gt != None:
			orginTree = self
			if self.child == 'left':
				self.parent.left = self.parent.right
			else:
				self.parent.right = self.parent.left
			predict_y = self.getRoot().predict(X)
			E = np.sum(np.abs(predict_y - y)/ 2)  / len(X)
			print(E)
			
			if self.child == 'right':
				self.parent.left = orginTree
			else:
				self.parent.right = orginTree

			tmp = self.parent.left
			self.parent.left = self.parent.right
			self.parent.right = tmp
		else:
			self.left.prune(X, y)
			self.right.prune(X, y)


		
def read_data(path):
	X, y = [], []
	with open(path) as f:
		for line in f:
			data = line.split()
			X.append(data[:2])
			y.append(data[2])
	X = np.array(X, dtype=float)
	y = np.array(y, dtype=int)
	return X, y

# read data
x_train, y_train = read_data(train_path)
x_test, y_test = read_data(test_path)


CART = Node(0, 'root', None)
CART.train(x_train, y_train)
CART.printNode()

predict_y = CART.predict(x_train)
Ein = np.sum(np.abs(predict_y - y_train)/ 2)  / len(x_train)

predict_y = CART.predict(x_test)
Eout = np.sum(np.abs(predict_y - y_test)/ 2)  / len(x_test)

print(Ein)
print(Eout)
print("====================================")
CART.prune(x_train, y_train)
print("====================================")
CART.prune(x_test, y_test)
