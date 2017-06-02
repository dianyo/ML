import numpy as np
import sys
import matplotlib.pyplot as plt
train_path = sys.argv[1]
test_path = sys.argv[2]

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

def get_error(g, y ,u):
	return np.sum(u * np.abs(g - y) / 2)

def plot_answer(problem,X, Y, X_label, Y_label, form='line'):
	if form == 'Histogram':
		plt.bar(X, Y, align='center')
		plt.xlabel(X_label)
		plt.ylabel(Y_label)
		plt.savefig(problem + '.png')
	else:
		plt.plot(X, Y)
		plt.xlabel(X_label)
		plt.ylabel(Y_label)
		plt.savefig(problem + '.png')
	plt.close()
def G(s, theta, alpha, x, feature):
	t = len(s)
	G = 0
	for i in range(t):
		G = G + alpha[i]*(s[i]*((x[feature[i]] > theta[i]) * 2 - 1))
	return (G > 0) * 2 - 1
# read data
x_train, y_train = read_data(train_path)
x_test, y_test = read_data(test_path)

# set T and init u = 1/N

T = 300
u = np.ones(len(x_train)) / len(x_train)
all_S = np.array([])
all_theta = np.array([])
all_alpha = np.array([])
all_Ein_gt = np.array([])
all_Eout_gt = np.array([])
all_feature = np.array([], dtype = int)
all_Ein_G = np.array([])
all_Eout_G = np.array([])
all_U = np.array([])
all_epsilon = np.array([])
# run adaptive boost by T = 300
for t in range(T):

	# init Error, F, and S
	minE = np.inf
	min_F = None
	min_S = None
	g_decision_stump = None
	for s in [+1, -1]:
		for feature in range(2):
			
			# sort
			order = np.argsort(x_train[:,feature])
			x_train = x_train[order]
			y_train = y_train[order]
			u = u[order]
			# decision stump
			g = np.ones(len(x_train)) * s
			for i in range(len(x_train)):
				if i > 0:
					g[i - 1] = g[i - 1] * -1
				Eg = get_error(g, y_train, u)
				if Eg < minE:
					minE = Eg
					if i == 0:
						theta = -np.inf
					else:
						theta = (x_train[i-1][feature] + x_train[i][feature]) / 2
					min_F = feature
					min_S = s

	# update u
	all_U = np.append(all_U, np.sum(u))
	order = np.argsort(x_train[:, min_F])
	x_train = x_train[order]
	y_train = y_train[order]
	u = u[order]
	epsilon = minE / np.sum(u)
	all_epsilon = np.append(all_epsilon, epsilon)
	diamond = np.sqrt((1-epsilon)/epsilon)

	predict_y = np.zeros(len(x_train))
	for i in range(len(predict_y)):
		predict_y[i] = min_S * ((x_train[i][min_F] > theta) * 2 - 1)
	u[np.where(predict_y - y_train == 0)] /= diamond
	u[np.where(predict_y - y_train != 0)] *= diamond
	all_Ein_gt = np.append(all_Ein_gt,
		get_error(predict_y, y_train, np.ones(len(x_train)) / len(x_train)))

	predict_y = np.zeros(len(x_test))
	for i in range(len(predict_y)):
		predict_y[i] = min_S * ((x_test[i][min_F] > theta) * 2 - 1)
	all_Eout_gt = np.append(all_Eout_gt,
		get_error(predict_y, y_test, np.ones(len(x_test)) / len(x_test)))


	# update G
	all_S = np.append(all_S, min_S)
	all_theta = np.append(all_theta, theta)
	all_alpha = np.append(all_alpha, np.log(diamond))
	all_feature = np.append(all_feature, min_F)
	predict_y = np.zeros(len(x_train))
	for i in range(len(predict_y)):
		predict_y[i] = G(all_S, all_theta, all_alpha, x_train[i], all_feature)
	all_Ein_G = np.append(all_Ein_G, 
		get_error(predict_y, y_train, np.ones(len(x_train)) / len(x_train)))

	predict_y = np.zeros(len(x_test))
	for i in range(len(predict_y)):
		predict_y[i] = G(all_S, all_theta, all_alpha, x_test[i], all_feature)
	all_Eout_G = np.append(all_Eout_G, 
		get_error(predict_y, y_test, np.ones(len(x_test)) / len(x_test)))

# Q7
plot_answer('Q7', np.arange(300), all_Ein_gt, 't', 'Ein(gt)')

# Q9
plot_answer('Q9', np.arange(300), all_Ein_G, 't', 'Ein(G)')

# Q10
plot_answer('Q10', np.arange(300), all_U, 't', 'Ut')
print(all_U[1])
print(all_U[-1])

# Q11
plot_answer('Q11', np.arange(300), all_epsilon, 't', 'epsilon')
print(np.min(all_epsilon))
print(np.argmin(all_epsilon))

# Q12
plot_answer('Q12', np.arange(300), all_Eout_gt, 't', 'Eout(gt)')
print(all_Eout_gt[0])

# Q13
plot_answer('Q13', np.arange(300), all_Eout_G, 't', 'Eout(G)')
print(all_Eout_G[-1])