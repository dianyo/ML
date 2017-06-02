import numpy as np
from scipy import misc
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import sys

def read_data():
	X = np.empty((100, 64*64))
	for i in range(100):
		file_name = chr(65 + int(i/10))
		file_name = file_name +  '{0:0=2d}'.format(i%10) + '.bmp'
		img = misc.imread('faceExpressionDatabase/' + file_name)
		flat = np.ravel(img)
		X[i] = flat
	return X
		

X = read_data()
X_mean = X.mean(axis=0, keepdims=True)
X_ctr = X - X_mean
u, s, v = np.linalg.svd(X_ctr)


first_ten = v[:9]

fig = plt.figure(figsize=(5,4))

for i in range(9):
	ax = fig.add_subplot(3, 3, i+1)
	ax.imshow(np.reshape(first_ten[i], (64,64)), cmap='gray')
	plt.xticks(np.array([]))
	plt.yticks(np.array([]))
	plt.tight_layout()
fig.savefig('eigenfaces.png')

# top_five = v[:5]
# x_reduced = np.dot((X - X_mean), top_five.T)
# x_recover = X_mean + np.dot(x_reduced ,top_five)
# fig = plt.figure(figsize=(10,8))
# for i in range(100):
# 	ax = fig.add_subplot(10, 10, i+1)
# 	ax.imshow(np.reshape(x_recover[i], (64,64)), cmap='gray')
# 	plt.xticks(np.array([]))
# 	plt.yticks(np.array([]))
# 	plt.tight_layout()
# fig.savefig('recoveredFaces.png')

# fig = plt.figure(figsize=(10,8))
# for i in range(100):
# 	ax = fig.add_subplot(10, 10, i+1)
# 	ax.imshow(np.reshape(X[i], (64,64)), cmap='gray')
# 	plt.xticks(np.array([]))
# 	plt.yticks(np.array([]))
# 	plt.tight_layout()
# fig.savefig('originFaces.png')

for i in range(100):
	top_k = v[:i+1]
	print("top " + str(i+1))
	x_reduced = np.dot((X - X_mean), top_k.T)
	x_recover = X_mean + np.dot(x_reduced, top_k)
	l = (X - x_recover)
	l = l * l
	E = np.sqrt(np.sum(l) / 409600)  / 256
	print(E)
	if E < 0.01 :
		print(i+1)
		break
	


