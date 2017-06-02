import numpy as np
from scipy import misc
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import sys

def read_data():
	X = np.empty((481, 512*480))
	for i in range(481):
		file_name = 'hand.seq' +  str(i+1) + '.png'
		img = misc.imread('hand/' + file_name)
		flat = np.ravel(img)
		X[i] = flat
	return X

X = read_data()
print(X.shape)
np.save('hand',X)