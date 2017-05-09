import os
from termcolor import colored, cprint
import argparse
from keras.utils import plot_model
from keras.models import load_model
import sys
import numpy as np
import matplotlib.pyplot as plt

def plot_acc(train_acc, valid_acc, path):
	X = np.arange(train_acc.size)
	plt.plot(X, train_acc)
	plt.plot(X, valid_acc)
	plt.legend(['train_acc', 'valid_acc'])
	plt.savefig( path)
	plt.close()

output_path = sys.argv[3]
train_acc_path = sys.argv[1]
valid_acc_path = sys.argv[2]
train_acc = []
valid_acc = []

with open(train_acc_path) as f:
	for line in f:
		train_acc.append(line.split())
	train_acc = np.array(train_acc, dtype=float)
with open(valid_acc_path) as f:
	for line in f:
		valid_acc.append(line.split())
	valid_acc = np.array(valid_acc, dtype=float)

train_acc = np.reshape(train_acc, (-1,))
valid_acc = np.reshape(valid_acc, (-1,))

plot_acc(train_acc, valid_acc, output_path)
