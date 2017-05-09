import os
import argparse
from keras.models import load_model
from termcolor import colored,cprint
import keras.backend as K
from utils import *
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy import misc

model_path = sys.argv[1]
emotion_classifier = load_model(model_path)

img = np.load('randomPicture.npy')
img = np.reshape(img, (48,48))
misc.imsave('origin.png', img)
img = np.reshape(img, (1, 48, 48, 1))
pixel = np.ravel(img)
input_img = emotion_classifier.input

val_proba = emotion_classifier.predict(img)
pred = val_proba.argmax(axis=-1)
print(val_proba)
target = K.mean(emotion_classifier.output[:, pred])
grads = K.gradients(target, input_img)[0]
fn = K.function([input_img, K.learning_phase()], [grads])

heatmap = None
tmp_grad = np.array(fn([img, 0]), dtype=float)
tmp_grad = np.absolute(tmp_grad)
heatmap = (tmp_grad - tmp_grad.min()) / (tmp_grad.max() - tmp_grad.min())


'''
Implement your heatmap processing here!
hint: Do some normalization or smoothening on grads
'''

thres = 0.1
see = pixel.reshape(48, 48)
heatmap = heatmap.reshape((48, 48))
see[np.where(heatmap <= thres)] = np.mean(see)


plt.figure()
plt.imshow(heatmap, cmap=plt.cm.jet)
plt.colorbar()
plt.tight_layout()
fig = plt.gcf()
plt.draw()
fig.savefig('heatMap.png', dpi=100)

plt.figure()
plt.imshow(see,cmap='gray')
plt.colorbar()
plt.tight_layout()
fig = plt.gcf()
plt.draw()
fig.savefig('grayPicture.png', dpi=100)
del emotion_classifier
