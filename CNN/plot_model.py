import os
from termcolor import colored, cprint
import argparse
from keras.utils import plot_model
from keras.models import load_model
import sys
import graphviz

model_path = sys.argv[1]
emotion_classifier = load_model(model_path)
emotion_classifier.summary()
plot_model(emotion_classifier,to_file='model_DNN.png')
del emotion_classifier