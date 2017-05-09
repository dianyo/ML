import keras
from keras.models import load_model
import sys
import csv
import numpy as np
test_file = sys.argv[1]
def read_test_data(path):
	data = None
	with open(path) as f:
		reader = csv.reader(f)
		data = list(reader)
	data.pop(0)
	x_test = np.array([data[i][1].split() for i in range(len(data))], dtype=float)
	x_test = np.reshape(x_test, (len(x_test), 48, 48, 1))
	return x_test
def predict_classes(model, x_test):
    '''Generate class predictions for the input samples
    batch by batch.
    # Arguments
        x: input data, as a Numpy array or list of Numpy arrays
            (if the model has multiple inputs).
        batch_size: integer.
        verbose: verbosity mode, 0 or 1.
    # Returns
        A numpy array of class predictions.
    '''
    proba = model.predict(x_test, batch_size=32, verbose=1)
    if proba.shape[-1] > 1:
        return proba.argmax(axis=-1)
    else:
        return (proba > 0.5).astype('int32')
model = load_model('VGG19model.h5')

x_test = read_test_data(test_file)
print(x_test[0])
x_test = x_test / 255
print(x_test[0])

predict_y = model.predict_classes(x_test, batch_size = 64, verbose=1)
# predict_y = predict_classes(model, x_test)

with open('answer2.csv', 'w') as f:
	f.write('id,label\n')
	for idx,a in enumerate(predict_y):
		f.write('{},{}\n'.format(idx,a))