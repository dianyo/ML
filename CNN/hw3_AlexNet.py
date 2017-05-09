import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
import sys
import csv
import numpy as np
import h5py
train_file = sys.argv[1]
def read_train_data(path):
	data = None
	with open(path) as f:
		reader = csv.reader(f)
		data = list(reader)
	data.pop(0)
	y_train = np.array([data[i][0] for i in range(len(data))], dtype=float)
	x_train = np.array([data[i][1].split() for i in range(len(data))], dtype=float)
	x_train = np.reshape(x_train, (len(x_train), 48, 48, 1))
	return (x_train, y_train)


batch_size = 64
num_classes = 7
epochs = 60
data_augmentation = True

x_train, y_train = read_train_data(train_file)
y_train = keras.utils.to_categorical(y_train, num_classes)


model = Sequential()
model.add(Conv2D(96, 3, strides=1, padding='valid', input_shape=(48,48,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(256, 3, strides=1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(384, 3, strides=1, padding='valid'))
model.add(Activation('relu'))

model.add(Conv2D(384, 3, strides=1, padding='valid'))
model.add(Activation('relu'))

model.add(Conv2D(256, 3, strides=1, padding='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(num_classes))
model.add(Activation('softmax'))

opt = keras.optimizers.Adam()

model.compile(loss='categorical_crossentropy',optimizer=opt, metrics=['accuracy'])

x_train = x_train / 255

if not data_augmentation:
	print('Not using data augmentation.')
	model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True)
else:
	print('Using real-time data augmentation.')
	datagen = ImageDataGenerator(featurewise_center=False, samplewise_center=False, featurewise_std_normalization=False,
		samplewise_std_normalization=False, zca_whitening=True, rotation_range=0, width_shift_range=0.1, height_shift_range=0.1,
		horizontal_flip=True, vertical_flip=False)
	datagen.fit(x_train, seed=1)
	model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
		steps_per_epoch= x_train.shape[0] // batch_size,
		epochs=epochs)


model.save('AlexNet_DNN2_Adam_else_data_model.h5')
del model