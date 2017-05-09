import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
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

def FeatExModel():
	inputs = Input(shape=(48,48,1))
	padding1 = ZeroPadding2D(padding=(3,3))(inputs)
	conv1 = Conv2D(64, 3, strides=1, padding='valid')(padding1)
	relu1 = Activation('relu')(conv1)
	pooling1 = MaxPooling2D(pool_size=(3,3), strides=2)(relu1)

	conv2a = Conv2D(96, 1, strides=1, padding='valid')(pooling1)
	relu2a = Activation('relu')(conv2a)
	padding2a = ZeroPadding2D(padding=(1,1))(relu2a)
	conv2b = Conv2D(208, 3, strides=1, padding='valid')(padding2a)
	relu2b = Activation('relu')(conv2b)

	padding2 = ZeroPadding2D(padding=(1,1))(pooling1)
	pooling2 = MaxPooling2D(pool_size=(3,3), strides=1, padding='valid')(padding2)
	conv2c = Conv2D(64, 1, strides=1, padding='valid')(pooling2)
	relu2c = Activation('relu')(conv2c)

	concat2 = keras.layers.concatenate([relu2b, relu2c])
	pooling2b = MaxPooling2D(pool_size=(3,3), strides=2, padding='valid')(concat2)

	conv3a = Conv2D(96, 1, strides=1, padding='valid')(pooling2b)
	relu3a = Activation('relu')(conv3a)
	padding3a = ZeroPadding2D(padding=(1,1))(relu3a)
	conv3b = Conv2D(208, 3, strides=1, padding='valid')(relu3a)
	relu3b = Activation('relu')(conv3b)

	padding3 = ZeroPadding2D(padding=(1,1))(pooling2b)
	pooling3 = MaxPooling2D(pool_size=(3,3), strides=1, padding='valid')(pooling2b)
	conv3c = Conv2D(64, 1, strides=1, padding='valid')(pooling3)
	relu3c = Activation('relu')(conv3c)

	concat3 = keras.layers.concatenate([relu3b, relu3c])
	pooling3b = MaxPooling2D(pool_size=(3,3), strides=2, padding='valid')(concat3)

	flat = Flatten()(pooling3b)
	# DNN1 = Dense(1024)(flat)
	# relu = Activation('relu')(DNN1)
	# drop = Dropout(0.5)(relu)
	# DNN2 = Dense(512)(drop)
	# relu = Activation('relu')(DNN2)
	# drop = Dropout(0.4)(relu)  
	output = Dense(num_classes)(flat)
	output = Activation('softmax')(output)

	model = Model(inputs=inputs, outputs=output)
	return model
batch_size = 64
num_classes = 7
epochs = 25
data_augmentation = True

x_train, y_train = read_train_data(train_file)


y_train = keras.utils.to_categorical(y_train, num_classes)


model = FeatExModel()

opt = keras.optimizers.Adam()

model.compile(loss='categorical_crossentropy',optimizer=opt, metrics=['accuracy'])

x_train = x_train / 255

if not data_augmentation:
	print('Not using data augmentation.')
	model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, validation_split=0.1)
else:
	print('Using real-time data augmentation.')
	datagen = ImageDataGenerator(featurewise_center=False, samplewise_center=False, featurewise_std_normalization=False,
		samplewise_std_normalization=False, zca_whitening=True, rotation_range=0.1, width_shift_range=0.1, height_shift_range=0.1,
		horizontal_flip=True, vertical_flip=False, shear_range=0.1, zoom_range=0.1)
	datagen.fit(x_train, seed=1)
	model.fit_generator(datagen.flow(x_train, y_train, batch_size=100),
		steps_per_epoch= 1000,
		epochs=epochs)

model.save('50epoch_100000data_featEx_model.h5')
del model