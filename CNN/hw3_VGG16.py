import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import Callback
import sys
import csv
import numpy as np
import h5py
import os
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
class History(Callback):
    def on_train_begin(self,logs={}):
        self.tr_losses=[]
        self.val_losses=[]
        self.tr_accs=[]
        self.val_accs=[]

    def on_epoch_end(self,epoch,logs={}):
        self.tr_losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.tr_accs.append(logs.get('acc'))
        self.val_accs.append(logs.get('val_acc'))
def dump_history(store_path,logs):
    with open(os.path.join(store_path,'train_loss'),'a') as f:
        for loss in logs.tr_losses:
            f.write('{}\n'.format(loss))
    with open(os.path.join(store_path,'train_accuracy'),'a') as f:
        for acc in logs.tr_accs:
            f.write('{}\n'.format(acc))
    with open(os.path.join(store_path,'valid_loss'),'a') as f:
        for loss in logs.val_losses:
            f.write('{}\n'.format(loss))
    with open(os.path.join(store_path,'valid_accuracy'),'a') as f:
        for acc in logs.val_accs:
            f.write('{}\n'.format(acc))
batch_size = 64
num_classes = 7
epochs = 30
data_augmentation = True

x_train, y_train = read_train_data(train_file)
x_val = x_train[0:2000]
y_val = y_train[0:2000]


np.save('randomPicture', x_train[100])
sys.exit(0)


x_train = x_train[2000:]
y_train = y_train[2000:]
y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)



print('x_val shape:', x_val.shape)
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')

model = Sequential()
model.add(Conv2D(32, 3, strides=1, padding='valid', input_shape=(48,48,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(32, 3, strides=1))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(64, 3, strides=1, padding='valid'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(64, 3, strides=1))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(128, 3, strides=1, padding='valid'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(128, 3, strides=1))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(128, 3, strides=1))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(128, 3, strides=1))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(256, 3, strides=1, padding='valid'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(256, 3, strides=1))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(256, 3, strides=1))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(256, 3, strides=1))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(400))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

opt = keras.optimizers.Adam()
model.compile(loss='categorical_crossentropy',optimizer=opt, metrics=['accuracy'])
history = History()
model.summary()
x_train = x_train / 255
x_val = x_val /255
if not data_augmentation:
	print('Not using data augmentation.')
	model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True)
else:
	print('Using real-time data augmentation.')
	datagen = ImageDataGenerator(featurewise_center=False, samplewise_center=False, featurewise_std_normalization=False,
		samplewise_std_normalization=False, zca_whitening=False, rotation_range=0, width_shift_range=0.1, height_shift_range=0.1,
		horizontal_flip=True, vertical_flip=False)
	datagen.fit(x_train, seed=1)
	model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
		steps_per_epoch= x_train.shape[0] // batch_size,
		epochs=epochs, validation_data=(x_val, y_val),
        callbacks=[history])

dump_history('historyForVGG16',history)
model.save('VGG16_model.h5')
del model
