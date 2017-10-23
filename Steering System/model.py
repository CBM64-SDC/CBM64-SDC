import csv
import cv2

import tensorflow as tf
import numpy as np

from scipy.misc import imread, imresize
from sklearn.utils import shuffle
from os import listdir
from os.path import join

from keras.models import Sequential
from keras.layers import Conv2D, ConvLSTM2D, Dense, MaxPooling2D, Dropout, Flatten
from keras.optimizers import Adam

from utils import *

base_dir = '/Users/mohammedamarnah/Desktop/SDCProject/data/IMG/'

img_files = listdir('../../data/IMG/')

with open('../../data/driving_log_clean.csv', 'w') as wf:
    with open('../../data/driving_log.csv', 'r') as rf:
        reader = csv.reader(rf)
        writer = csv.writer(wf)
        for row in reader:
            rel_path = row[0].split('/')[-1]
            if rel_path in img_files:
                real_path = join(base_dir, rel_path)
                writer.writerow([real_path, row[3]])

with open('../../data/driving_log_clean.csv', 'r') as f:
    reader = csv.reader(f)
    data = np.array([row for row in reader])

np.random.shuffle(data)
split_i = int(len(data) * 0.9)
X_train, y_train = list(zip(*data[:split_i]))
X_val, y_val = list(zip(*data[split_i:]))

X_train, y_train = np.array(X_train), np.array(y_train)
X_val, y_val = np.array(X_val), np.array(y_val)

model = Sequential()

# keras.layers.Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)


# model.add(Conv2D(32,3,3, input_shape=(32,16,1), border_mode='same', activation='relu'))
# model.add(Conv2D(64,3,3, border_mode='same', activation='relu'))
# model.add(Dropout(0.5))
# model.add(Conv2D(128,3,3, border_mode='same', activation='relu'))
# model.add(Conv2D(256,3,3, border_mode='same', activation='relu'))
# model.add(Dropout(0.5))
# model.add(Flatten())
# model.add(Dense(1024, activation='relu'))
# model.add(Dense(512, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(1, name='output', activation='tanh'))

model.add(Conv2D(3, 5, 5, subsample=(2, 2), border_mode='same', input_shape=(32,16,3), activation='relu'))
model.add(Conv2D(24, 5, 5, subsample=(2, 2), border_mode='same', activation='relu'))
model.add(Conv2D(36, 5, 5, subsample=(2, 2), border_mode='same', activation='relu'))
model.add(Conv2D(48, 3, 3, border_mode='same', activation='relu'))
model.add(Conv2D(64, 3, 3, border_mode='same', activation='relu'))
model.add(Flatten())
model.add(Dense(1164, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, name='output', activation='tanh'))

model.compile(optimizer=Adam(lr=1e-4), loss='mse')
# model.fit(X_train, y_train, validation_split=0.2, nb_epoch=3)
history = model.fit_generator(gen_batches(X_train, y_train, 128),
                                len(X_train),
                                5,
                                validation_data=gen_batches(X_val, y_val, 128),
                                nb_val_samples=len(X_val))

model.save('model.h5')

json = model.to_json()
model.save_weights('save/model.h5')
with open('save/model.json', 'w') as f:
    f.write(json)