import csv

import numpy as np

from sklearn.utils import shuffle
from os import listdir
from os.path import join

from keras.models import Sequential
from keras.layers import Conv2D, ConvLSTM2D, Dense, MaxPooling2D, Dropout, Flatten, Lambda
from keras.layers.convolutional import Convolution2D
#from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam

from utils import *

###########################

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

###########################

base_dir = '/home/lacosa/Downloads/SDCGPKHARA/data/IMG/'

### '/Users/mohammedamarnah/Desktop/SDCProject/data/IMG/'

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

##########################################################

model = Sequential()
model.add(Convolution2D(64, 11, 11, border_mode='same', input_shape=(64,64,3)))
#model.add(BatchNormalization((64,226,226)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Convolution2D(128, 64, 7, 7, border_mode='same'))
#model.add(BatchNormalization((128,115,115)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Convolution2D(192, 128, 3, 3, border_mode='same'))
#model.add(BatchNormalization((128,112,112)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Convolution2D(256, 192, 3, 3, border_mode='same'))
#model.add(BatchNormalization((128,108,108)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Flatten())
model.add(Dense(12*12*256, 4096, init='normal'))
#model.add(BatchNormalization(4096))
model.add(Activation('relu'))
model.add(Dense(4096, 4096, init='normal'))
#model.add(BatchNormalization(4096))
model.add(Activation('relu'))
model.add(Dense(4096, 1000, init='normal'))
#model.add(BatchNormalization(1000))
model.add(Activation('softmax'))

###########################################

train = gen_batches(X_train, y_train, 128)
valid = gen_batches(X_val, y_val, 128)

model.compile(optimizer=Adam(lr=0.01), loss='mse')
history = model.fit_generator(train,
                            samples_per_epoch=20032,
                            nb_epoch=8,
                            validation_data=valid,
                            nb_val_samples=6400,
                            verbose=1)

#model.summary()

model.save('model.h5')

json = model.to_json()
model.save_weights('../../save/model.h5')
with open('../../save/model.json', 'w') as f:
    f.write(json)