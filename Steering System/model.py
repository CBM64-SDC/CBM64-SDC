import csv

import numpy as np

from sklearn.utils import shuffle
from os import listdir
from os.path import join

from keras.models import Sequential
from keras.layers import Conv2D, ConvLSTM2D, Dense, MaxPooling2D, Dropout, Flatten, Lambda
from keras.optimizers import Adam

from utils import *

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

model = Sequential()


########### CUSTOM ARCHITECTURE ###########

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

###########################################

########### Nvidia ARCHITECTURE ###########

model.add(Conv2D(24, 5, 5, subsample=(2, 2), border_mode='same', activation='relu', input_shape=(64,64,3)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Conv2D(36, 5, 5, subsample=(2, 2), border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Conv2D(48, 5, 5, subsample=(2, 2), border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Conv2D(64, 3, 3, subsample=(1, 1), border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Conv2D(64, 3, 3, subsample=(1, 1), border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Flatten())

model.add(Dense(1164, activation='relu'))
#model.add(Dropout(0.5))

model.add(Dense(100, activation='relu'))
#model.add(Dropout(0.5))

model.add(Dense(50, activation='relu'))
#model.add(Dropout(0.5))

model.add(Dense(10, activation='relu'))

model.add(Dense(1, name='output', activation='tanh'))

###########################################

train = gen_batches(X_train, y_train, 128)
valid = gen_batches(X_val, y_val, 128)

model.compile(optimizer=Adam(lr=1e-4), loss='mse')
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