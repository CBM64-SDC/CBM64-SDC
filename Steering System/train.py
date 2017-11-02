import csv
import models

import numpy as np

from sklearn.utils import shuffle
from os import listdir
from os.path import join

from keras.models import Sequential
from keras.optimizers import Adam

from utils import *

##################### Pre-processing - 1 ###########################
base_dir = '/home/lacosa/Downloads/SDCGPKHARA/data/IMG/'
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
####################################################################


####################### Model Training #############################
model = Sequential()

models.nvidia(model)

model.summary()

train = gen_batches(X_train, y_train, 32)
valid = gen_batches(X_val, y_val, 32)

model.compile(optimizer=Adam(lr=1e-4), loss='mse', metrics=['accuracy'])

history = model.fit_generator(train,
                            samples_per_epoch=20032,
                            nb_epoch=8,
                            validation_data=valid,
                            nb_val_samples=6400)
####################################################################

model.save('model.h5')

json = model.to_json()
model.save_weights('../../save/model.h5')
with open('../../save/model.json', 'w') as f:
    f.write(json)