import csv
import models
import utils

import numpy as np

import pandas as pd
pd.options.display.float_format = '${:,.10f}'.format

from sklearn.utils import shuffle

from keras.models import Sequential

###### TRAINING CONSTANTS ######
SPLIT = 0.2
BATCH_SIZE = 64
EPOCHS = 25
VAL_SAMPLES = 30000
SAMPLES_PER_EPOCH = (20000//BATCH_SIZE)*BATCH_SIZE
############################

# Reading from the clean csv after the path fixing
utils.fixPath()
data = pd.read_csv('../../data/driving_log_clean.csv')

# Shuffling the data
data = data.sample(frac=1).reset_index(drop=True)

# Splitting the data: 70% Training, 30% Validation (See SPLIT under SOME CONSTANTS)
train_num = int(data.shape[0]*SPLIT)
training_data = data.loc[0:train_num-1]
validation_data = data.loc[train_num:]

# Visualize the distribution of (Y) -- the steering angles
# utils.visualize_distribution(training_data['steering'])

# Freeing the memory block cause you know, it needs to be free.
data = None

# Generate training and validation data for the model compilation
train = utils.gen_batches(training_data, BATCH_SIZE)
valid = utils.gen_batches(validation_data,  BATCH_SIZE)

######################### Model Training ###########################

model = models.nvidia(comp=True, summary=True)

history = model.fit_generator(train, samples_per_epoch=SAMPLES_PER_EPOCH,
                            nb_epoch=EPOCHS, validation_data=valid,
                            nb_val_samples=VAL_SAMPLES)

####################################################################

json = model.to_json()
model.save_weights('../../save/model.h5')
with open('../../save/model.json', 'w') as f:
    f.write(json)
