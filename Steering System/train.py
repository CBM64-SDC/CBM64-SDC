import csv
import models
import utils

import numpy as np

import pandas as pd
pd.options.display.float_format = '${:,.10f}'.format

from sklearn.utils import shuffle

from keras.models import Sequential

###### TRAINING CONSTANTS ######
SPLIT = 0.6
BATCH_SIZE = 64
EPOCHS = 8
VAL_SAMPLES = 30000
SAMPLES_PER_EPOCH = (20000//BATCH_SIZE)*BATCH_SIZE
############################

# Reading from the clean csv after the path fixing
# Note that custom collected data is joined with 
# the udacity data
path1 = '/Users/mohammedamarnah/Desktop/SDCProject/data/IMG/'
path2 = '/Users/mohammedamarnah/Desktop/SDCProject/custom-data/IMG/'
utils.fixPath(path1, '../../data', append=False)
utils.fixPath(path2, '../../custom-data', append=True)

data = pd.read_csv('../../data/driving_log_clean.csv')

# Shuffling the data
data = data.sample(frac=1).reset_index(drop=True)

# Visualize the distribution of (Y) -- the steering angles
ans = str(input("Visualize the distribution? (Y/N) --- "))
if (ans == 'Y' or ans == 'y'):
	utils.visualize_distribution(data['steering'])

# Splitting the data: 70% Training, 30% Validation (See SPLIT under SOME CONSTANTS)
train_num = int(data.shape[0]*SPLIT)
training_data = data.loc[0:train_num-1]
validation_data = data.loc[train_num:]

print("Training Data Size: ", training_data.shape)
print("Validation Data Size: ", validation_data.shape)

# Freeing the memory block cause you know, it needs to be free.
data = None

ans = str(input("Continue? (Y/N) --- "))
if (ans == 'N' or ans == 'n'):
	exit()

# Generate training and validation data for the model compilation
train = utils.gen_batches(training_data, BATCH_SIZE)
valid = utils.gen_batches(validation_data,  BATCH_SIZE)

######################### Model Training ###########################

model, name = models.comma_ai(comp=True, summary=False)

print("Training model: ", name)

history = model.fit_generator(train, samples_per_epoch=SAMPLES_PER_EPOCH,
                            nb_epoch=EPOCHS, validation_data=valid,
                            nb_val_samples=VAL_SAMPLES)

####################################################################

json = model.to_json()
model.save_weights('../save/model-'+name+'.h5')
with open('../save/model-'+name+'.json', 'w') as f:
    f.write(json)
