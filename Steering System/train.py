import csv
import models
import utils

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
pd.options.display.float_format = '${:,.10f}'.format

from sklearn.utils import shuffle

from scipy.misc import imread

from keras.models import Sequential

###### TRAINING CONSTANTS ######
SPLIT = 0.7
BATCH_SIZE = 64
EPOCHS = 20
VAL_SAMPLES = 2000
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

# Dropping some of the data to balance the dataset
bad = 0
for index1, row1 in data.iterrows():
	if row1['steering'] == 0:
		bad += 1
ind = []
for index, row in data.iterrows():
	if row['steering'] == 0:
		ind.append(index)
		bad -= 1
	if bad == 750:
		break
data = data.drop(data.index[ind]).reset_index(drop=True)

# Splitting the data: (See SPLIT under SOME CONSTANTS)
train_num = int(data.shape[0]*SPLIT)
training_data = data.loc[0:train_num-1]
validation_data = data.loc[train_num:]

print("Full Data Size: ", data.shape)
print("Splitting with split rate: ", SPLIT)
print("Training Data Size: ", training_data.shape)
print("Validation Data Size: ", validation_data.shape)

SAMPLES_PER_EPOCH = (1000//BATCH_SIZE)*BATCH_SIZE
VAL_SAMPLES = 1000


# Visualize the distribution of (Y) -- the steering angles
ans = str(input("Visualize the distribution? (Y/[N]) --- "))
if (ans == 'Y' or ans == 'y'):
	utils.visualize_distribution(data['steering'])

# Freeing the memory block cause you know, it needs to be free.
data = None

ans = str(input("Continue? ([Y]/N) --- "))
if (ans == 'N' or ans == 'n'):
	exit()

# Generate training and validation data for the model compilation
train = utils.gen_batches(training_data, BATCH_SIZE)
valid = utils.gen_batches(validation_data,  BATCH_SIZE)

######################### Model Training ###########################

#model, name = models.nvidia_2(LR=0.01, inputshape=(80, 320, 1), comp=True, summary=False)
#model, name = models.nvidia(LR=1e-4, inputshape=(80,320,1), comp=True, summary=False)
model, name = models.comma_ai(LR=0.001, inputshape=(80,320,1), comp=True, summary=False)

print("Training model: ", name)

history = model.fit_generator(train, samples_per_epoch=SAMPLES_PER_EPOCH,
                            nb_epoch=EPOCHS, validation_data=valid,
                            nb_val_samples=VAL_SAMPLES)

####################################################################

json = model.to_json()
model.save_weights('../save/model-'+name+'.h5')
with open('../save/model-'+name+'.json', 'w') as f:
    f.write(json)
