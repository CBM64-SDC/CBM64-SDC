import csv
import models
import utils

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
pd.options.display.float_format = '${:,.10f}'.format

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from scipy.misc import imread

from keras.models import Sequential

###### TRAINING CONSTANTS ######
SPLIT = 0.7
BATCH_SIZE = 40
EPOCHS = 10
SAMPLES_PER_EPOCH = (20000//BATCH_SIZE)*BATCH_SIZE
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
############################

# Reading from the clean csv after the path fixing
# Note that custom collected data is joined with 
# the udacity data
path1 = '/Users/mohammedamarnah/Desktop/SDCProject/data/IMG/'
path2 = '/Users/mohammedamarnah/Desktop/SDCProject/custom-data/IMG/'
path3 = '/Users/mohammedamarnah/Desktop/SDCProject/custom-data-2/IMG/'
path4 = '/Users/mohammedamarnah/Desktop/SDCProject/custom-data-3/IMG/'
utils.fixPath(path1, '../../data', append=False)
utils.fixPath(path2, '../../custom-data', append=True)
utils.fixPath(path3, '../../custom-data-2', append=True)
utils.fixPath(path4, '../../custom-data-3', append=True)

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
	if bad == 2000:
		break
data = data.drop(data.index[ind]).reset_index(drop=True)
# #################

# Reading the data from the pandas dataframe
X = data[['center', 'left', 'right']].values
y = data['steering'].values

# Splitting the data: (See SPLIT under SOME CONSTANTS)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=SPLIT, random_state=0)

# Some information about the data after splitting
print("Full Data Size: ", data.shape)
print("Splitting with split rate: ", SPLIT)
print("Training Data Size: ", X_train.shape, y_train.shape)
print("Validation Data Size: ", X_valid.shape, y_valid.shape)

# Visualize the distribution of (Y) -- the steering angles
ans = str(input("Visualize the distribution? (Y/[N]) --- "))
if (ans == 'Y' or ans == 'y'):
	utils.visualize_distribution(data['steering'])

# Freeing the memory block cause you know, it needs to be free.
data = None
X = None
y = None

ans = str(input("Continue? ([Y]/N) --- "))
if (ans == 'N' or ans == 'n'):
	exit()

# Generate training and validation data for the model compilation
train = utils.gen_batches(X_train, y_train, True, BATCH_SIZE)
valid = utils.gen_batches(X_valid, y_valid,  False, BATCH_SIZE)

######################### Model Training ###########################

model, name = models.nvidia(LR=1e-4, inputshape=INPUT_SHAPE, comp=True, summary=False)
#model, name = models.comma_ai(LR=1e-4, inputshape=INPUT_SHAPE, comp=True, summary=False)

print("Training model: ", name)

history = model.fit_generator(train, epochs=EPOCHS,
                            steps_per_epoch=SAMPLES_PER_EPOCH, validation_steps=len(X_valid), 
							max_q_size=1, verbose=1, validation_data=valid)

####################################################################

save_path = '/Users/mohammedamarnah/Desktop/SDCProject/save/'
json = model.to_json()
model.save_weights('./save/model-'+name+'.h5')
with open(save_path+'model-'+name+'.json', 'w') as f:
    f.write(json)