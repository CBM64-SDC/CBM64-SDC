from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import BatchNormalization, Conv2D, ConvLSTM2D, Dense, MaxPooling2D, Dropout, Flatten, Lambda, Convolution2D
from keras.layers.advanced_activations import ELU
# from keras.regularizers import l2, activity_l2

def nvidia(LR=1e-4, inputshape=(64, 64, 1), comp=False, summary=False):
	model = Sequential()
	model.add(Lambda(lambda x: x/127.5-1.0, input_shape=inputshape))
	model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='elu'))
	model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='elu'))
	model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='elu'))
	model.add(Conv2D(64, (3, 3), activation='elu'))
	model.add(Conv2D(64, (3, 3), activation='elu'))
	model.add(Dropout(0.5))
	model.add(Flatten())
	model.add(Dense(100, activation='elu'))
	model.add(Dense(50, activation='elu'))
	model.add(Dense(10, activation='elu'))
	model.add(Dense(1))
	
	if comp:
		model.compile(optimizer=Adam(lr=LR), loss='mse', metrics=['accuracy'])
	if summary:
		model.summary()

	return model, 'nvidia'

def nvidia_2(LR=1e-4, inputshape=(64, 64, 1), comp=False, summary=False):
	model = Sequential()
	model.add(Conv2D(24, 5, 5, subsample=(2, 2), border_mode='same', activation='relu', input_shape=inputshape))
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
	model.add(Dropout(0.5))
	model.add(Dense(100, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(50, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(10, activation='relu'))
	model.add(Dense(1, activation='tanh', name='output'))

	if comp:
		model.compile(optimizer=Adam(lr=LR), loss='mse', metrics=['accuracy'])
	if summary:
		model.summary()

	return model, 'nvidia_2'

def comma_ai(LR=1e-4, inputshape=(64, 64, 1), comp=False, summary=False):
	model = Sequential()
	model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same", input_shape=inputshape))
	model.add(ELU())
	model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
	model.add(ELU())
	model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
	model.add(Flatten())
	model.add(Dropout(.2))
	model.add(ELU())
	model.add(Dense(512))
	model.add(Dropout(.5))
	model.add(ELU())
	model.add(Dense(1))

	if comp:
		model.compile(optimizer=Adam(lr=LR), loss='mse', metrics=['accuracy'])
	if summary:
		model.summary()

	return model, 'comma_ai'

def alexnet(LR=1e-4, inputshape=(64, 64, 1), comp=False, summary=False):
	model = Sequential()

	model.add(Conv2D(64, 11, 11, border_mode='same', activation='relu', input_shape=inputshape))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(128, 7, 7, border_mode='same', activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(192, 3, 3, border_mode='same', activation='relu'))
	#model.add(BatchNormalization((128,112,112)))
	#model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(256, 3, 3, border_mode='same', activation='relu'))
	#model.add(BatchNormalization((128,108,108)))
	#model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Flatten())

	model.add(Dense(12*12*256, activation='relu'))

	model.add(Dense(4096, activation='relu'))

	model.add(Dense(4096, activation='relu'))

	model.add(Dense(1, activation='tanh'))

	if comp:
		model.compile(optimizer=Adam(lr=LR), loss='mse', metrics=['accuracy'])
	if summary:
		model.summary()

	return model, 'alexnet'

def custom_architecture(comp=False, summary=False):
	model = Sequential()
	
	model.add(Conv2D(32,3,3, input_shape=(32,16,1), border_mode='same', activation='relu'))
	
	model.add(Conv2D(64,3,3, border_mode='same', activation='relu'))
	model.add(Dropout(0.5))
	
	model.add(Conv2D(128,3,3, border_mode='same', activation='relu'))
	
	model.add(Conv2D(256,3,3, border_mode='same', activation='relu'))
	model.add(Dropout(0.5))
	
	model.add(Flatten())
	
	model.add(Dense(1024, activation='relu'))
	
	model.add(Dense(512, activation='relu'))
	
	model.add(Dense(128, activation='relu'))
	
	model.add(Dense(1, name='output', activation='tanh'))

	if comp:
		model.compile(optimizer=Adam(lr=1e-4), loss='mse', metrics=['accuracy'])
	if summary:
		model.summary()

	return model
