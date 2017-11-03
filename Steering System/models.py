from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import BatchNormalization, Conv2D, ConvLSTM2D, Dense, MaxPooling2D, Dropout, Flatten, Lambda

def nvidia(comp=False, summary=False):
	model = Sequential()

	model.add(BatchNormalization(epsilon=0.001,mode=2, axis=2,input_shape=(64, 64, 1)))

	model.add(Conv2D(24, 5, 5, subsample=(2, 2), border_mode='valid', activation='relu'))
	#model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

	model.add(Conv2D(36, 5, 5, subsample=(2, 2), border_mode='valid', activation='relu'))
	#model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

	model.add(Conv2D(48, 5, 5, subsample=(2, 2), border_mode='valid', activation='relu'))
	#model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

	model.add(Conv2D(64, 3, 3, subsample=(1, 1), border_mode='valid', activation='relu'))
	#model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

	model.add(Conv2D(64, 3, 3, subsample=(1, 1), border_mode='valid', activation='relu'))
	#model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

	model.add(Flatten())

	model.add(Dense(1164, activation='relu'))
	#model.add(Dropout(0.5))

	model.add(Dense(100, activation='relu'))
	#model.add(Dropout(0.5))

	model.add(Dense(50, activation='relu'))
	#model.add(Dropout(0.5))

	model.add(Dense(10, activation='relu'))

	model.add(Dense(1, name='output', activation='tanh'))

	if comp:
		model.compile(optimizer=Adam(lr=1e-4), loss='mse', metrics=['accuracy'])
	if summary:
		model.summary()

	return model

def alexnet(comp=False, summary=False):
	model = Sequential()

	model.add(Conv2D(64, 11, 11, border_mode='same', activation='relu', input_shape=(64,64,3)))
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
		model.compile(optimizer=Adam(lr=1e-4), loss='mse', metrics=['accuracy'])
	if summary:
		model.summary()

	return model

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
