from keras.layers import Conv2D, ConvLSTM2D, Dense, MaxPooling2D, Dropout, Flatten, Lambda

def nvidia(model):
	model.add(Conv2D(36, 5, 5, subsample=(2, 2), border_mode='valid', activation='relu', input_shape=(64,64,1)))
	#model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

	model.add(Conv2D(48, 5, 5, subsample=(2, 2), border_mode='valid', activation='relu'))
	#model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

	model.add(Conv2D(64, 5, 5, subsample=(2, 2), border_mode='valid', activation='relu'))
	#model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

	model.add(Conv2D(64, 3, 3, subsample=(1, 1), border_mode='valid', activation='relu'))
	#model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

	model.add(Conv2D(128, 3, 3, subsample=(1, 1), border_mode='valid', activation='relu'))
	#model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

	model.add(Flatten())

	model.add(Dense(1164, activation='relu'))
	model.add(Dropout(0.5))

	model.add(Dense(100, activation='relu'))
	model.add(Dropout(0.5))

	model.add(Dense(50, activation='relu'))
	model.add(Dropout(0.5))

	model.add(Dense(10, activation='relu'))

	model.add(Dense(1, name='output', activation='tanh'))

	return model

def alexnet(model):
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

def custom_architecture(model):
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
	return model

	model.add(Dense(4096, activation='relu'))

	model.add(Dense(4096, activation='relu'))

	model.add(Dense(1, activation='tanh'))

	return model