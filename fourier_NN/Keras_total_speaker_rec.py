from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras import optimizers
from keras.models import model_from_json
import pickle

import numpy as np
import sys
import pickle

def tensorModel(output_neurons, output_activation_func, loss_calculator, trainFile):
	global x_train
	global y_train
	global x_test
	global y_test
	global x_test1
	global y_test1
	neurons = 50
	activation_func = 'relu'
	print(x_train.shape)
	model = Sequential()
	# Input layer
	model.add(Dense(neurons, kernel_initializer="uniform", input_dim=neurons, activation=activation_func))

	# First hidden layer
	model.add(Dense(neurons, kernel_initializer="uniform", activation=activation_func))

	model.add(Dense(neurons, kernel_initializer="uniform", activation=activation_func))
	model.add(Dropout(0.5))

	model.add(Dense(neurons, kernel_initializer="uniform", activation=activation_func))
	model.add(Dropout(0.2))

	model.add(Dense(neurons, kernel_initializer="uniform", activation=activation_func))
	model.add(Dropout(0.2))

	# Output layer
	model.add(Dense(output_neurons, kernel_initializer="uniform", activation=output_activation_func))
	adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	model.compile(loss=loss_calculator, optimizer=adam)
	print(y_test.shape)
	model.fit(x_train, y_train, epochs=120, batch_size=5, validation_data=(x_test, y_test))
	score = model.evaluate(x_test, y_test, batch_size=1)
	model_json = model.to_json()
	with open("model50.json", "w") as json_file:
		json_file.write(model_json)
	# serialize weights to HDF5
	model.save_weights("model50.h5")
	print("Saved model to disk")
	print(score)
	x = 0
	for x_indi in x_test:
		x += 1
		print(x)
		indiArray = [x_indi]
		print(model.predict(np.array(indiArray)))

typeMulti = True
if typeMulti:
	trainFile = 'per10_three_3'

f = open(trainFile + '.pckl', 'rb')
x_train, y_train, x_test, y_test = pickle.load(f)
f.close()

#f = open(trainFile + '_test_individual.pckl', 'rb')
#x_individual, s = pickle.load(f)
#f.close()


if typeMulti:
	tensorModel(3, 'softmax', 'mean_squared_error', trainFile)



