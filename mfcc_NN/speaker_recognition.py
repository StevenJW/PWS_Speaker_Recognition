from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras import optimizers
import pickle
from pathlib import Path

import numpy as np
import sys

def tensorModel(output_neurons, output_activation_func, loss_calculator, trainFile):
	global x_train
	global y_train
	global x_test
	global y_test
	global x_individual
	neurons = 120
	activation_func = 'relu'

	model = Sequential()
	# Input layer
	model.add(Dense(neurons, kernel_initializer="uniform", input_dim=neurons, activation=activation_func))

	# First hidden layer
	model.add(Dense(neurons, kernel_initializer="uniform", activation=activation_func))

	model.add(Dense(neurons, kernel_initializer="uniform", activation=activation_func))
	model.add(Dropout(0.5))

	model.add(Dense(neurons, kernel_initializer="uniform", activation=activation_func))
	model.add(Dropout(0.2))

	# Output layer
	model.add(Dense(output_neurons, kernel_initializer="uniform", activation=output_activation_func))
	adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	model.compile(loss=loss_calculator, optimizer=adam)

	model.fit(x_train, y_train, epochs=50, batch_size=5, validation_data=(x_test, y_test))
	score = model.evaluate(x_test, y_test, batch_size=1)
	print(score)
	x = 0
	#print(model.predict(x_individual[0]))
	print('\n')
	for x_indi in x_test:
		x += 1
		#print(x_indi)
		indiArray = [x_indi]
		print(model.predict(np.array(indiArray)))


typeMulti = True
if typeMulti:
	trainFile = 'multi_mfcc_3_1'
	#z = 1
	#while True:
	#	if not Path(trainFile + str(z+1) + '.pckl').is_file():
	#		trainFile = trainFile + str(z)
	#		break
	#	z += 1
else:
	trainFile = 'store_1_sigmoid_1'

print(trainFile)
f = open(trainFile + '.pckl', 'rb')
x_train, y_train, x_test, y_test = pickle.load(f)
f.close()


f = open('multi_kamer_1.pckl', 'rb')
x_individual, dier = pickle.load(f)
f.close()


if typeMulti:
	tensorModel(2, 'softmax', 'mean_squared_error', trainFile)
else: 
	tensorModel(1, 'sigmoid', 'binary_crossentropy', trainFile)