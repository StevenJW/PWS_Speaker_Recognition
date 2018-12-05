from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras import optimizers
from keras.preprocessing import sequence
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
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
	neurons = 20
	max_review_length = 20
	#print(x_train.shape)
	#x_train = sequence.pad_sequences(x_train, maxlen=max_review_length)
	#x_test = sequence.pad_sequences(x_test, maxlen=max_review_length)
	activation_func = 'relu'

	embedding_vecor_length = 20
	model = Sequential()

	"""
	model.add(Embedding(200, embedding_vecor_length, input_length=max_review_length))
	model.add(Dropout(0.2))
	model.add(LSTM(100))
	model.add(Dropout(0.2))
"""
	# Input layer
	model.add(Dense(neurons, kernel_initializer="uniform", input_dim=neurons, activation=activation_func))

	# First hidden layer
	model.add(Dense(int(neurons/2), kernel_initializer="uniform", activation=activation_func))

	model.add(Dense(int(neurons/2/2), kernel_initializer="uniform", activation=activation_func))
	model.add(Dropout(0.5))

	model.add(Dense(int(neurons/2/2/2), kernel_initializer="uniform", activation=activation_func))
	model.add(Dropout(0.2))

	# Output layer
	model.add(Dense(output_neurons, kernel_initializer="uniform", activation=output_activation_func))
	adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	model.compile(loss=loss_calculator, optimizer=adam)

	model.fit(x_train, y_train, epochs=20, batch_size=5, validation_data=(x_test, y_test))
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
	trainFile = 'mfcc_all_4'
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


#f = open('multi_kamer_1.pckl', 'rb')
#x_individual, dier = pickle.load(f)
#f.close()


if typeMulti:
	tensorModel(2, 'softmax', 'mean_squared_error', trainFile)
else: 
	tensorModel(1, 'sigmoid', 'binary_crossentropy', trainFile)