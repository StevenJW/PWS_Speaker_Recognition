from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras import optimizers
import pickle

'''if typeMulti == True:
	output_neurons = 2
	output_activation_func = 'softmax'
	loss_calculator = 'mean_squared_error'
	trainFile = 'store_1_multi'
else:
	output_neurons = 1
	output_activation_func = 'sigmoid'
	loss_calculator = 'binary_crossentropy'
	trainFile = 'store_1_sigmoid'''
def tensorModel(output_neurons, output_activation_func, loss_calculator, trainFile):
	neurons = 800
	activation_func = 'relu'

	model = Sequential()
	# Input layer
	model.add(Dense(neurons, kernel_initializer="uniform", input_dim=neurons, activation=activation_func))

	# First hidden layer
	model.add(Dense(neurons, kernel_initializer="uniform", activation=activation_func))
	model.add(Dropout(0.2))
	# Second hidden layer
	model.add(Dense(neurons, kernel_initializer="uniform", activation=activation_func))
	model.add(Dropout(0.5))

	# Output layer
	model.add(Dense(output_neurons, kernel_initializer="uniform", activation=output_activation_func))
	adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	model.compile(loss=loss_calculator, optimizer=adam)

	model.fit(x_train, y_train, epochs=40, batch_size=5, validation_data=(x_test, y_test))
	score = model.evaluate(x_test, y_test, batch_size=1)
	print(score)

typeMulti = False
trainFile = 'store_1_sigmoid'

f = open(trainFile + '.pckl', 'rb')
x_train, y_train, x_test, y_test = pickle.load(f)
f.close()

if typeMulti:
	tensorModel(2, 'softmax', 'mean_squared_error', trainFile)
else: 
	tensorModel(1, 'sigmoid', 'binary_crossentropy', trainFile)



