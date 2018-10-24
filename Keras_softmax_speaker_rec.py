from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras import optimizers
import pickle

# Train data
trainFile = 'storeMulti-95'
testFile = 'storeMulti-Test10'

f = open(trainFile + '.pckl', 'rb')
x_train, y_train = pickle.load(f)
f.close()

# Test data
t = open(testFile + '.pckl', 'rb')
x_test, y_test = pickle.load(t)
t.close()

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
model.add(Dropout(0.2))
# Output layer
model.add(Dense(2, kernel_initializer="uniform", activation='softmax'))
adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='mean_squared_error', optimizer=adam)

model.fit(x_train, y_train, epochs=30, batch_size=5, validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, batch_size=1)
print(score)