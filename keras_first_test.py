from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
import pickle
from data_conversion import convert_train_data

# Train data
f = open('store1.pckl', 'rb')
x_train, y_train = pickle.load(f)
f.close()

# Test data
t = open('storeTest1.pckl', 'rb')
x_test, y_test = pickle.load(t)
t.close()

neurons = 800
activation_func = 'relu'

model = Sequential()
model.add(Dense(neurons, kernel_initializer="uniform", input_dim=neurons))
model.add(Activation(activation_func))
model.add(Dropout(0.2))
model.add(Dense(neurons, kernel_initializer="uniform"))
model.add(Activation(activation_func))
model.add(Dropout(0.2))
model.add(Dense(neurons, kernel_initializer="uniform"))
model.add(Activation(activation_func))
model.add(Dense(1, kernel_initializer="uniform"))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam')

model.fit(x_train, y_train, epochs=100, batch_size=5)
score = model.evaluate(x_test, y_test, batch_size=1)
print(score)