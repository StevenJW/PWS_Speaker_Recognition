from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from data_conversion import convert_train_data

x_train, y_train = convert_train_data('E')

model = Sequential()
model.add(Dense(800, kernel_initializer="uniform", input_dim=800))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(800, kernel_initializer="uniform"))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2, kernel_initializer="uniform"))
model.add(Activation('softmax'))

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(x_train, y_train, epochs=1, batch_size=1)
#score = model.evaluate(X_test, y_test, batch_size=16)
