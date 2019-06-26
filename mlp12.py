import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import math
import os

np.random.seed(7)
x = np.random.uniform(low = -2, high = 2, size=(100,))
m = np.linspace(-2,2,100)

actualOutput = m * np.sin(6 * np.pi * m)* np.exp(-1 * np.power(m,2))
output = x * np.sin(6 * np.pi * x)* np.exp(-1 * np.power(x,2))

#actualOutput = np.exp(-1 * np.power(m,2)) * np.arctan(m) * np.sin(4 * np.pi * m)
#output = x * np.exp(-1 * np.power(x,2)) * np.arctan(x) * np.sin(4 * np.pi * x)

model = Sequential();
model.add(Dense(1000, activation='sigmoid', input_shape=(1,), kernel_initializer=RandomNormal(mean=0., stddev=30),
bias_initializer=RandomNormal(mean=0., stddev=10)));
model.add(Dense(1,));
#model.compile(loss='mean_squared_error', optimizer='adam')
model.compile(loss='mean_squared_error', optimizer='adam')
h = model.fit(x, output, epochs=1000, validation_split=0.3, verbose=2)
y = model.predict(m);

#plt.plot(m, y, 'r')
#plt.plot(m, actualOutput, 'g')

plt.plot(h.history['loss'])
plt.plot(h.history['val_loss'])
plt.title('100 data points')

plt.ylabel('loss')
plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')
plt.show()
