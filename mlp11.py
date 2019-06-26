import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.datasets import mnist
from keras.optimizers import SGD
from keras.initializers import RandomNormal
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import math
import os


np.random.seed(7)
input = np.random.uniform(low = -1, high = 1, size=(200,))
m= np.linspace(-1,1,200)

actualOutput = m * np.sin(6 * np.pi * m)* np.exp(-1 * np.power(m,2))
f = lambda x: x * np.sin(6 * np.pi * x)* np.exp(-1 * np.power(x,2))

vf = np.vectorize(f)
output = vf(input)

model = Sequential();
model.add(Dense(784,activation='sigmoid', input_shape=(1,), kernel_initializer=RandomNormal(mean=0., stddev=30),
bias_initializer=RandomNormal(mean=0., stddev=10)));
model.add(Dense(1, activation='linear'));
model.compile(loss='mean_squared_error', optimizer='adam')
h = model.fit(input[:140],output[:140],epochs=100)
s = slice(141,200,1)
y = model.predict(input[s]);

plt.plot(input[s], y, 'r')
plt.plot(actualOutput, 'g')
plt.plot(y, output, 'y')

plt.show()
