import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import math
import os


input = np.random.uniform(low = -1, high = 1, size=(200,))
m = np.random.uniform(low = -1, high = 1, size=(200,))

actualOutput = m * np.sin(6 * np.pi * m)* np.exp(-1 * np.power(m,2))
f = lambda x: x * np.sin(6 * np.pi * x)* np.exp(-1 * np.power(x,2))
print ("actualOutput: ", actualOutput)
print ("fucntion: ", f)

vf = np.vectorize(f)
output = vf(input)

model = Sequential();
model.add(Dense(1, activation='sigmoid', input_shape=(1,)));
#model.add(Dense(1, activation='logistic'));

model.compile(loss='mean_squared_error', optimizer='adam')

h = model.fit(input[:140],output[:140],epochs=100)
y = model.predict(input);
print ("H: ", h)
print ("Y: ", y)

plt.plot(input,y)
plt.plot(y, input)
plt.plot(m,actualOutput)
plt.show()
