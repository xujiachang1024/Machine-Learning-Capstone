from __future__ import print_function
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, TimeDistributed


print('Loading data...')


# splits = np.load('data.npz')

df = pd.read_csv("kenya_oct_15_data_labeled.csv")
x_train = df[[' x', ' y', ' z']][:10000]
y_train = df['label'][:10000]
x_test = df[[' x', ' y', ' z']][10000:14000]
y_test = df['label'][10000:14000]

maes = []
r2s = []
rmses = []

x_train = np.array(x_train)
x_train = x_train.reshape((x_train.shape[0], 1,x_train.shape[1]))
y_train = np.array(y_train)
x_test = np.array(x_test)
x_test = x_test.reshape((x_test.shape[0],1, x_test.shape[1]))
y_test = np.array(y_test)

print(x_train.shape)
print(x_train.shape)

model = Sequential()
model.add(Bidirectional(LSTM(12), input_shape=(1, 3)))
# model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='mean_absolute_error', optimizer='rmsprop', metrics=['mse'])

print('Train...')
history = model.fit(x=x_train, y=y_train,
            batch_size=32,
            epochs=8,
            validation_data=(x_test, y_test))

maes.append(history.history['val_loss'][-1])
rmses.append(np.sqrt(history.history['val_mean_squared_error'][-1]))

print(maes)
print("MAE: ", np.array(maes).mean())
print("RMSE: ", np.array(rmses).mean())