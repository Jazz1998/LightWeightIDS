from __future__ import print_function
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import datetime
import math
import itertools
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from math import sqrt
import math
from tensorflow import keras
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation,Dropout,Flatten,Conv1D,MaxPooling1D
from tensorflow.keras import layers
from keras.layers.recurrent import LSTM
from keras import losses
from keras import optimizers
from sklearn.datasets import make_regression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
from keras.models import load_model
from timeit import default_timer as timer
import prepare_dataset as data


def build_model(input):
    model = Sequential()
    model.add(Dense(128, input_shape=(input[0], input[1])))
    model.add(Conv1D(filters=112, kernel_size=1, padding='valid', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling1D(pool_size=2, padding='valid'))
    model.add(Conv1D(filters=64, kernel_size=1, padding='valid', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling1D(pool_size=1, padding='valid'))
    model.add(Conv1D(filters=64, kernel_size=1, padding='valid', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling1D(pool_size=1, padding='valid'))
    model.add(Conv1D(filters=32, kernel_size=1, padding='valid', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling1D(pool_size=1, padding='valid'))
    model.add(Conv1D(filters=32, kernel_size=1, padding='valid', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling1D(pool_size=1, padding='valid'))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(100, activation='relu', kernel_initializer='uniform'))
    model.add(Dense(1, activation='relu', kernel_initializer='uniform'))
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    return model

'''
def build_model(input):
    model = Sequential()
    model.add(LSTM(128, input_shape=(input[0], input[1]), return_sequences=True))
    #model.add(LSTM(64, input_shape=(input[0], input[1]), return_sequences=False))
    model.add(LSTM(32, input_shape=(input[0], input[1]), return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation="relu", kernel_initializer="uniform"))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation="relu", kernel_initializer="uniform"))
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    return model
'''
'''
def build_model(input):
    model = Sequential()
    model.add(Dense(128, input_shape=(input[0], input[1])))
    model.add(Conv1D(filters=80, kernel_size=1, padding='same', activation='relu', kernel_initializer="glorot_uniform"))
    model.add(MaxPooling1D(pool_size=2, padding='valid'))
    model.add(Conv1D(filters=48, kernel_size=1, padding='same', activation='relu', kernel_initializer="glorot_uniform"))
    model.add(MaxPooling1D(pool_size=2, padding='valid'))
    model.add(LSTM(32, return_sequences=True))
    model.add(LSTM(16, return_sequences=False))
    model.add(Dense(32, activation="relu", kernel_initializer="uniform"))
    model.add(Dense(1, activation="relu", kernel_initializer="uniform"))
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    return model
'''
'''
def build_model(input):
    model = keras.Sequential(
        [
            layers.Input(shape=(input[0], input[1])),
            layers.Conv1D(filters=32, kernel_size=7, padding="same", strides=2, activation="relu"),
            layers.Dropout(rate=0.2),
            layers.Conv1D(filters=16, kernel_size=7, padding="same", strides=2, activation="relu"),
            layers.Conv1DTranspose(filters=16, kernel_size=7, padding="same", strides=2, activation="relu"),
            layers.Dropout(rate=0.2),
            layers.Conv1DTranspose(filters=32, kernel_size=7, padding="same", strides=2, activation="relu"),
            layers.Conv1DTranspose(filters=1, kernel_size=7, padding="same"),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    return model
'''

model = build_model([20, 5])

# Summary of the Model
print(model.summary())

# In[8]:

# 训练数据
from timeit import default_timer as timer
start = timer()
history = model.fit(data.x_train, data.y_train, batch_size=64, epochs=20, validation_split=0.2, verbose=2)
end = timer()
print(end - start)

# save model
model.save('cnn.h5')  # creates a HDF5 file 'my_model.h5'
#model.save('lstm.h5')
#model.save('cnn_lstem.h5')
#model.save('ac.h5')
#del model  # deletes the existing model

# In[9]:

# 返回history
history_dict = history.history
history_dict.keys()

# In[10]:

# 画出训练集和验证集的损失曲线

loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
loss_values50 = loss_values[0:150]
val_loss_values50 = val_loss_values[0:150]
epochs = range(1, len(loss_values50) + 1)
plt.plot(epochs, loss_values50, 'b', color='blue', label='Training loss')
plt.plot(epochs, val_loss_values50, 'b', color='red', label='Validation loss')
plt.rc('font', size=18)
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xticks(epochs)
fig = plt.gcf()
fig.set_size_inches(15, 7)
# fig.savefig('img/tcstest&validationlosscnn.png', dpi=300)
plt.show()

