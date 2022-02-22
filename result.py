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
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv1D, MaxPooling1D
from keras.layers.recurrent import LSTM
from keras import losses
from keras import optimizers
from sklearn.datasets import make_regression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
from tensorflow.keras.models import load_model
import prepare_dataset as data
from sklearn.metrics import mean_absolute_error

x_train = data.x_train
y_train = data.y_train
x_test = data.x_test
y_test = data.y_test
y_test2 = data.y_test2
#load model
model = load_model('cnn.h5')
#model = load_model('lstm.h5')
#model = load_model('cnn_lstem.h5')
#model = load_model('ac.h5')

#检测阈值
p = model.predict(x_test)
p1 = model.predict(x_train)
print(p1)
print('-------------------')
print(y_train)

plt.plot(p1[:,0], color='red', label='prediction')
plt.plot(y_train, color='blue', label='y_train')
plt.xlabel('time')
plt.ylabel('Close Value (scaled)')
plt.legend(loc='upper left')
fig = plt.gcf()
fig.set_size_inches(15, 7)
#fig.savefig('img/tcstestcnn.png', dpi=300)
plt.show()
'''
plt.plot(p1[:,1], color='red', label='prediction')
plt.plot(y_train, color='blue', label='y_train')
plt.xlabel('time')
plt.ylabel('Close Value (scaled)')
plt.legend(loc='upper left')
fig = plt.gcf()
fig.set_size_inches(15, 7)
#fig.savefig('img/tcstestcnn.png', dpi=300)
plt.show()
'''
# In[17]:

threshold = 0
for i in range(len(y_train)):
    if np.abs(p1[i] - y_train[i]) > threshold:
        threshold = np.abs(p1[i] - y_train[i])

print(threshold)
plt.plot(p1, color='red', label='prediction')
plt.plot(y_train, color='blue', label='y_train')
plt.xlabel('time')
plt.ylabel('Close Value (scaled)')
plt.legend(loc='upper left')
fig = plt.gcf()
fig.set_size_inches(15, 7)
#fig.savefig('img/tcstestcnn.png', dpi=300)
plt.show()

# In[14]:

# In[15]:

# 画出真实值和测试集的预测值之间的误差图像
plt.plot(p, color='red', label='prediction')
plt.plot(y_test2[:,1], color='blue', label='y_test')
plt.xlabel('time')
plt.ylabel('Close Value (scaled)')
plt.legend(loc='upper left')
fig = plt.gcf()
fig.set_size_inches(15, 7)
#fig.savefig('img/tcstestcnn.png', dpi=300)
#计算阈值
for i in range(len(y_test2)):
    #if y_test2[i,5] == 1:
    if np.abs(p[i] - y_test2[i,1]) > threshold:
        plt.scatter(i, y_test2[i,1], s=10, c='green')

plt.show()

model.metrics_names

# In[18]
trainScore = model.evaluate(x_train, y_train, verbose=0)
testScore = model.evaluate(x_test, y_test, verbose=0)
print(trainScore)
print(testScore)

y = data.y_test * 1000  # 原始数据经过除以10000进行缩放，因此乘以10000,返回到原始数据规模
y_pred = p.reshape(134970)  # 测试集数据大小为265
y_pred = y_pred * 1000  # 原始数据经过除以10000进行缩放，因此乘以10000,返回到原始数据规模

print('Trainscore RMSE \tTrain Mean abs Error \tTestscore Rmse \t Test Mean abs Error')
print('%.9f \t\t %.9f \t\t %.9f \t\t %.9f' % (math.sqrt(trainScore[0]), trainScore[1], math.sqrt(testScore[0]), testScore[1]))

# In[19]:

print('mean absolute error \t mean absolute percentage error')
print(' %.9f \t\t\t %.9f' % (mean_absolute_error(y, y_pred), (np.mean(np.abs((y - y_pred) / y)) * 100)))
'''
# In[20]:

#  训练集、验证集、测试集 之间的比较

Y = np.concatenate((y_train, y_test),axis = 0)
P = np.concatenate((p1, p), axis = 0)
#plotting the complete Y set with predicted values on x_train and x_test(variable p1 & p respectively given above)
#for
plt.plot(P,color='red', label='prediction on training samples')
#for validating samples
z = np.array(range(848,1060))
plt.plot(P, color = 'black',label ='prediction on validating samples')
#for testing samples
x = np.array(range(1060,1325))
plt.plot(x,color = 'green',label ='prediction on testing samples(x_test)')

plt.plot(Y, color='blue', label='Y')
plt.legend(loc='upper left')
fig = plt.gcf()
fig.set_size_inches(20,12)
plt.show()
'''