import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('s_normal0.csv')
df1 = pd.read_csv('s_attack.csv')

# 弃用一些字段
drop_columns = ['Normal/Attack']
df = df.drop(drop_columns, axis=1)
df1 = df1.drop(drop_columns, axis=1)

# 将Timestamp字段设置为索引
df = df.set_index(' Timestamp')
df1 = df1.set_index(' Timestamp')

# 攻击是1，正常是0
# df1.loc[df1['Normal/Attack'] != 'Normal', 'Normal/Attack'] = 1
# df1.loc[df1['Normal/Attack'] == 'Normal', 'Normal/Attack'] = 0

# 标准化
df['FIT101'] = df['FIT101'] / 1000
df['LIT101'] = df['LIT101'] / 1000
df1['FIT101'] = df1['FIT101'] / 1000
df1['LIT101'] = df1['LIT101'] / 1000
''''
df['MV101'] = df['MV101'] / 1000
df['P101'] = df['P101'] / 1000
df['P102'] = df['P102'] / 1000
df1['MV101'] = df['MV101'] / 1000
df1['P101'] = df['P101'] / 1000
df1['P102'] = df['P102'] / 1000
df['FIT201'] = df['FIT201'] / 10
df['MV201'] = df['MV201'] / 10
df['P202'] = df['P202'] / 10
df['P203'] = df['P203'] / 10
df['P204'] = df['P204'] / 10
df['P205'] = df['P205'] / 10
df['P206'] = df['P206'] / 10
df['DPIT301'] = df['DPIT301'] / 100
df['FIT301'] = df['FIT301'] / 10
df['LIT301'] = df['LIT301'] / 1000
df['MV301'] = df['MV301'] / 10
df['MV302'] = df['MV302'] / 10
df['MV303'] = df['MV303'] / 10
df['MV304'] = df['MV304'] / 10
df['P301'] = df['P301'] / 10
df['P302'] = df['P302'] / 10
df['FIT401'] = df['FIT401'] / 10
df['LIT401'] = df['LIT401'] / 1000
df['P401'] = df['P401'] / 10
df['P402'] = df['P402'] / 10
df['P403'] = df['P403'] / 10
df['P404'] = df['P404'] / 10
df['UV401'] = df['UV401'] / 10
df['FIT501'] = df['FIT501'] / 10
df['FIT502'] = df['FIT502'] / 10
df['P501'] = df['P501'] / 10
df['P502'] = df['P502'] / 10
df['FIT601'] = df['FIT601'] / 10
df['P601'] = df['P601'] / 10
df['P602'] = df['P602'] / 10
df['P603'] = df['P603'] / 10
'''
print(df.head())

# 将dataframe 转化为 array
data = df.iloc[:, :].values
print(len(data))
data1 = df1.iloc[:, :].values

# 数据切分
result = []
result1 = []
time_steps = 21

for i in range(len(data) - time_steps):
    result.append(data[i:i + time_steps])

result = np.array(result)

for i in range(len(data1) - time_steps):
    result1.append(data1[i:i + time_steps])

result1 = np.array(result1)

# 训练集和测试集的数据量划分
print(len(result1))
train_size = int(0.9 * len(result))
test_size = int(0.7 * len(result1))

# 训练集切分
train = result[:, :]

x_train = result[:, :-1]
y_train = result[:, -1][:, 1]#只预测一个值
#y_train = result[:, -1]#预测多个值

x_test = result1[test_size:, :-1]
y_test = result1[test_size:, -1][:, 1]#只预测一个值
#y_test = result1[test_size:, -1]#预测多个值

feature_nums = len(df.columns)

# 数据重塑
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2])

print("X_train", x_train.shape)
print("y_train", y_train.shape)
print("X_test", x_test.shape)
print("y_test", y_test.shape)

df2 = pd.read_csv('s_attack.csv')

# 将Timestamp字段设置为索引
df2 = df2.set_index(' Timestamp')

#攻击是1，正常是0
df2.loc[df2['Normal/Attack'] != 'Normal', 'Normal/Attack'] = 1
df2.loc[df2['Normal/Attack'] == 'Normal', 'Normal/Attack'] = 0

df2['FIT101'] = df2['FIT101'] / 1000
df2['LIT101'] = df2['LIT101'] / 1000

# 将dataframe 转化为 array
data2 = df2.iloc[:,:].values

#数据切分
result2 = []
time_steps = 21
#test_size = int(0.7*len(result2))
for i in range(len(data2)-time_steps):
    result2.append(data2[i:i+time_steps])

result2 = np.array(result2)

y_test2 = result2[test_size:, -1]