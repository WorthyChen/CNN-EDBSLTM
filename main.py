# multivariate multi-step encoder-decoder lstm example
import keras
from numpy import array
from numpy import hstack
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed

from keras.layers import Bidirectional

from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Flatten


import tensorflow as tf
import datetime
import math

import numpy as np
import keras.backend as K
import xlrd
import os
os.chdir(os.getcwd())
os.environ['CUDA_VISIBLE_DEVICES'] = '/gpu:0'


# split a multivariate sequence into samples
def split_sequences(sequences, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
		# check if we are beyond the dataset
		if out_end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)
#定义函数读取exl表格
def rdxls(col, start, end, sheet):
	list_values = []
	for x in range(start, end):
		values = []
		row = sheet.col(col)[x].value
		list_values.append(row)
	data = np.asarray(list_values)
	data = data.reshape((len(data), 1))
	return data


#定义数据集
workbook = xlrd.open_workbook('data\data.xlsx')
sheet = workbook.sheets()[0]

input1 = rdxls(0, 400, 500, sheet)
input2 = rdxls(1, 400, 500, sheet)
input3 = rdxls(2, 400, 500, sheet)
input4 = rdxls(3, 400, 500, sheet)
input5 = rdxls(4, 400, 500, sheet)
input6 = rdxls(5, 400, 500, sheet)
input7 = rdxls(6, 400, 500, sheet)
input8 = rdxls(7, 400, 500, sheet)
input9 = rdxls(8, 400, 500, sheet)
input10 = rdxls(9, 400, 500, sheet)
input11 = rdxls(10, 400, 500, sheet)
input12 = rdxls(11, 400, 500, sheet)
print(input1)
dataset = hstack((input1, input2, input3, input4, input5, input6, input7, input8, input9, input10, input11, input12))
print(dataset)
# choose a number of time steps 步长
n_steps_in, n_steps_out = 8, 2

# covert into input/output
X, y = split_sequences(dataset, n_steps_in, n_steps_out)

# the dataset knows the number of features, e.g. 2
n_features = X.shape[2]
#print(n_features)
# define loss function  定义损失函数
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt((K.mean(K.square((y_pred - y_true)/y_true), axis=-1)))*100


# # define model   标准LSTM
# model = Sequential()
# model.add(LSTM(200, activation='relu', input_shape=(n_steps_in, n_features)))
# model.add(RepeatVector(n_steps_out))
# model.add(LSTM(200, activation='relu', return_sequences=True))
# model.add(TimeDistributed(Dense(n_features)))
# model.compile(optimizer='adam', loss=root_mean_squared_error)


# # CNNLSTM模型
# #定义子序列的拆分比例
# n_seq = 2
# n_s_seq = int(n_steps_in/n_seq)
# X = X.reshape((X.shape[0], n_seq, n_s_seq, n_features))
# model = Sequential()
# model.add(TimeDistributed(Conv1D(filters=81, kernel_size=2, activation='relu'), input_shape=(None, n_s_seq, n_features)))
# model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
# model.add(TimeDistributed(Flatten()))
# model.add(LSTM(200, activation='relu'))
# model.add(RepeatVector(n_steps_out))
# model.add(LSTM(200, activation='relu', return_sequences=True))
# model.add(TimeDistributed(Dense(n_features)))
# model.compile(optimizer='adam', loss=root_mean_squared_error)


# # define model   双向LSTM
# model = Sequential()
# model.add(Bidirectional(LSTM(200, activation='relu', input_shape=(n_steps_in, n_features))))
# model.add(RepeatVector(n_steps_out))
# model.add(Bidirectional(LSTM(200, activation='relu', return_sequences=True)))
# model.add(TimeDistributed(Dense(n_features)))
# model.compile(optimizer='adam', loss=root_mean_squared_error)


# define model   CNN双向LSTM
#定义子序列的拆分比例
n_seq = 2
n_s_seq = int(n_steps_in/n_seq)
X = X.reshape((X.shape[0], n_seq, n_s_seq, n_features))
model = Sequential()
model.add(TimeDistributed(Conv1D(filters=81, kernel_size=2, activation='relu'), input_shape=(None, n_s_seq, n_features)))
model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
model.add(TimeDistributed(Flatten()))
model.add(Bidirectional(LSTM(200, activation='relu')))
model.add(RepeatVector(n_steps_out))
model.add(Bidirectional(LSTM(200, activation='relu', return_sequences=True)))
model.add(TimeDistributed(Dense(n_features)))
model.compile(optimizer='adam', loss=root_mean_squared_error)






# test tb
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# fit model
# model.fit(X, y, epochs=300, validation_split=0.2, verbose=1, shuffle=False, callbacks=[tensorboard_callback])
history = model.fit(X, y, epochs=500, validation_split=0.2, verbose=1, shuffle=False, callbacks=[tensorboard_callback])

# print(history.history)
# f = open('CNNEDBLSTM.txt', 'a')
# f.write(str(history.history['val_loss']))
# f.write('\n')
# f.write(str(history.epoch))
# f.close()

#demonstrate prediction
x_input = array([[-4.169857, 33.166969, 21.059763, 154.770576, -116.921195, -61.462602, 93.08574, 22.810073, 175.824206, 146.04614, 152.77581, 69.994408],
                 [-4.867398, 33.059429,	22.105852,	153.863247,	-117.313052,	-61.624988,	93.500702,	23.195869,	176.162389,	145.81227,	153.09487,	71.527054],
				 [-6.219046,	32.688941,	22.11186,	153.73059,	-117.716073,	-61.571074,	93.062867,	24.142904,	175.687953,	145.708678,	153.045102,	72.417263],
                 [-5.484129,	32.861755,	22.111623,	153.407071,	-117.658937,	-61.749264,	92.412966,	25.228471,	175.567492,	145.387182,	153.083346,	71.273886],
                 [-4.947102,	33.526602,	21.267107,	153.988822,	-117.228138,	-61.647614,	92.639599,	24.824892,	175.312882,	145.29918,	152.329637,	70.55238],
                 [-6.317924,	33.787216,	21.09935,	154.672883,	-116.6155,	-61.859175,	93.399842,	24.928542,	175.235835,	145.122777,	153.290942,	69.940175],
                 [-6.890814,	34.246313,	21.266023,	155.047301,	-116.716504,	-61.70697,	94.050089,	24.34064,	175.24088,	145.113196,	153.86795,	70.635626],
                 [-9.080958,	34.079685,	20.782186,	154.910914,	-116.851086,	-61.851455,	94.150457,	24.556946,	175.576247,	145.262202,	153.787668,	71.617565],
])

# x_input = x_input.reshape(1, n_steps_in, n_features)


#CNN专用
x_input = x_input.reshape(1, n_seq, n_s_seq, n_features)

yhat = model.predict(x_input, verbose=0)
print(yhat)










