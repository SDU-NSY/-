import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras import losses
from keras.utils import plot_model

# 最多8位二进制
BINARY_DIM = 8

# 将整数表示成为binary_dim位的二进制数，高位用0补齐
def int_2_binary(number, binary_dim):
    binary_list = list(map(lambda x: int(x), bin(number)[2:]))
    number_dim = len(binary_list)
    result_list = [0]*(binary_dim-number_dim)+binary_list
    return result_list

# 将一个二进制数组转为整数
def binary2int(binary_array):
    out = 0
    for index, x in enumerate(reversed(binary_array)):
        out += x * pow(2, index)
    return out

# 将[0,2**BINARY_DIM)所有数表示成二进制
binary = np.array([int_2_binary(x, BINARY_DIM) for x in range(2**BINARY_DIM)])
# print(binary)

# 样本的输入向量和输出向量
dataX = []
dataY = []
for i in range(binary.shape[0]):
    for j in range(binary.shape[0]):
        dataX.append(np.append(binary[i], binary[j]))
        dataY.append(int_2_binary(i+j, BINARY_DIM+1))

# print(dataX)
# print(dataY)

# 重新特征X和目标变量Y数组，适应LSTM模型的输入和输出
X = np.reshape(dataX, (len(dataX), 2*BINARY_DIM, 1))
# print(X.shape)
Y = np.array(dataY)
# print(dataY.shape)

# 定义LSTM模型
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(Y.shape[1], activation='sigmoid'))
model.compile(loss=losses.mean_squared_error, optimizer='adam')
# print(model.summary())

# plot model
# plot_model(model, to_file=r'./model.png', show_shapes=True)
# train model
epochs = 100
model.fit(X, Y, epochs=epochs, batch_size=128)
# save model
mp = r'./LSTM_Operation.h5'
model.save(mp)

# use LSTM model to predict
for _ in range(100):
    start = np.random.randint(0, len(dataX)-1)
    # print(dataX[start])
    number1 = dataX[start][0:BINARY_DIM]
    number2 = dataX[start][BINARY_DIM:]
    print('='*30)
    print('%s: %s'%(number1, binary2int(number1)))
    print('%s: %s'%(number2, binary2int(number2)))
    sample = np.reshape(X[start], (1, 2*BINARY_DIM, 1))
    predict = np.round(model.predict(sample), 0).astype(np.int32)[0]
    print('%s: %s'%(predict, binary2int(predict)))
