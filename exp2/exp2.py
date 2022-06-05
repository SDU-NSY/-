import numpy as np
from random import choice
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

'''
1.将权重初始化为 0 或一个很小的随机数
2.对于每个训练样本 x(i) 执行下列步骤： 
   计算输出值 y^.
    更新权重
'''


def load_data_and():
    input_data = [[1, 1], [0, 0], [1, 0], [0, 1]]
    labels = [1, 0, 0, 0]
    return input_data, labels


def load_data_or():
    input_data = [[1, 1], [0, 0], [1, 0], [0, 1]]
    labels = [1, 0, 1, 1]
    return input_data, labels


def load_data_not():
    input_data = [[0], [1]]
    labels = [1, 0]
    return input_data, labels


def activate_func(x):
    return 0 if x < 0 else 1


def train_pre(input_data, y, iteration, rate):
    """
    input_data:输入数据
    y：标签列表
    iteration：训练轮数
    rate:学习率
    """
    w = np.random.rand(len(input_data[0]))  # 随机生成[0,1)之间,作为初始化w
    bias = 0.0  # 偏置

    for i in range(iteration):  # 迭代次数
        samples = zip(input_data, y)
        for (input_i, label) in samples:  # 对每一组样本
            # 计算f(w*xi+b),此时x有两个
            result = float(sum(input_i * w)) + bias
            # print(result)
            # result = float(sum(result))
            y_pred = float(activate_func(result))  # 计算输出值 y^
            w = w + rate * (label - y_pred) * np.array(input_i)  # 更新权重
            bias = bias + rate * (label - y_pred)  # 更新bias
    return w, bias


def predict(input_i, w, b):
    result = sum(input_i * w) + b
    y_pred = float(activate_func(result))
    print(y_pred)


if __name__ == '__main__':
    input_data, y = load_data_not()
    w, b = train_pre(input_data, y, 100, 0.01)
    predict([0], w, b)