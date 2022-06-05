import numpy as np
import matplotlib.pyplot as plt

class Hebb(object):
    def __init__(self):
        self.weights = np.zeros((30, 30))

    def train(self, input_vector, steps, rate):
        """
        input_vector: 用于训练的输入向量
        steps: 迭代次数
        rate: 学习率
        """
        for i in range(steps):
            for vec in input_vector:
                vec = np.matrix(vec)
                self.weights = self.weights + rate * vec.getT().dot(vec) # 列向量矩阵乘行向量得到矩阵
        return self.weights

    def predict(self, input_vector):
        return self.weights.dot(np.matrix(input_vector).getT())


def init_data():
    zero = np.array([[1, 1, 1, 1, 1],
                     [1, -1, -1, -1, 1],
                     [1, -1, -1, -1, 1],
                     [1, -1, -1, -1, 1],
                     [1, -1, -1, -1, 1],
                     [1, 1, 1, 1, 1]])

    one = np.array([[-1, 1, 1, -1, -1],
                     [-1, -1, 1, -1, -1],
                     [-1, -1, 1, -1, -1],
                     [-1, -1, 1, -1, -1],
                     [-1, -1, 1, -1, -1],
                     [-1, -1, 1, -1, -1]])

    two = np.array([[-1, 1, 1, 1, -1],
                     [-1, -1, -1, 1, -1],
                     [-1, -1, -1, 1, -1],
                     [-1, 1, 1, 1, -1],
                     [-1, 1, -1, -1, -1],
                     [-1, 1, 1, 1, -1]])
    return zero, one, two


def add_noise(data):
    for idx in range(len(data)):
        if np.random.random() < 0.2:
            data[idx] = -data[idx]
    return data


def acti_func(data):
    for index in range(len(data)):
        if data[index] > 0:
            data[index] = 1
        else:
            data[index] = -1
    return data


zero, one, two = init_data()
zero, one, two = zero.reshape(-1), one.reshape(-1), two.reshape(-1)

hebb = Hebb()
hebb.train([zero, one, two], 600, 0.2)
data_change = add_noise(two)

plt.imshow(np.matrix(data_change.reshape(6, 5)))
plt.show()

res = acti_func(hebb.predict(data_change))
plt.imshow(np.matrix(res.reshape(6, 5)))
plt.show()





