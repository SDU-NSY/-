import numpy as np
import copy
import matplotlib.pyplot as plt


class HopfieldNet:
    def __init__(self, node_nums, Vs):  # Vs 是需要记住的状态
        self.node_nums = node_nums
        self.W = np.zeros((node_nums, node_nums))  # node_nums是神经元个数
        # self.learnW(Vs)  # method 2: learn weights by Hebb rule
        for i in range(node_nums):
            for j in range(node_nums):
                if i == j:
                    self.W[i, j] = 0
                else:
                    self.W[i, j] = sum([Vs[a][i] * Vs[a][j] for a in range(len(Vs))])  # a 代表是第几个需要记住的状态
        print(self.W)

    def learnW(self, Vs):
        for i in range(100):
            for j in range(len(Vs)):
                for c in range(len(Vs[j])):
                    for r in range(len(Vs[j])):
                        if c != r:
                            if Vs[j][c] == Vs[j][r]:
                                self.W[c, r] += 0.1
                            else:
                                self.W[c, r] -= 0.1
        print(self.W)

    def fit(self, v):
        new_v = np.zeros(len(v))  # v 是输入的向量,代表所有神经元的初始状态
        indexs = range(len(v))  # 顺序遍历所有神经元
        while np.sum(np.abs(new_v - v)) != 0:
            new_v = copy.deepcopy(v)
            for i in indexs:
                temp = np.dot(v, self.W[:, i])
                if temp >= 0:
                    v[i] = 1
                else:
                    v[i] = -1
        return v


def random_change(state):
    num_change = 0
    for i in range(len(state)):
        if np.random.rand() < 0.1:
            state[i] = - state[i]
            num_change += 1
            if num_change >= 3:
                break
    return state




zero = np.array([
    0, 1, 1, 1, 0,
    1, 0, 0, 0, 1,
    1, 0, 0, 0, 1,
    1, 0, 0, 0, 1,
    1, 0, 0, 0, 1,
    0, 1, 1, 1, 0
])

one = np.array([
    0, 1, 1, 0, 0,
    0, 0, 1, 0, 0,
    0, 0, 1, 0, 0,
    0, 0, 1, 0, 0,
    0, 0, 1, 0, 0,
    0, 0, 1, 0, 0
])

two = np.array([
    1, 1, 1, 0, 0,
    0, 0, 0, 1, 0,
    0, 0, 0, 1, 0,
    0, 1, 1, 0, 0,
    1, 0, 0, 0, 0,
    1, 1, 1, 1, 1,
])
three = np.array([
    1, 1, 1, 1, 1,
    0, 0, 0, 0, 1,
    1, 1, 1, 1, 0,
    1, 1, 1, 1, 0,
    0, 0, 0, 0, 1,
    1, 1, 1, 1, 1,
])
four = np.array([
    0, 0, 1, 1, 0,
    0, 1, 0, 1, 0,
    1, 0, 0, 1, 0,
    1, 1, 1, 1, 1,
    0, 0, 0, 1, 0,
    0, 0, 0, 1, 0,
])
five = np.array([
    1, 1, 1, 1, 1,
    1, 0, 0, 0, 0,
    1, 1, 1, 0, 0,
    0, 0, 1, 1, 1,
    0, 0, 0, 0, 1,
    1, 1, 1, 1, 1,
])
six = np.array([
    0, 0, 1, 1, 0,
    0, 1, 0, 0, 0,
    0, 1, 1, 1, 0,
    1, 0, 0, 0, 1,
    1, 0, 0, 0, 1,
    0, 1, 1, 1, 0,
])
seven = np.array([
    1, 1, 1, 1, 1,
    0, 0, 0, 0, 1,
    0, 0, 0, 1, 0,
    0, 0, 1, 0, 0,
    0, 1, 0, 0, 0,
    1, 0, 0, 0, 0,
])
eight = np.array([
    1, 1, 1, 1, 1,
    1, 0, 0, 0, 1,
    0, 1, 1, 1, 0,
    0, 1, 1, 1, 0,
    1, 0, 0, 0, 1,
    1, 1, 1, 1, 1,
])
nine = np.array([
    1, 1, 1, 1, 1,
    1, 0, 0, 0, 1,
    1, 0, 0, 0, 1,
    1, 1, 1, 1, 1,
    0, 0, 0, 0, 1,
    1, 1, 1, 1, 1,
])
total_states = [zero, one, two, three, four, five, six, seven, eight, nine]
for i in total_states:
    for j in range(len(i)):
        if i[j] == 0:
            i[j] = -1
net = HopfieldNet(30, total_states)
change_id = 6
change_fit = random_change(total_states[change_id].copy())
plt.imshow(np.matrix(total_states[change_id].reshape(6, 5)))
plt.show()

plt.imshow(np.matrix(change_fit.reshape(6, 5)))
plt.show()

ans = net.fit(change_fit)
plt.imshow(np.matrix(ans.reshape(6, 5)))
plt.show()

