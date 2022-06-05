import numpy as np

def softmax(x):
    x = np.array(x)
    max_x = np.max(x)
    return np.exp(x - max_x) / np.sum(np.exp(x - max_x))

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def tanh(x):
    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

class myLSTM:
    def __init__(self, data_dim, hidden_dim=100):
        # data_dim: 词向量维度，即词典长度; hidden_dim: 隐单元维度
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        # 初始化权重向量
        self.whi, self.wxi, self.bi = self._init_wh_wx()
        self.whf, self.wxf, self.bf = self._init_wh_wx()
        self.who, self.wxo, self.bo = self._init_wh_wx()
        self.wha, self.wxa, self.ba = self._init_wh_wx()
        self.wy = np.random.uniform(-np.sqrt(1.0 / self.hidden_dim),
                                    np.sqrt(1.0 / self.hidden_dim),
                                    (self.data_dim, self.hidden_dim))

        self.by = np.random.uniform(-np.sqrt(1.0 / self.hidden_dim),
                          np.sqrt(1.0 / self.hidden_dim),
                          (self.data_dim, 1))

    def _init_wh_wx(self):
        wh = np.random.uniform(-np.sqrt(1.0 / self.hidden_dim),
                               np.sqrt(1.0 / self.hidden_dim),
                               (self.hidden_dim, self.hidden_dim))

        wx = np.random.uniform(-np.sqrt(1.0 / self.data_dim),
                               np.sqrt(1.0 / self.data_dim),
                              (self.hidden_dim, self.data_dim))

        b = np.random.uniform(-np.sqrt(1.0 / self.data_dim),
                              np.sqrt(1.0 / self.data_dim),
                             (self.hidden_dim, 1))

        return wh, wx, b

    def _init_s(self, T):
        # input gate state
        iss = np.array([np.zeros((self.hidden_dim, 1))] * (T + 1))

        # forget gate state
        fss = np.array([np.zeros((self.hidden_dim, 1))] * (T + 1))

        # output gate state
        oss = np.array([np.zeros((self.hidden_dim, 1))] * (T + 1))

        # current input state
        ass = np.array([np.zeros((self.hidden_dim, 1))] * (T + 1))

        #  hidden state
        hss = np.array([np.zeros((self.hidden_dim, 1))] * (T + 1))

        # cell state
        css = np.array([np.zeros((self.hidden_dim, 1))] * (T + 1))

        # output value
        ys = np.array([np.zeros((self.hidden_dim, 1))] * T)

        return {'iss': iss, 'fss': fss, 'oss': oss, 'ass': ass, 'hss': hss, 'css': css, 'ys': ys}

    def forward(self, x):
        T = len(x)  # 时间步的个数
        stats = self._init_s(T)
        for t in range(T):
            ht_pre = np.array(stats['hss'][t-1]).reshape(-1, 1)
            # input gate
            stats['iss'][t] = self._cal_gate(self.whi, self.wxi,
                                             self.bi, ht_pre, x[t], sigmoid)
            # forget gate
            stats['fss'][t] = self._cal_gate(self.whf, self.wxf,
                                             self.bf, ht_pre, x[t], sigmoid)
            # output gate
            stats['oss'][t] = self._cal_gate(self.who, self.wxo,
                                             self.bo, ht_pre, x[t], sigmoid)
            # current input state
            stats['ass'][t] = self._cal_gate(self.wha, self.wxa,
                                             self.ba, ht_pre, x[t], tanh)
            # cell state, ct = ft * ct_pre + it * at
            stats['css'][t] = stats['fss'][t] * stats['css'][t - 1] + stats['iss'][t] * stats['ass'][t]
            # hidden state, ht = ot * tanh(ct)
            stats['hss'][t] = stats['oss'][t] * tanh(stats['css'][t])
            # output value, yt = softmax(self.wy.dot(ht) + self.by)
            stats['ys'][t] = softmax(self.wy.dot(stats['hss'][t]) +
                                     self.by)
            return stats

    def _cal_gate(self, wh, wx, b, ht_pre, x, activation):
        return activation(wh.dot(ht_pre) + wx[:, x].reshape(-1, 1) + b)

    # 预测输出，单个 x
    def predict(self, x):
        stats = self.forward(x)
        pre_y = np.argmax(stats['ys'].reshape(len(x), -1), axis=1)
        return pre_y

    # 计算损失， softmax 交叉熵损失函数， (x,y)为多个样本
    def loss(self, x, y):
        cost = 0

        for i in range(len(y)):
            stats = self.forward(x[i])
            # 取出 y[i] 中每一时刻对应的预测值
            pre_yi = stats['ys'][range(len(y[i])), y[i]]
            cost -= np.sum(np.log(pre_yi))
        # 统计所有 y 中词的个数, 计算平均损失
        N = np.sum([len(yi) for yi in y])
        ave_loss = cost / N
        return ave_loss

    # 初始化偏导数 dwh, dwx, db
    def _init_wh_wx_grad(self):
        dwh = np.zeros(self.whi.shape)
        dwx = np.zeros(self.wxi.shape)
        db = np.zeros(self.bi.shape)
        return dwh, dwx, db

    # 求梯度, (x,y)为一个样本
    def bptt(self, x, y):
        dwhi, dwxi, dbi = self._init_wh_wx_grad()
        dwhf, dwxf, dbf = self._init_wh_wx_grad()
        dwho, dwxo, dbo = self._init_wh_wx_grad()
        dwha, dwxa, dba = self._init_wh_wx_grad()
        dwy, dby = np.zeros(self.wy.shape), np.zeros(self.by.shape)
        # 初始化 delta_ct，因为后向传播过程中，此值需要累加
        delta_ct = np.zeros((self.hidden_dim, 1))
        # 前向计算
        stats = self.forward(x)
        # 目标函数对输出 y 的偏导数
        delta_o = stats['ys']
        delta_o[np.arange(len(y)), y] -= 1

        for t in np.arange(len(y))[::-1]:
            # 输出层 wy, by 的偏导数，由于所有时刻的输出共享输出权值矩阵，故所有时刻累加
            dwy += delta_o[t].dot(stats['hss'][t].reshape(1, -1))
            dby += delta_o[t]
            # 目标函数对隐藏状态的偏导数
            delta_ht = self.wy.T.dot(delta_o[t])
            # 各个门及状态单元的偏导数
            delta_ot = delta_ht * tanh(stats['css'][t])
            delta_ct += delta_ht * stats['oss'][t] * (1 - tanh(stats['css'][t]) ** 2)
            delta_it = delta_ct * stats['ass'][t]
            delta_ft = delta_ct * stats['css'][t - 1]
            delta_at = delta_ct * stats['iss'][t]
            delta_at_net = delta_at * (1 - stats['ass'][t] ** 2)
            delta_it_net = delta_it * stats['iss'][t] * (1 - stats['iss'][t])
            delta_ft_net = delta_ft * stats['fss'][t] * (1 - stats['fss'][t])
            delta_ot_net = delta_ot * stats['oss'][t] * (1 - stats['oss'][t])
            # 各个门及状态单元的偏导数
            delta_ot = delta_ht * tanh(stats['css'][t])
            delta_ct += delta_ht * stats['oss'][t] * (1 - tanh(stats['css'][t]) ** 2)
            delta_it = delta_ct * stats['ass'][t]
            delta_ft = delta_ct * stats['css'][t - 1]
            delta_at = delta_ct * stats['iss'][t]
            delta_at_net = delta_at * (1 - stats['ass'][t] ** 2)
            delta_it_net = delta_it * stats['iss'][t] * (1 - stats['iss'][t])
            delta_ft_net = delta_ft * stats['fss'][t] * (1 - stats['fss'][t])
            delta_ot_net = delta_ot * stats['oss'][t] * (1 - stats['oss'][t])
        return [dwhf, dwxf, dbf,
                dwhi, dwxi, dbi,
                dwha, dwxa, dba,
                dwho, dwxo, dbo,
                dwy, dby]

    # 更新各权重矩阵的偏导数
    def _cal_grad_delta(self, dwh, dwx, db, delta_net, ht_pre, x):
        dwh += delta_net * ht_pre
        dwx += delta_net * x
        db += delta_net
        return dwh, dwx, db

    # 计算梯度, (x,y)为一个样本
    def sgd_step(self, x, y, learning_rate):
        dwhf, dwxf, dbf, \
        dwhi, dwxi, dbi, \
        dwha, dwxa, dba, \
        dwho, dwxo, dbo, \
        dwy, dby = self.bptt(x, y)
        # 更新权重矩阵
        self.whf, self.wxf, self.bf = self._update_wh_wx(learning_rate, self.whf, self.wxf, self.bf, dwhf, dwxf, dbf)
        self.whi, self.wxi, self.bi = \
        self._update_wh_wx(learning_rate, self.whi, self.wxi, self.bi, dwhi,
                           dwxi, dbi)
        self.wha, self.wxa, self.ba = \
        self._update_wh_wx(learning_rate, self.wha, self.wxa, self.ba, dwha,
                           dwxa, dba)
        self.who, self.wxo, self.bo = \
        self._update_wh_wx(learning_rate, self.who, self.wxo, self.bo, dwho,
                           dwxo, dbo)
        self.wy, self.by = self.wy - learning_rate * dwy, self.by - learning_rate * dby

    # 更新权重矩阵
    def _update_wh_wx(self, learning_rate, wh, wx, b, dwh, dwx, db):
        wh -= learning_rate * dwh
        wx -= learning_rate * dwx
        b -= learning_rate * db
        return wh, wx, b

    # 训练 LSTM
    def train(self, X_train, y_train, learning_rate=0.005, n_epoch=5):
        losses = []

        num_examples = 0
        for epoch in range(n_epoch):
            for i in range(len(y_train)):
                self.sgd_step(X_train[i], y_train[i], learning_rate)
        num_examples += 1
        loss = self.loss(X_train, y_train)
        losses.append(loss)
        print('epoch {0}: loss = {1}'.format(epoch + 1, loss))
        if len(losses) > 1 and losses[-1] > losses[-2]:
            learning_rate *= 0.5
        print(f'decrease learning_rate to{learning_rate}')

