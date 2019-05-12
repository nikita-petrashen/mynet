import numpy as np


def compute_accuracy(y_hat, y):
    y_hat_classes = np.argmax(y_hat, axis=1)
    y_classes = np.argmax(y, axis=1)
    return np.sum(np.where(y_hat_classes == y_classes, 1, 0)) / y.shape[0]


def get_batched_indices(N, batch_size=100):
    N_batches = N // batch_size
    indices = np.arange(N)
    np.random.shuffle(indices)
    batches = indices[: N_batches * batch_size]
    batches = batches.reshape(N_batches, batch_size)
    batches = list(batches)
    if N != N_batches * batch_size:
        batches.append(list(indices[N_batches * batch_size:]))
    return batches


class ReLu:
    def __init__(self):
        self.x = None
        self.g = None

    def forward(self, x, mode=None):
        self.x = x

        return np.maximum(np.zeros(self.x.shape), self.x)

    def backward(self, g):
        self.g = g * (self.x > 0).astype(np.float32)

        return self.g

    def print_params(self):
        pass


def Softmax_Loss(x, y):
    exp_x = np.exp(x)
    y_hat = exp_x / np.sum(exp_x, axis=0)

    loss = -np.log(y_hat[y == 1]).sum() / x.shape[1]
    grad = (y_hat - y) / x.shape[1]


    return y_hat, loss, grad


def Softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=0)


class BatchNorm:
    def __init__(self, n):
        self.x = None
        self.x_c = None
        self.disp = None
        self.mean = None
        self.gamma = np.ones(n).reshape(-1, 1)
        self.beta = np.zeros(n).reshape(-1, 1)
        self.g = None
        self.g_out = None
        self.avg_mean = 0
        self.avg_disp = 0
        self.count = 0.0

    def forward(self, x, mode=None):
        if mode == 'train':
            self.x = x
            N = x.shape[1]
            self.mean = np.sum(x, axis=1).reshape(-1, 1) / N
            self.x_c = x - self.mean
            self.disp = np.sum(np.power(self.x_c, 2), axis=1).reshape(-1, 1) / N + 0.00000001
            self.count += 1
            self.avg_mean = (self.count - 1) / self.count * self.avg_mean + self.mean / self.count
            self.avg_disp = (self.count - 1) / self.count * self.avg_disp + self.disp / self.count

            return self.x_c * np.power(self.disp, -1. / 2) * self.gamma + self.beta
        if mode == 'test':
            self.x = x
            return (self.x - self.avg_mean) * np.power(self.avg_disp, -1. / 2) * self.gamma + self.beta

    def backward(self, grad):
        N = grad.shape[1]
        self.g = grad
        self.g_out = np.power(self.disp, -1. / 2) * self.gamma * (N * grad -
                                                                  np.sum(grad, axis=1).reshape(-1, 1) -
                                                                  np.power(self.disp, -1) *
                                                                  self.x_c * np.sum(grad * self.x_c, axis=1).reshape(-1,
                                                                                                                     1)) / N


        return self.g_out

    def print_params(self):
        print('BatchNorm layer \ngamma: \n', self.gamma, '\nbeta: \n', self.beta)


class Linear:
    def __init__(self, m, n, dropout=0):
        self.W = np.random.randn(m, n)
        self.b = np.random.uniform(low=0, high=2, size=m)
        self.x = None
        self.g = None
        self.dropout = dropout

    def forward(self, x, mode=None):
        self.x = x
        return np.matmul(self.W, self.x) + self.b.reshape(self.b.shape[0], 1)

    def backward(self, g):
        self.g = g
        return np.matmul(self.W.T, self.g)

    def print_params(self):
        print('Linear layer \nW: \n', self.W, '\nb:\n', self.b)


class MyNet:
    def __init__(self, layers, reg='l1', l=0.01):
        self.layers = layers
        self.linear_layers = [el for el in self.layers if type(el) == Linear]
        self.batch_norm_layers = [el for el in self.layers if type(el) == BatchNorm]
        self.pred = None
        self.loss = None
        self.grad = None
        self.lr = 0.0003
        self.reg = reg
        self.l = l
        self.m_w = []
        self.m_b = []
        self.m_gamma = []
        self.m_beta = []
        self.v_w = []
        self.v_b = []
        self.v_gamma = []
        self.v_beta = []
        self.t = 0
        self.epochs = 0
        self.loss_history = []
        for layer in self.linear_layers:
            self.m_w.append(np.zeros_like(layer.W))
            self.v_w.append(np.zeros_like(layer.W))
            self.m_b.append(np.zeros_like(layer.b))
            self.v_b.append(np.zeros_like(layer.b))
        for layer in self.batch_norm_layers:
            self.m_gamma.append(np.zeros_like(layer.gamma))
            self.v_gamma.append(np.zeros_like(layer.gamma))
            self.m_beta.append(np.zeros_like(layer.beta))
            self.v_beta.append(np.zeros_like(layer.beta))

    def forward(self, x, y, mode=None):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if y.ndim == 1:
            y = y.reshape(1, -1)
        h = x.T
        for layer in self.layers:
            h = layer.forward(h, mode)
        self.pred, self.loss, self.grad = Softmax_Loss(h, y.T)
        return self.pred, self.loss

    def predict(self, x, mode=None):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        h = x.T
        for layer in self.layers:
            h = layer.forward(h, mode)
        self.pred = Softmax(h)
        return self.pred

    def backward(self):
        cur_grad = self.grad
        for layer in reversed(self.layers):
            cur_grad = layer.backward(cur_grad)

    def print_net(self):
        for layer in self.layers:
            print(type(layer), layer.x, layer.g)

    def make_adam_step(self, beta1, beta2, epsilon):
        for i in range(len(self.linear_layers)):
            layer = self.linear_layers[i]
            if self.reg == 'none':
                grad = layer.g @ layer.x.T
            elif self.reg == 'l1':
                grad = layer.g @ layer.x.T + self.l * np.where(layer.W > 0, 1, -1)
            elif self.reg == 'l2':
                grad = layer.g @ layer.x.T + self.l * 2 * layer.W
            self.m_w[i] = beta1 * self.m_w[i] + (1 - beta1) * grad
            self.v_w[i] = beta2 * self.v_w[i] + (1 - beta2) * np.power(grad, 2)
            m_hat = self.m_w[i] / (1 - np.power(beta1, self.t))
            v_hat = self.v_w[i] / (1 - np.power(beta2, self.t))
            layer.W -= self.lr * m_hat / (np.sqrt(v_hat) + epsilon)
            grad = layer.g.sum(axis=1) / layer.x.shape[1]
            self.m_b[i] = beta1 * self.m_b[i] + (1 - beta1) * grad
            self.v_b[i] = beta2 * self.v_b[i] + (1 - beta2) * np.power(grad, 2)
            m_hat = self.m_b[i] / (1 - np.power(beta1, self.t))
            v_hat = self.v_b[i] / (1 - np.power(beta2, self.t))
            layer.b -= self.lr * m_hat / (np.sqrt(v_hat) + epsilon)
        for i in range(len(self.batch_norm_layers)):
            layer = self.batch_norm_layers[i]
            grad = np.sum(layer.g * layer.x_c * np.power(layer.disp, -1. / 2), axis=1).reshape(-1, 1)
            self.m_gamma[i] = beta1 * self.m_gamma[i] + (1 - beta1) * grad
            self.v_gamma[i] = beta2 * self.v_gamma[i] + (1 - beta2) * np.power(grad, 2)
            m_hat = self.m_gamma[i] / (1 - np.power(beta1, self.t))
            v_hat = self.v_gamma[i] / (1 - np.power(beta2, self.t))
            layer.gamma -= self.lr * m_hat / (np.sqrt(v_hat) + epsilon)
            grad = np.sum(layer.g, axis=1).reshape(-1, 1)
            self.m_beta[i] = beta1 * self.m_beta[i] + (1 - beta1) * grad
            self.v_beta[i] = beta2 * self.v_beta[i] + (1 - beta2) * np.power(grad, 2)
            self.m_hat = self.m_beta[i] / (1 - np.power(beta1, self.t))
            self.v_hat = self.v_beta[i] / (1 - np.power(beta2, self.t))
            layer.beta -= self.lr * m_hat / (np.sqrt(v_hat) + epsilon)

    def fit(self, x, y, epochs=5, bs=100):
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 0.0000000001
        for epoch in range(epochs):
            batches = get_batched_indices(x.shape[0], batch_size=bs)
            loss = 0
            for batch in batches:
                self.t += 1
                loss += self.forward(x[batch], y[batch])[1]
                self.backward()
                self.make_adam_step(beta1, beta2, epsilon)
            self.epochs += 1
            self.loss_history.append(loss)
            if self.epochs % 20 == 0 and floor(self.epochs / 20) < 10:
                self.lr = self.lr * 0.5 ** floor(self.epochs / 20)
            print('epoch = ', epoch, 'loss = ', loss / len(batches))
        return self
    def check_w1_grad(self, x, y):
        grad = np.zeros_like(self.linear_layers[0].W)
        for i in range(self.linear_layers[0].W.shape[0]):
            for j in range(self.linear_layers[0].W.shape[1]):
                _, f0 = net.forward(x, y)
                self.linear_layers[0].W[i][j] += 0.00000001
                _, f1 = net.forward(x, y)
                grad[i][j] = (f1 - f0) / 0.00000001
                self.linear_layers[0].W[i][j] -= 0.00000001
        return grad

    def check_w2_grad(self, x, y):
        grad = np.zeros_like(self.linear_layers[1].W)
        for i in range(self.linear_layers[1].W.shape[0]):
            for j in range(self.linear_layers[1].W.shape[1]):
                _, f0 = net.forward(x, y)
                self.linear_layers[1].W[i][j] += 0.00000001
                _, f1 = net.forward(x, y)
                grad[i][j] = (f1 - f0) / 0.00000001
                self.linear_layers[1].W[i][j] -= 0.00000001
        return grad

    def check_batch_grad(self, x, y):
        layer = self.batch_norm_layers[0]
        beta_grad = np.zeros_like(layer.beta)
        for i in range(layer.beta.shape[0]):
            _, f0 = net.forward(x, y)
            layer.beta[i] += 0.00000001
            _, f1 = net.forward(x, y)
            beta_grad[i] = (f1 - f0) / 0.00000001
            layer.beta[i] -= 0.00000001
        gamma_grad = np.zeros_like(layer.gamma)
        for i in range(layer.gamma.shape[0]):
            _, f0 = net.forward(x, y)
            layer.gamma[i] += 0.00000001
            _, f1 = net.forward(x, y)
            gamma_grad[i] = (f1 - f0) / 0.00000001
            layer.gamma[i] -= 0.00000001
        print('beta_grad = \n', beta_grad, '\ngamma_grad = \n', gamma_grad)

    def print_params(self):
        for layer in self.layers:
            layer.print_params()