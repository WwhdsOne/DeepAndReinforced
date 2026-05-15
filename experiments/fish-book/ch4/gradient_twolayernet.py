import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(logits):
    max_logit = np.max(logits)
    exp_logits = np.exp(logits - max_logit)
    sum_exp_logits = np.sum(exp_logits)
    return exp_logits / sum_exp_logits


def cross_entropy_error(y, t):
    """
    交叉熵误差。
    支持：
    - y, t 为一维向量
    - y, t 为二维批量输入
    - t 为 one-hot 编码或标签索引
    """
    y = np.asarray(y)
    t = np.asarray(t)
    delta = 1e-7

    if y.ndim == 1:
        y = y.reshape(1, -1)
        t = t.reshape(1, -1) if t.ndim != 0 else np.array([t])

    if t.ndim == 1 and y.shape[0] == 1 and t.size == y.shape[1]:
        t = t.reshape(1, -1)

    batch_size = y.shape[0]

    if t.ndim == 1 or (t.ndim == 2 and t.shape == y.shape):
        if t.ndim == 2 and t.shape == y.shape:
            return -1 * np.sum(t * np.log(y + delta)) / batch_size
        return -1 * np.sum(np.log(y[np.arange(batch_size), t] + delta)) / batch_size

    return -1 * np.sum(t * np.log(y + delta)) / batch_size


# 对给定参数张量计算数值梯度
def compute_numerical_gradient(func, params: np.ndarray):
    h = 1e-4
    gradients = np.zeros_like(params)

    it = np.nditer(params, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        original_value = params[idx]
        params[idx] = original_value + h
        loss_plus_h = func(params)

        params[idx] = original_value - h
        loss_minus_h = func(params)

        gradients[idx] = (loss_plus_h - loss_minus_h) / (2 * h)
        params[idx] = original_value
        it.iternext()
    return gradients


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {
            'W1': weight_init_std * np.random.randn(input_size, hidden_size),
            'b1': np.zeros(hidden_size),
            'W2': weight_init_std * np.random.randn(hidden_size, output_size),
            'b2': np.zeros(output_size),
        }

    def predict(self, x):
        a1 = np.dot(x, self.params['W1']) + self.params['b1']
        z1 = sigmoid(a1)
        a2 = np.dot(z1, self.params['W2']) + self.params['b2']
        predictions = softmax(a2)
        return predictions

    def loss(self, x, t):
        predictions = self.predict(x)
        return cross_entropy_error(predictions, t)

    def accuracy(self, x, t):
        predictions = self.predict(x)
        # 将预测结果转为标签索引
        predicted_labels = np.argmax(predictions, axis=1)
        t = np.argmax(t, axis=1)
        accuracy = np.sum(predicted_labels == t) / float(x.shape[0])
        return accuracy

    def compute_gradients(self, x, t):
        loss_for_current_batch = lambda _: self.loss(x, t)
        gradients = {
            'W1': compute_numerical_gradient(loss_for_current_batch, self.params['W1']),
            'b1': compute_numerical_gradient(loss_for_current_batch, self.params['b1']),
            'W2': compute_numerical_gradient(loss_for_current_batch, self.params['W2']),
            'b2': compute_numerical_gradient(loss_for_current_batch, self.params['b2'])
        }
        return gradients


if __name__ == '__main__':
    net = TwoLayerNet(input_size=2, hidden_size=3, output_size=2)
    x = np.random.rand(5, 2)
    t = np.random.rand(5, 2)
    gradients = net.compute_gradients(x, t) # 计算梯度
    for name, gradient in gradients.items():
        print(f"{name} =")
        print(gradient)
        print()
