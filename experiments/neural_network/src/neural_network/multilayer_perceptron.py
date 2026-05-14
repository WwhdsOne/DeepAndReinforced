"""基于 NumPy 的多层感知机实现。"""

from __future__ import annotations

import numpy as np
from tqdm.auto import tqdm


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Sigmoid 激活函数。"""
    z = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_gradient(z: np.ndarray) -> np.ndarray:
    """Sigmoid 函数的梯度。"""
    s = sigmoid(z)
    return s * (1.0 - s)


def relu(z: np.ndarray) -> np.ndarray:
    """ReLU 激活函数。"""
    return np.maximum(0.0, z)


def relu_gradient(z: np.ndarray) -> np.ndarray:
    """ReLU 的梯度。"""
    return (z > 0).astype(z.dtype)


def softmax(z: np.ndarray) -> np.ndarray:
    """按行计算 softmax，保证数值稳定。"""
    shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def prepare_for_training(
        data: np.ndarray,
        normalize_data: bool = False,
) -> np.ndarray:
    """训练前预处理：可选归一化，并添加偏置列。"""
    data_processed = np.asarray(data, dtype=float).copy()
    if data_processed.ndim != 2:
        raise ValueError("输入数据必须是二维数组")

    if normalize_data:
        data_min = data_processed.min(axis=0, keepdims=True)
        data_max = data_processed.max(axis=0, keepdims=True)
        scale = np.maximum(data_max - data_min, 1e-8)
        data_processed = (data_processed - data_min) / scale

    bias = np.ones((data_processed.shape[0], 1), dtype=data_processed.dtype)
    return np.hstack((bias, data_processed))


class MultilayerPerceptron:
    """多层感知机分类器。"""

    def __init__(
            self,
            data: np.ndarray,
            labels: np.ndarray,
            layers: tuple[int, ...],
            normalize_data: bool = False,
    ) -> None:
        if len(layers) < 2:
            raise ValueError("layers 至少需要包含输入层和输出层")

        self.data = prepare_for_training(data, normalize_data=normalize_data)
        self.labels = np.asarray(labels).reshape(-1)
        self.layers = layers
        self.normalize_data = normalize_data
        self.weights = self.init_weights(self.layers)

        if self.data.shape[0] != self.labels.shape[0]:
            raise ValueError("data 和 labels 的样本数必须一致")
        if self.data.shape[1] != layers[0] + 1:
            raise ValueError("data 的特征维度与 layers[0] 不一致")

    def train(
        self,
        max_iterations: int = 1000,
        alpha: float = 0.01,
        batch_size: int | None = None,
        optimizer: str = "adam",
        show_progress: bool = False,
        epoch_callback: callable | None = None,
    ) -> list[float]:
        """使用 mini-batch 训练模型，返回损失历史。"""
        unrolled_weights = self.unroll_weights(self.weights)
        optimized_weights, cost_history = self.gradient_descent(
            self.data,
            self.labels,
            unrolled_weights,
            self.layers,
            max_iterations,
            alpha,
            batch_size=batch_size,
            optimizer=optimizer,
            show_progress=show_progress,
            return_cost_history=True,
            epoch_callback=epoch_callback,
        )
        self.weights = self.roll_weights(optimized_weights, self.layers)
        return cost_history

    def predict(self, data: np.ndarray) -> np.ndarray:
        """对输入数据进行预测，返回类别索引。"""
        prepared_data = prepare_for_training(
            data,
            normalize_data=self.normalize_data,
        )
        probabilities = self.feedforward_propagation(
            prepared_data,
            self.weights,
            self.layers,
        )
        return np.argmax(probabilities, axis=1)

    @staticmethod
    def init_weights(layers: tuple[int, ...]) -> dict[int, np.ndarray]:
        """初始化各层权重矩阵，使用较小的随机值。"""
        weights: dict[int, np.ndarray] = {}
        for layer_index in range(len(layers) - 1):
            in_count = layers[layer_index]
            out_count = layers[layer_index + 1]
            epsilon = np.sqrt(6.0 / (in_count + out_count))
            # 均匀分布
            weights[layer_index] = np.random.uniform(
                -epsilon,
                epsilon,
                size=(out_count, in_count + 1),
            )
        return weights

    @staticmethod
    def unroll_weights(weights: dict[int, np.ndarray]) -> np.ndarray:
        """将权重矩阵字典展开为一维向量。"""
        return np.concatenate([weights[index].ravel() for index in sorted(weights)])

    @staticmethod
    def roll_weights(
            unrolled: np.ndarray,
            layers: tuple[int, ...],
    ) -> dict[int, np.ndarray]:
        """将展开后的向量恢复为各层权重矩阵字典。"""
        weights: dict[int, np.ndarray] = {}
        start = 0
        for layer_index in range(len(layers) - 1):
            in_count = layers[layer_index]
            out_count = layers[layer_index + 1]
            num_elements = out_count * (in_count + 1)
            end = start + num_elements
            weights[layer_index] = unrolled[start:end].reshape(out_count, in_count + 1)
            start = end
        return weights

    @staticmethod
    def gradient_descent(
            data: np.ndarray,
            labels: np.ndarray,
            unrolled_weights: np.ndarray,
            layers: tuple[int, ...],
            max_iterations: int,
            alpha: float,
            batch_size: int | None = None,
            optimizer: str = "adam",
            show_progress: bool = False,
            return_cost_history: bool = False,
            epoch_callback: callable | None = None,
    ) -> np.ndarray | tuple[np.ndarray, list[float]]:
        """使用反向传播和梯度下降优化权重。"""
        optimized_weights = np.asarray(unrolled_weights, dtype=float).copy()
        cost_history: list[float] = []
        num_examples = data.shape[0]
        effective_batch_size = num_examples if batch_size is None or batch_size <= 0 else min(batch_size, num_examples)

        optimizer_name = optimizer.lower()
        if optimizer_name not in {"adam", "sgd"}:
            raise ValueError("optimizer 仅支持 adam 或 sgd")

        adam_m = np.zeros_like(optimized_weights)
        adam_v = np.zeros_like(optimized_weights)
        adam_t = 0
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8

        iterator = tqdm(range(max_iterations), desc="训练进度", unit="轮", leave=True) if show_progress else range(max_iterations)
        for iteration in iterator:
            indices = np.random.permutation(num_examples)
            for start in range(0, num_examples, effective_batch_size):
                batch_indices = indices[start:start + effective_batch_size]
                batch_data = data[batch_indices]
                batch_labels = labels[batch_indices]
                weights = MultilayerPerceptron.roll_weights(optimized_weights, layers)
                gradients = MultilayerPerceptron.back_propagation(batch_data, batch_labels, weights, layers)
                grad_vector = MultilayerPerceptron.unroll_weights(gradients)

                if optimizer_name == "adam":
                    adam_t += 1
                    adam_m = beta1 * adam_m + (1.0 - beta1) * grad_vector
                    adam_v = beta2 * adam_v + (1.0 - beta2) * (grad_vector ** 2)
                    m_hat = adam_m / (1.0 - beta1 ** adam_t)
                    v_hat = adam_v / (1.0 - beta2 ** adam_t)
                    optimized_weights -= alpha * m_hat / (np.sqrt(v_hat) + epsilon)
                else:
                    optimized_weights -= alpha * grad_vector

            weights = MultilayerPerceptron.roll_weights(optimized_weights, layers)
            cost_history.append(MultilayerPerceptron.compute_cost(data, labels, weights, layers))
            if epoch_callback is not None:
                epoch_callback(iteration, MultilayerPerceptron.roll_weights(optimized_weights, layers))

        if return_cost_history:
            return optimized_weights, cost_history
        return optimized_weights

    @staticmethod
    def back_propagation(
            data: np.ndarray,
            labels: np.ndarray,
            weights: dict[int, np.ndarray],
            layers: tuple[int, ...],
    ) -> dict[int, np.ndarray]:
        """反向传播，计算每层权重梯度。"""
        num_examples = data.shape[0]
        num_layers = len(layers)
        labels_flat = np.asarray(labels).reshape(-1)

        activations: dict[int, np.ndarray] = {0: data}
        z_values: dict[int, np.ndarray] = {}

        activation = data
        for layer_index in range(num_layers - 1):
            z = activation @ weights[layer_index].T
            z_values[layer_index] = z
            activation = softmax(z) if layer_index == num_layers - 2 else relu(z)
            if layer_index < num_layers - 2:
                # hstack = horizontal stack意思：水平方向拼接数组 / 张量（左右拼）。
                activation = np.hstack(
                    (np.ones((num_examples, 1), dtype=activation.dtype), activation)
                )
            activations[layer_index + 1] = activation

        num_classes = layers[-1]
        y_one_hot = np.zeros((num_examples, num_classes), dtype=float)
        y_one_hot[np.arange(num_examples), labels_flat] = 1.0

        deltas: dict[int, np.ndarray] = {
            num_layers - 1: activations[num_layers - 1] - y_one_hot
        }

        for layer_index in range(num_layers - 2, 0, -1):
            weight_next = weights[layer_index][:, 1:]
            delta_next = deltas[layer_index + 1]
            z = z_values[layer_index - 1]
            deltas[layer_index] = (delta_next @ weight_next) * relu_gradient(z)

        gradients: dict[int, np.ndarray] = {}
        for layer_index in range(num_layers - 1):
            gradients[layer_index] = (
                                             deltas[layer_index + 1].T @ activations[layer_index]
                                     ) / num_examples

        return gradients

    @staticmethod
    def compute_cost(
            data: np.ndarray,
            labels: np.ndarray,
            weights: dict[int, np.ndarray],
            layers: tuple[int, ...],
    ) -> float:
        """计算交叉熵损失。"""
        labels_flat = np.asarray(labels).reshape(-1)
        predictions = MultilayerPerceptron.feedforward_propagation(data, weights, layers)
        num_examples = data.shape[0]
        num_classes = layers[-1]

        y_one_hot = np.zeros((num_examples, num_classes), dtype=float)
        y_one_hot[np.arange(num_examples), labels_flat] = 1.0

        eps = 1e-12
        predictions = np.clip(predictions, eps, 1.0 - eps)
        cost = -1 * np.sum(y_one_hot * np.log(predictions)) / num_examples
        return float(cost)

    @staticmethod
    def feedforward_propagation(
            data: np.ndarray,
            weights: dict[int, np.ndarray],
            layers: tuple[int, ...],
    ) -> np.ndarray:
        """前向传播，返回输出层概率。"""
        num_layers = len(layers)
        activation = data

        for layer_index in range(num_layers - 1):
            z = activation @ weights[layer_index].T
            activation = softmax(z) if layer_index == num_layers - 2 else relu(z)
            if layer_index < num_layers - 2:
                activation = np.hstack(
                    (np.ones((activation.shape[0], 1), dtype=activation.dtype), activation)
                )

        return activation
