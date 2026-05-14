"""MNIST 手写数字识别命令行工具。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from neural_network.mnist_dataset import (
    load_image_as_mnist_features,
    load_mnist_dataset,
)
from neural_network.multilayer_perceptron import MultilayerPerceptron


_PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent  # experiments/neural_network/
DEFAULT_DATA_DIR = _PACKAGE_ROOT / "data"
DEFAULT_OUTPUT_DIR = _PACKAGE_ROOT / "artifacts"


def train_main() -> None:
    """训练入口。"""
    import argparse

    args = argparse.ArgumentParser(
        description="使用多层感知机训练 MNIST 手写数字识别模型"
    )
    args.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR / "mnist_mlp.npz")
    args.add_argument("--epochs", type=int, default=50)
    args.add_argument("--learning-rate", type=float, default=0.001)
    args.add_argument("--hidden-size", type=int, default=256)
    args.add_argument("--batch-size", type=int, default=64)
    args.add_argument("--optimizer", choices=("adam", "sgd"), default="adam")
    args.add_argument("--limit-train", type=int, default=60000)
    args.add_argument("--limit-test", type=int, default=2000)
    parsed = args.parse_args()

    train_command(parsed)


def predict_main() -> None:
    """预测入口。"""
    import argparse

    args = argparse.ArgumentParser(description="对单张 MNIST 图片进行识别")
    args.add_argument("image", type=Path)
    args.add_argument("--model", type=Path, default=DEFAULT_OUTPUT_DIR / "mnist_mlp.npz")
    parsed = args.parse_args()

    predict_command(parsed)


def evaluate_main() -> None:
    """评估入口。"""
    import argparse

    args = argparse.ArgumentParser(description="评估已训练的 MNIST 模型")
    args.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args.add_argument("--model", type=Path, default=DEFAULT_OUTPUT_DIR / "mnist_mlp.npz")
    args.add_argument("--limit-test", type=int, default=2000)
    parsed = args.parse_args()

    evaluate_command(parsed)


def train_command(args) -> None:
    """训练 MNIST 模型并保存权重。"""
    x_train, y_train, x_test, y_test = load_mnist_dataset(args.data_dir)
    x_train, y_train = _limit_dataset(x_train, y_train, args.limit_train)
    x_test, y_test = _limit_dataset(x_test, y_test, args.limit_test)

    layers = (784, args.hidden_size, 10)
    model = MultilayerPerceptron(
        x_train,
        y_train,
        layers=layers,
        normalize_data=False,
    )

    val_accuracy_history: list[float] = []

    def record_accuracy(iteration: int, weights: dict) -> None:
        saved = model.weights
        model.weights = weights
        val_accuracy_history.append(_accuracy(model.predict(x_test), y_test))
        model.weights = saved

    history = model.train(
        max_iterations=args.epochs,
        alpha=args.learning_rate,
        batch_size=args.batch_size,
        optimizer=args.optimizer,
        show_progress=True,
        epoch_callback=record_accuracy,
    )
    accuracy = val_accuracy_history[-1] if val_accuracy_history else _accuracy(model.predict(x_test), y_test)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    _save_model(args.output, model)

    history_path = args.output.with_suffix(".history.json")
    history_path.write_text(json.dumps({
        "loss": history,
        "accuracy": val_accuracy_history,
        "hyperparameters": {
            "hidden_size": args.hidden_size,
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "optimizer": args.optimizer,
            "limit_train": args.limit_train,
            "limit_test": args.limit_test,
            "layers": list(layers),
        },
        "final_accuracy": accuracy,
        "final_loss": history[-1],
    }, indent=2, ensure_ascii=False))

    print(f"训练完成，最后一次损失：{history[-1]:.6f}")
    print(f"测试集准确率：{accuracy:.4f}")
    print(f"模型已保存到：{args.output}")
    print(f"训练历史已保存到：{history_path}")


def predict_command(args) -> None:
    """对单张图片进行识别。"""
    model = _load_model(args.model)
    features = load_image_as_mnist_features(args.image)
    prediction = model.predict(features)[0]
    print(int(prediction))


def evaluate_command(args) -> None:
    """评估保存好的模型。"""
    _, _, x_test, y_test = load_mnist_dataset(args.data_dir)
    x_test, y_test = _limit_dataset(x_test, y_test, args.limit_test)
    model = _load_model(args.model)
    accuracy = _accuracy(model.predict(x_test), y_test)
    print(f"测试集准确率：{accuracy:.4f}")


def _limit_dataset(
    features: np.ndarray,
    labels: np.ndarray,
    limit: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """按数量截断数据集。"""
    if limit is None or limit <= 0:
        return features, labels
    return features[:limit], labels[:limit]


def _accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    """计算准确率。"""
    return float(np.mean(predictions == labels))


def _save_model(path: Path, model: MultilayerPerceptron) -> None:
    """保存模型参数。"""
    np.savez(
        path,
        layers=np.array(model.layers, dtype=np.int64),
        weights=np.array([model.weights[index] for index in sorted(model.weights)], dtype=object),
    )


def _load_model(path: Path) -> MultilayerPerceptron:
    """从磁盘恢复模型。"""
    if not path.exists():
        raise FileNotFoundError(f"模型文件不存在：{path}")

    with np.load(path, allow_pickle=True) as payload:
        layers = tuple(int(value) for value in payload["layers"].tolist())
        weights_array = payload["weights"].tolist()

    model = MultilayerPerceptron(
        np.zeros((1, layers[0])),
        np.zeros((1,), dtype=np.int64),
        layers=layers,
        normalize_data=False,
    )
    model.weights = {index: np.asarray(weight) for index, weight in enumerate(weights_array)}
    return model
