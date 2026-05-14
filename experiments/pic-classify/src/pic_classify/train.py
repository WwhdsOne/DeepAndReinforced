import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from tqdm import tqdm

from pic_classify.data import CLASS_NAMES, build_eval_transform, build_train_transform
from pic_classify.model import CIFAR10Classifier

EXPERIMENT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = EXPERIMENT_DIR / "data"
DEFAULT_MODEL_PATH = EXPERIMENT_DIR / "artifacts" / "cifar10_cnn.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CIFAR-10 classifier.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_MODEL_PATH,
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--optimizer",
        choices=("adam", "sgd"),
        default="adam",
        help="训练时使用的优化器。",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def build_dataloaders(
    data_dir: Path, batch_size: int, num_workers: int
) -> tuple[DataLoader, DataLoader]:
    train_dataset = CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=build_train_transform(),
    )
    test_dataset = CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=build_eval_transform(),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, test_loader


def train_one_epoch(
    model: CIFAR10Classifier,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0

    progress_bar = tqdm(loader, desc="训练中", leave=False)
    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            samples=total_samples,
        )

    return total_loss / total_samples


@torch.no_grad()
def evaluate(
    model: CIFAR10Classifier,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return correct / total


def save_model(model: CIFAR10Classifier, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": CLASS_NAMES,
        },
        output_path,
    )


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    model = CIFAR10Classifier().to(device)
    loss_fn = nn.CrossEntropyLoss()
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(1, args.epochs + 1):
        print(f"epoch {epoch}/{args.epochs}", flush=True)
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        accuracy = evaluate(model, test_loader, device)
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} test_accuracy={accuracy:.4f}",
            flush=True,
        )

    save_model(model, args.output)
    print(f"saved model to {args.output}", flush=True)


if __name__ == "__main__":
    main()
