import argparse
from pathlib import Path

import torch
from PIL import Image

from pic_classify.data import CLASS_NAMES, build_eval_transform
from pic_classify.model import CIFAR10Classifier

EXPERIMENT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = EXPERIMENT_DIR / "artifacts" / "cifar10_cnn.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict a class for one image.")
    parser.add_argument("image", type=Path)
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
    )
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def load_model(
    model_path: Path, device: torch.device
) -> tuple[CIFAR10Classifier, tuple[str, ...]]:
    checkpoint = torch.load(model_path, map_location=device)
    class_names = tuple(checkpoint.get("class_names", CLASS_NAMES))

    model = CIFAR10Classifier().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, class_names


@torch.no_grad()
def predict(
    model: CIFAR10Classifier,
    image_path: Path,
    device: torch.device,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    image = Image.open(image_path).convert("RGB")
    tensor = build_eval_transform()(image).unsqueeze(0).to(device)
    logits = model(tensor)
    probabilities = torch.softmax(logits, dim=1)
    return torch.topk(probabilities, k=top_k, dim=1)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names = load_model(args.model, device)
    scores, indices = predict(model, args.image, device, args.top_k)

    for score, index in zip(scores[0], indices[0]):
        print(f"{class_names[index]}: {score.item():.4f}")


if __name__ == "__main__":
    main()
