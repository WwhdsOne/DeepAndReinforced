import torch

from pic_classify.model import CIFAR10Classifier, NUM_CLASSES


def test_model_returns_logits_for_ten_classes():
    model = CIFAR10Classifier()
    batch = torch.randn(4, 3, 32, 32)

    logits = model(batch)

    assert logits.shape == (4, NUM_CLASSES)
