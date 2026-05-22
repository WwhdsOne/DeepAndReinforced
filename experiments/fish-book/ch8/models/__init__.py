"""模型统一导出，方便 train.py 按名称选取模型。"""
from .vgg import VGG

MODEL_REGISTRY = {
    "VGG11": lambda **kw: VGG("VGG11", **kw),
    "VGG13": lambda **kw: VGG("VGG13", **kw),
    "VGG16": lambda **kw: VGG("VGG16", **kw),
    "VGG19": lambda **kw: VGG("VGG19", **kw),
}


def get_model(name, **kwargs):
    """按名称获取模型实例。

    后续加 ResNet 时只需在 MODEL_REGISTRY 中注册即可：
        from .resnet import ResNet18
        MODEL_REGISTRY["ResNet18"] = lambda **kw: ResNet18(**kw)
    """
    if name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY)
        raise KeyError(f"未知模型 {name!r}，可选: {available}")
    return MODEL_REGISTRY[name](**kwargs)
