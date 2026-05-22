"""VGG 系列模型定义。"""
import torch.nn as nn

# VGG 配置: 数字 = Conv2d 输出通道, 'M' = MaxPool2d
VGG_CFGS = {
    "VGG11":  [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13":  [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16":  [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "VGG19":  [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


class VGG(nn.Module):
    """通用 VGG 实现，支持 VGG11 / VGG13 / VGG16 / VGG19。

    Args:
        name: 配置名称，如 "VGG16"
        in_channels: 输入图片通道数（默认 3）
        num_classes: 分类数（默认 200，Tiny ImageNet）
        fc_channels: 全连接层神经元数列表（默认 [4096, 4096]）
    """

    def __init__(self, name="VGG16", in_channels=3, num_classes=200,
                 fc_channels=None):
        super().__init__()
        cfg = VGG_CFGS[name]
        fc_channels = fc_channels or [4096, 4096]
        self.features = self._make_layers(cfg, in_channels)

        # 计算卷积输出尺寸：Tiny ImageNet 64×64 → 5次池化 → 2×2
        # 完整 ImageNet 224×224 → 5次池化 → 7×7
        pool_times = sum(1 for v in cfg if v == "M")
        final_size = 64 // (2 ** pool_times)  # 64 / 32 = 2
        fc_input = 512 * final_size * final_size

        fc_layers = []
        for ch in fc_channels:
            fc_layers.extend([
                nn.Linear(fc_input, ch),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
            ])
            fc_input = ch
        fc_layers.append(nn.Linear(fc_input, num_classes))
        self.classifier = nn.Sequential(*fc_layers)

        # 权重初始化
        self._init_weights()

    def _make_layers(self, cfg, in_channels):
        layers = []
        for v in cfg:
            if v == "M":
                layers.append(nn.MaxPool2d(2, 2))
            else:
                layers.append(nn.Conv2d(in_channels, v, 3, padding=1))
                layers.append(nn.ReLU(inplace=True))
                in_channels = v
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
