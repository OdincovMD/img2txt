"""
Модель отбора важных признаков: предобученный backbone (EfficientNet/ResNet) + голова с N выходами (sigmoid).
"""

from typing import Optional

import torch
import torch.nn as nn

from config.importance_config import NUM_LABELS

try:
    import torchvision.models as tv_models
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False


def get_backbone(name: str = "efficientnet_b0", pretrained: bool = True, in_channels: int = 3):
    """
    name: "efficientnet_b0", "resnet18", "resnet34".
    in_channels: 3 (RGB) или 4 (RGB + маска).
    Возвращает (backbone модуль, размер фичи на выходе).
    """
    if not TORCHVISION_AVAILABLE:
        raise ImportError("torchvision required for EfficientNet/ResNet backbone.")

    if name == "efficientnet_b0":
        weights = "DEFAULT" if pretrained else None
        m = tv_models.efficientnet_b0(weights=weights)
        if in_channels != 3:
            old = m.features[0][0]
            m.features[0][0] = nn.Conv2d(
                in_channels, old.out_channels, kernel_size=old.kernel_size,
                stride=old.stride, padding=old.padding, bias=old.bias is not None,
            )
            if pretrained:
                with torch.no_grad():
                    m.features[0][0].weight[:, :3] = old.weight
                    if in_channels == 4:
                        m.features[0][0].weight[:, 3] = m.features[0][0].weight[:, 0].mean()
        feat_dim = m.classifier[1].in_features
        m.classifier = nn.Identity()
        return m, feat_dim

    if name == "resnet18":
        weights = "DEFAULT" if pretrained else None
        m = tv_models.resnet18(weights=weights)
        if in_channels != 3:
            old = m.conv1
            m.conv1 = nn.Conv2d(
                in_channels, old.out_channels, kernel_size=old.kernel_size,
                stride=old.stride, padding=old.padding, bias=old.bias is not None,
            )
            if pretrained:
                with torch.no_grad():
                    m.conv1.weight[:, :3] = old.weight
                    if in_channels == 4:
                        m.conv1.weight[:, 3] = m.conv1.weight[:, 0].mean()
        feat_dim = m.fc.in_features
        m.fc = nn.Identity()
        return m, feat_dim

    if name == "resnet34":
        weights = "DEFAULT" if pretrained else None
        m = tv_models.resnet34(weights=weights)
        if in_channels != 3:
            old = m.conv1
            m.conv1 = nn.Conv2d(
                in_channels, old.out_channels, kernel_size=old.kernel_size,
                stride=old.stride, padding=old.padding, bias=old.bias is not None,
            )
            if pretrained:
                with torch.no_grad():
                    m.conv1.weight[:, :3] = old.weight
                    if in_channels == 4:
                        m.conv1.weight[:, 3] = m.conv1.weight[:, 0].mean()
        feat_dim = m.fc.in_features
        m.fc = nn.Identity()
        return m, feat_dim

    raise ValueError(f"Unknown backbone: {name}. Use efficientnet_b0, resnet18, resnet34.")


class ImportanceModel(nn.Module):
    """
    Backbone (CNN) + линейная голова с num_labels выходами.
    Логиты для BCEWithLogitsLoss; при инференсе после sigmoid берём топ-10.
    """

    def __init__(
        self,
        backbone_name: str = "efficientnet_b0",
        num_labels: int = NUM_LABELS,
        pretrained: bool = True,
        in_channels: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.backbone, feat_dim = get_backbone(backbone_name, pretrained=pretrained, in_channels=in_channels)
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feat_dim, num_labels),
        )
        self.num_labels = num_labels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)
