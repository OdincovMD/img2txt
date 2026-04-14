"""
Модель отбора важных признаков: предобученный backbone (EfficientNet/ResNet) +
ветка табличных признаков + голова с N выходами (sigmoid).

Вход:
  x        — тензор изображения (B, C, H, W), C=3 или 4
  features — нормализованный вектор числовых признаков (B, FEAT_DIM)
"""

from typing import Optional

import torch
import torch.nn as nn

from config.importance_config import NUM_LABELS, FEAT_DIM

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
    Backbone (CNN) + ветка табличных признаков + голова с num_labels выходами.

    forward(x, features) → логиты (B, num_labels).
    При инференсе после sigmoid берём топ-k.
    """

    def __init__(
        self,
        backbone_name: str = "efficientnet_b0",
        num_labels: int = NUM_LABELS,
        pretrained: bool = True,
        in_channels: int = 3,
        dropout: float = 0.2,
        feat_input_dim: int = FEAT_DIM,
    ):
        super().__init__()
        self.backbone, feat_dim = get_backbone(backbone_name, pretrained=pretrained, in_channels=in_channels)
        # Ветка для табличных признаков
        self.feature_branch = nn.Sequential(
            nn.Linear(feat_input_dim, 64),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feat_dim + 64, num_labels),
        )
        self.num_labels = num_labels
        self.feat_input_dim = feat_input_dim

    def forward(self, x: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        img_feat = self.backbone(x)
        tab_feat = self.feature_branch(features)
        combined = torch.cat([img_feat, tab_feat], dim=1)
        return self.head(combined)
