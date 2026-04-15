"""
Модель v2: CNN backbone + табличная ветка + голова (24 выхода, sigmoid).

Поддерживаемые backbone: efficientnet_b0, resnet18, resnet34.
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models

from training_v2.config import NUM_LABELS


def get_backbone(name: str, pretrained: bool = True, in_channels: int = 3):
    """
    Возвращает (backbone_module, feature_dim).
    Если in_channels != 3, первый conv-слой заменяется с сохранением
    претренированных весов для RGB-каналов.
    """
    builders = {
        "efficientnet_b0": (tv_models.efficientnet_b0, "efficientnet"),
        "resnet18":        (tv_models.resnet18,        "resnet"),
        "resnet34":        (tv_models.resnet34,        "resnet"),
    }
    if name not in builders:
        raise ValueError(f"Unknown backbone: {name}. Choose: {list(builders)}")

    builder_fn, family = builders[name]
    weights = "DEFAULT" if pretrained else None
    m = builder_fn(weights=weights)

    if family == "efficientnet":
        if in_channels != 3:
            old = m.features[0][0]
            new_conv = nn.Conv2d(
                in_channels, old.out_channels,
                kernel_size=old.kernel_size, stride=old.stride,
                padding=old.padding, bias=(old.bias is not None),
            )
            if pretrained:
                with torch.no_grad():
                    new_conv.weight[:, :3] = old.weight
                    if in_channels == 4:
                        new_conv.weight[:, 3] = old.weight[:, 0]
            m.features[0][0] = new_conv
        feat_dim = m.classifier[1].in_features
        m.classifier = nn.Identity()
    else:  # resnet
        if in_channels != 3:
            old = m.conv1
            new_conv = nn.Conv2d(
                in_channels, old.out_channels,
                kernel_size=old.kernel_size, stride=old.stride,
                padding=old.padding, bias=(old.bias is not None),
            )
            if pretrained:
                with torch.no_grad():
                    new_conv.weight[:, :3] = old.weight
                    if in_channels == 4:
                        new_conv.weight[:, 3] = old.weight[:, 0]
            m.conv1 = new_conv
        feat_dim = m.fc.in_features
        m.fc = nn.Identity()

    return m, feat_dim


class ImportanceModelV2(nn.Module):
    """
    Backbone (CNN)  ──┐
                      ├── concat ── head ── logits (B, NUM_LABELS)
    Feature branch ───┘

    forward(image, features) → logits  (без sigmoid, для BCEWithLogitsLoss)
    """

    def __init__(
        self,
        backbone_name: str = "efficientnet_b0",
        num_labels: int = NUM_LABELS,
        pretrained: bool = True,
        in_channels: int = 3,
        dropout: float = 0.3,
        feat_input_dim: int = 0,
    ):
        super().__init__()
        self.backbone, img_dim = get_backbone(
            backbone_name, pretrained=pretrained, in_channels=in_channels
        )
        self.num_labels = num_labels
        self.feat_input_dim = feat_input_dim

        # Табличная ветка (если есть бакетированные признаки)
        tab_dim = 0
        if feat_input_dim > 0:
            tab_dim = 64
            self.feature_branch = nn.Sequential(
                nn.Linear(feat_input_dim, tab_dim),
                nn.ReLU(),
            )
        else:
            self.feature_branch = None

        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(img_dim + tab_dim, num_labels),
        )

    def forward(self, x: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        img_feat = self.backbone(x)                         # (B, img_dim)
        if self.feature_branch is not None:
            tab_feat = self.feature_branch(features)        # (B, 64)
            combined = torch.cat([img_feat, tab_feat], dim=1)
        else:
            combined = img_feat
        return self.head(combined)                          # (B, num_labels)


def debug_model(model: ImportanceModelV2, in_channels: int = 3, feat_dim: int = 0):
    """Прогоняет фиктивный батч через модель — для дебага."""
    model.eval()
    B = 2
    x = torch.randn(B, in_channels, 224, 224)
    f = torch.randn(B, feat_dim) if feat_dim > 0 else torch.zeros(B, 0)
    with torch.no_grad():
        out = model(x, f)
    print(f"\n{'='*60}")
    print(f"  Model debug")
    print(f"{'='*60}")
    print(f"  Input image : {x.shape}")
    print(f"  Input feats : {f.shape}")
    print(f"  Output      : {out.shape}")
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters  : {total:,} total, {trainable:,} trainable")
    print(f"{'='*60}\n")
