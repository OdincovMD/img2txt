"""
Трансформации изображений v2.

Режимы аугментации:
  - "none":    только Resize + Normalize (eval-режим)
  - "medical": ColorJitter + RandomErasing (безопасно для пространственных признаков)
  - "mild":    + мягкие spatial (crop, flip, rotate)
  - "strong":  + агрессивные spatial
"""

import random

import numpy as np
import torch
from PIL import Image as PILImage
from torchvision import transforms


def get_transforms(
    image_size: int = 224,
    is_training: bool = True,
    in_channels: int = 3,
    aug_strength: str = "medical",
):
    """
    Возвращает callable transform: np.ndarray (H,W,C) → Tensor (C,H,W).

    in_channels=3: обычный RGB
    in_channels=4: RGB + mask (4-й канал)
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    # ── Определяем параметры в зависимости от режима ──
    use_spatial = False   # flip / crop / rotate
    use_color = False     # ColorJitter
    use_erasing = False   # RandomErasing (cutout)
    cj = (0, 0, 0, 0)
    rot = 0
    crop_scale = (1.0, 1.0)

    if aug_strength == "none":
        is_training = False
    elif aug_strength == "medical":
        # Безопасные для пространственных признаков:
        # только цветовые + маскировка участков
        use_color = True
        use_erasing = True
        cj = (0.2, 0.2, 0.15, 0.02)
    elif aug_strength == "mild":
        use_spatial = True
        use_color = True
        use_erasing = True
        cj = (0.15, 0.15, 0.10, 0.0)
        rot = 15
        crop_scale = (0.92, 1.0)
    elif aug_strength == "strong":
        use_spatial = True
        use_color = True
        use_erasing = True
        cj = (0.3, 0.3, 0.2, 0.05)
        rot = 30
        crop_scale = (0.8, 1.0)
    else:
        raise ValueError(f"Unknown aug_strength: {aug_strength}")

    # ── 3 канала (RGB) ──
    if in_channels == 3:
        if is_training:
            steps = [transforms.ToPILImage()]

            if use_spatial:
                steps += [
                    transforms.RandomResizedCrop(
                        image_size, scale=crop_scale, ratio=(0.95, 1.05)
                    ),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomVerticalFlip(0.5),
                    transforms.RandomRotation(rot),
                ]
            else:
                steps.append(transforms.Resize((image_size, image_size)))

            if use_color:
                steps.append(transforms.ColorJitter(*cj))

            steps += [
                transforms.ToTensor(),
                transforms.Normalize(imagenet_mean, imagenet_std),
            ]

            if use_erasing:
                steps.append(
                    transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3))
                )

            return transforms.Compose(steps)
        else:
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(imagenet_mean, imagenet_std),
            ])

    # ── 4 канала (RGB + mask) ──
    def transform_4ch(img: np.ndarray) -> torch.Tensor:
        rgb = img[..., :3]
        mask = img[..., 3:4].squeeze(-1)
        if mask.max() <= 1:
            mask = (mask * 255).astype(np.uint8)

        rgb_pil = PILImage.fromarray(rgb)
        mask_pil = PILImage.fromarray(mask)

        if is_training:
            if use_spatial:
                if random.random() > 0.5:
                    rgb_pil = rgb_pil.transpose(PILImage.FLIP_LEFT_RIGHT)
                    mask_pil = mask_pil.transpose(PILImage.FLIP_LEFT_RIGHT)
                if random.random() > 0.5:
                    rgb_pil = rgb_pil.transpose(PILImage.FLIP_TOP_BOTTOM)
                    mask_pil = mask_pil.transpose(PILImage.FLIP_TOP_BOTTOM)
                angle = random.uniform(-rot, rot)
                rgb_pil = rgb_pil.rotate(angle, resample=PILImage.BILINEAR)
                mask_pil = mask_pil.rotate(angle, resample=PILImage.NEAREST)

            if use_color:
                b, c, s, h = cj
                if any(v > 0 for v in (b, c, s, h)):
                    rgb_pil = transforms.ColorJitter(b, c, s, h)(rgb_pil)

        rgb_pil = rgb_pil.resize((image_size, image_size))
        mask_pil = mask_pil.resize((image_size, image_size))

        rgb_t = transforms.ToTensor()(rgb_pil)      # (3, H, W) [0..1]
        mask_t = torch.from_numpy(
            np.array(mask_pil, dtype=np.float32)
        ).unsqueeze(0) / 255.0                       # (1, H, W) [0..1]

        x = torch.cat([rgb_t, mask_t], dim=0)        # (4, H, W)
        mean_t = torch.tensor(imagenet_mean).view(3, 1, 1)
        std_t  = torch.tensor(imagenet_std).view(3, 1, 1)
        x[:3] = (x[:3] - mean_t) / std_t

        if is_training and use_erasing:
            # RandomErasing только на RGB каналах (0:3)
            eraser = transforms.RandomErasing(p=0.3, scale=(0.02, 0.15))
            x[:3] = eraser(x[:3])

        return x

    return transform_4ch
