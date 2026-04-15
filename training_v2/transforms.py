"""
Трансформации изображений v2.
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
    aug_strength: str = "mild",
):
    """
    Возвращает callable transform: np.ndarray (H,W,C) → Tensor (C,H,W).

    in_channels=3: обычный RGB
    in_channels=4: RGB + mask (4-й канал)
    """
    # Параметры аугментаций по уровню
    aug_params = {
        "none":   dict(crop_scale=(1.0, 1.0), rot=0,  cj=(0, 0, 0, 0)),
        "mild":   dict(crop_scale=(0.92, 1.0), rot=15, cj=(0.15, 0.15, 0.10, 0.0)),
        "strong": dict(crop_scale=(0.8, 1.0),  rot=30, cj=(0.3, 0.3, 0.2, 0.05)),
    }
    params = aug_params.get(aug_strength, aug_params["mild"])
    if aug_strength == "none":
        is_training = False

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    if in_channels == 3:
        if is_training:
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(
                    image_size, scale=params["crop_scale"], ratio=(0.95, 1.05)
                ),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.RandomRotation(params["rot"]),
                transforms.ColorJitter(*params["cj"]),
                transforms.ToTensor(),
                transforms.Normalize(imagenet_mean, imagenet_std),
            ])
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
            if random.random() > 0.5:
                rgb_pil = rgb_pil.transpose(PILImage.FLIP_LEFT_RIGHT)
                mask_pil = mask_pil.transpose(PILImage.FLIP_LEFT_RIGHT)
            if random.random() > 0.5:
                rgb_pil = rgb_pil.transpose(PILImage.FLIP_TOP_BOTTOM)
                mask_pil = mask_pil.transpose(PILImage.FLIP_TOP_BOTTOM)
            angle = random.uniform(-params["rot"], params["rot"])
            rgb_pil = rgb_pil.rotate(angle, resample=PILImage.BILINEAR)
            mask_pil = mask_pil.rotate(angle, resample=PILImage.NEAREST)
            b, c, s, h = params["cj"]
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
        return x

    return transform_4ch
