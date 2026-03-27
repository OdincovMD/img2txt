"""
Инференс модели отбора важных признаков: загрузка чекпоинта, предсказание топ-10 меток в формате "признак:значение".
"""

import json
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import torch

from importance_config import LABEL_NAMES, NUM_LABELS
from importance_model import ImportanceModel
from threshold_rules import row_to_labels

try:
    from torchvision import transforms
except ImportError:
    transforms = None


def load_model(
    checkpoint_path: Union[str, Path],
    device: Optional[torch.device] = None,
    backbone: str = "efficientnet_b0",
    in_channels: int = 3,
) -> ImportanceModel:
    """Загружает модель из чекпоинта train_importance."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = ckpt.get("args", {})
    backbone = args.get("backbone", backbone)
    in_channels = 4 if args.get("use_mask") else 3
    model = ImportanceModel(
        backbone_name=backbone,
        num_labels=NUM_LABELS,
        pretrained=False,
        in_channels=in_channels,
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    return model


def get_inference_transform(image_size: int = 224, in_channels: int = 3):
    if transforms is None:
        raise ImportError("torchvision required for inference transform.")
    if in_channels == 3:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def transform_4ch(img):
        from PIL import Image as PILImage
        rgb = img[..., :3]
        mask = img[..., 3:4].squeeze(-1)
        if mask.max() <= 1:
            mask = (np.clip(mask * 255, 0, 255)).astype(np.uint8)
        rgb_pil = PILImage.fromarray(rgb).resize((image_size, image_size))
        mask_pil = PILImage.fromarray(mask).resize((image_size, image_size))
        rgb_t = transforms.ToTensor()(rgb_pil)
        mask_t = torch.from_numpy(np.array(mask_pil)).float().unsqueeze(0) / 255.0
        x = torch.cat([rgb_t, mask_t], dim=0)
        x[0:3] = (x[0:3] - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return x
    return transform_4ch


def top10_indices_to_labels(
    top10_indices: List[int],
    labels_dict: dict,
    k: int = 10,
) -> List[str]:
    """
    Переводит индексы топ-10 в строки "признак:значение".
    labels_dict: словарь из row_to_labels (или features_to_labels) — ключ = имя признака, значение = строка (или dict/list пропускаем).
    """
    result = []
    for i in top10_indices:
        if i >= len(LABEL_NAMES):
            continue
        key = LABEL_NAMES[i]
        val = labels_dict.get(key)
        if isinstance(val, str):
            result.append(f"{key}:{val}")
        elif isinstance(val, dict):
            for subk, subv in val.items():
                if isinstance(subv, str):
                    result.append(f"{key}.{subk}:{subv}")
            if not val and key in labels_dict:
                result.append(f"{key}:—")
        if len(result) >= k:
            break
    return result[:k]


def predict_top10(
    model: ImportanceModel,
    image: np.ndarray,
    labels_dict: dict,
    device: torch.device,
    transform=None,
    image_size: int = 224,
    k: int = 10,
    mask: Optional[np.ndarray] = None,
) -> List[str]:
    """
    Один сэмпл: изображение (H,W,3) или (H,W,4) с маской.
    labels_dict — полный набор меток для этого изображения (из row_to_labels).
    Возвращает список из k строк "признак:значение".
    """
    # Model may expect 4 channels (RGB+mask); if no mask, use zeros
    if hasattr(model, "head") and next(model.backbone.parameters()).shape[1] == 4 and image.shape[-1] == 3:
        if mask is None:
            mask = np.zeros((image.shape[0], image.shape[1], 1), dtype=np.uint8)
        else:
            mask = np.expand_dims(mask.astype(np.uint8), axis=-1)
        image = np.concatenate([image, mask], axis=-1)
    elif mask is not None and image.shape[-1] == 3:
        mask = np.expand_dims(mask, axis=-1)
        image = np.concatenate([image, mask], axis=-1)
    if transform is None:
        transform = get_inference_transform(image_size, in_channels=image.shape[-1])
    x = transform(image)
    x = x.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
    probs = torch.sigmoid(logits[0]).cpu().numpy()
    top10_idx = np.argsort(probs)[::-1][:k].tolist()
    return top10_indices_to_labels(top10_idx, labels_dict, k=k)


def predict_top10_from_row(
    model: ImportanceModel,
    row: pd.Series,
    image_dir: Union[str, Path],
    device: torch.device,
    image_col: str = "filename",
    image_id_col: str = "image_id",
    mask_dir: Optional[Union[str, Path]] = None,
    mask_suffix: str = "_mask.png",
    transform=None,
    image_size: int = 224,
    k: int = 10,
) -> List[str]:
    """
    Предсказание по строке датасета (с колонками features_json, filename и т.д.).
    Загружает изображение по image_dir/row[image_col], при необходимости маску.
    Метки для формата "признак:значение" берутся из row_to_labels(row).
    """
    from PIL import Image

    image_dir = Path(image_dir)
    fname = row.get(image_col) or row.get("filename")
    image_id = row.get(image_id_col) or row.get("image_id", "")
    image_path = image_dir / fname
    if not image_path.is_file():
        image_path = image_dir / image_id / fname
    if not image_path.is_file():
        return []

    image = np.array(Image.open(image_path).convert("RGB"))
    labels_dict = row_to_labels(row.to_dict())
    mask = None
    if mask_dir:
        mask_path = Path(mask_dir) / f"{image_id}{mask_suffix}"
        if mask_path.is_file():
            mask = np.array(Image.open(mask_path).convert("L")) > 0

    return predict_top10(model, image, labels_dict, device, transform=transform, image_size=image_size, k=k, mask=mask)


def add_important_labels_to_dataframe(
    df: pd.DataFrame,
    model: ImportanceModel,
    device: torch.device,
    image_dir: Union[str, Path],
    image_col: str = "filename",
    image_id_col: str = "image_id",
    mask_dir: Optional[Union[str, Path]] = None,
    image_size: int = 224,
    k: int = 10,
) -> pd.DataFrame:
    """
    Добавляет к df колонку important_labels: список из k строк "признак:значение" для каждой строки.
    В df должны быть колонки features_json (и при необходимости filename, image_id) для row_to_labels.
    """
    in_channels = next(model.backbone.parameters()).shape[1]
    transform = get_inference_transform(image_size, in_channels=in_channels)
    out = []
    for idx, row in df.iterrows():
        pred = predict_top10_from_row(
            model, row, image_dir, device,
            image_col=image_col, image_id_col=image_id_col,
            mask_dir=mask_dir, transform=transform, image_size=image_size, k=k,
        )
        out.append(pred)
    df = df.copy()
    df["important_labels"] = out
    return df
