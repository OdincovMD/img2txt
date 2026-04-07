"""
Step 3: Feature importance ranking.
Loads trained model, predicts top-k important features per image.
Public API: rank_features_batch(df, ...) → df with 'important_labels' column.
"""

import json
import random
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from config.importance_config import LABEL_NAMES, NUM_LABELS
from importance.importance_model import ImportanceModel
from analysis.threshold_rules import row_to_labels

try:
    from torchvision import transforms
except ImportError:
    transforms = None


def load_model(
    checkpoint_path: Union[str, Path],
    device: Optional[torch.device] = None,
) -> ImportanceModel:
    """Load importance model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = ckpt.get("args", {})
    backbone = args.get("backbone", "efficientnet_b0")
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


def _get_inference_transform(image_size: int = 224, in_channels: int = 3):
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


def _top10_indices_to_labels(
    top10_indices: List[int],
    labels_dict: dict,
    k: int = 10,
) -> List[str]:
    """Convert model output indices to 'feature:value' strings."""
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


def _random_extra_labels(
    selected_indices: List[int],
    labels_dict: dict,
    n: int = 2,
) -> List[str]:
    """Pick n random unselected labels and return as 'feature:value' strings."""
    selected_set = set(selected_indices)
    unselected = [i for i in range(len(LABEL_NAMES)) if i not in selected_set]
    random.shuffle(unselected)
    extras = []
    for i in unselected:
        if len(extras) >= n:
            break
        key = LABEL_NAMES[i]
        val = labels_dict.get(key)
        if isinstance(val, str):
            extras.append(f"{key}:{val}")
        elif isinstance(val, dict):
            for subk, subv in val.items():
                if isinstance(subv, str):
                    extras.append(f"{key}.{subk}:{subv}")
                    break
    return extras[:n]


def _predict_top10(
    model: ImportanceModel,
    image: np.ndarray,
    labels_dict: dict,
    device: torch.device,
    transform=None,
    image_size: int = 224,
    k: int = 10,
    mask: Optional[np.ndarray] = None,
    n_random: int = 2,
) -> List[str]:
    """Single-sample prediction: image (H,W,3) + labels → top-k + n_random 'feature:value' strings."""
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
        transform = _get_inference_transform(image_size, in_channels=image.shape[-1])
    x = transform(image)
    x = x.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
    probs = torch.sigmoid(logits[0]).cpu().numpy()
    top_idx = np.argsort(probs)[::-1][:k].tolist()
    selected = _top10_indices_to_labels(top_idx, labels_dict, k=k)
    if n_random > 0:
        selected += _random_extra_labels(top_idx, labels_dict, n=n_random)
    return selected


def _predict_from_row(
    model: ImportanceModel,
    row: pd.Series,
    device: torch.device,
    mask_dir: Optional[Union[str, Path]] = None,
    mask_suffix: str = "_mask.png",
    transform=None,
    image_size: int = 224,
    k: int = 10,
    n_random: int = 2,
) -> List[str]:
    """Predict from a DataFrame row. Reads image from row['image_path'], labels from row['features_json']."""
    from PIL import Image

    image_path = row.get("image_path")
    if not image_path or not Path(image_path).is_file():
        return []

    image = np.array(Image.open(image_path).convert("RGB"))
    labels_dict = row_to_labels(row.to_dict())

    mask = None
    if mask_dir:
        image_id = row.get("image_id", Path(image_path).stem)
        mask_path = Path(mask_dir) / f"{image_id}{mask_suffix}"
        if mask_path.is_file():
            mask = np.array(Image.open(mask_path).convert("L")) > 0

    return _predict_top10(
        model, image, labels_dict, device,
        transform=transform, image_size=image_size, k=k, mask=mask, n_random=n_random,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rank_features_batch(
    df: pd.DataFrame,
    importance_model_path: Optional[Union[str, Path]] = None,
    device: Optional[torch.device] = None,
    image_size: int = 224,
    k: int = 10,
    n_random: int = 2,
    mask_dir: Optional[Union[str, Path]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Step 3: Rank features by importance for all rows in df.

    Args:
        df: DataFrame with 'image_path' and 'features_json' columns (from steps 1-2).
        importance_model_path: Path to model checkpoint.
            If None, 'important_labels' will be empty lists.
        device: torch device (defaults to cuda if available)
        image_size: Input image size for model
        k: Number of top features selected by model
        n_random: Number of random unselected features to append (default 2, total = k + n_random)
        mask_dir: Optional directory with mask files
        verbose: Show progress bar

    Returns:
        df with added column:
        - important_labels: list of k + n_random strings ["feature:value", ...]
    """
    df = df.copy()

    if importance_model_path is None:
        df['important_labels'] = [[] for _ in range(len(df))]
        return df

    try:
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = load_model(importance_model_path, device=device)
        in_channels = next(model.backbone.parameters()).shape[1]
        transform = _get_inference_transform(image_size, in_channels=in_channels)

        results = []
        iterator = tqdm(df.iterrows(), total=len(df)) if verbose else df.iterrows()
        for _, row in iterator:
            pred = _predict_from_row(
                model, row, device,
                mask_dir=mask_dir, transform=transform,
                image_size=image_size, k=k, n_random=n_random,
            )
            results.append(pred)

        df['important_labels'] = results

    except Exception as e:
        print(f"Warning: Could not rank features: {e}")
        df['important_labels'] = [[] for _ in range(len(df))]

    return df
