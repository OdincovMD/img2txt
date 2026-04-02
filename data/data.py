"""Построение датасета: мета из файла классификации + признаки."""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from config.config import FEATURE_ROUTING, _safe_number


@dataclass
class MetaRow:
    image_id: str
    filename: str
    class_dir: str
    diagnosis: Optional[str] = None
    feature_type: Optional[str] = None
    structure: Optional[str] = None
    properties: Optional[str] = None


def build_row(meta: MetaRow, features_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Строит строку датасета: мета + features_json (признаки в структурированном виде)."""
    row = asdict(meta)
    features_tree: Dict[str, Any] = {
        "color": {"local": {}, "global": {}},
        "shape": {},
        "border": {},
        "texture": {},
    }

    for key, v in features_dict.items():
        if key not in FEATURE_ROUTING:
            continue
        value = _safe_number(v)
        block, nicename, unit = FEATURE_ROUTING[key]
        if block == "color.local":
            features_tree["color"]["local"][key] = [nicename, value, unit]
        elif block == "color.global":
            features_tree["color"]["global"][key] = [nicename, value, unit]
        elif block == "shape":
            features_tree["shape"][key] = [nicename, value, unit]
        elif block == "border":
            features_tree["border"][key] = [nicename, value, unit]
        else:
            features_tree["texture"][key] = [nicename, value, unit]

    row["features_json"] = json.dumps(features_tree, ensure_ascii=False)
    return row


def iter_images(root: Path, recursive: bool = False) -> List[Path]:
    """Итерация по изображениям. recursive=True — искать в поддиректориях."""
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    if not root.is_dir():
        return []
    if recursive:
        return [p for p in root.rglob("*") if p.suffix.lower() in exts and p.is_file()]
    return [p for p in root.iterdir() if p.suffix.lower() in exts]


def build_dataset(
    root_dir: str,
    meta_table: pd.DataFrame,
    mask_fn,
    extractors: Dict[str, Any],
    recursive: bool = True,
) -> pd.DataFrame:
    """
    Строит датасет: изображения + маски + признаки.
    Классификация (diagnosis, feature_type, structure, properties) берётся из meta_table по filename.
    recursive=True — искать изображения в поддиректориях root_dir.
    """
    rows = []
    meta_idx = meta_table.set_index("filename").to_dict(orient="index")
    root = Path(root_dir)
    images = iter_images(root, recursive=recursive)

    for img_path in tqdm(images, desc="Извлечение признаков"):
        fname = img_path.name
        image_id = fname.rsplit(".", 1)[0]
        meta_info = meta_idx.get(fname, {})
        meta = MetaRow(
            image_id=image_id,
            filename=fname,
            class_dir="all",
            diagnosis=meta_info.get("diagnosis"),
            feature_type=meta_info.get("feature_type"),
            structure=meta_info.get("structure"),
            properties=meta_info.get("properties"),
        )

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = mask_fn(str(img_path))

        feats: Dict[str, Any] = {}
        for name, fn in extractors.items():
            try:
                if name in ("global_color", "local_color", "texture"):
                    feats.update(fn(image, mask))
                else:
                    feats.update(fn(mask))
            except Exception as e:
                print(f"Error in {name} for {image_id}: {e}")

        row = build_row(meta, feats)
        rows.append(row)

    df = pd.DataFrame(rows)
    meta_cols = [
        "image_id", "filename", "class_dir", "diagnosis",
        "feature_type", "structure", "properties", "features_json",
    ]
    feat_cols = sorted([c for c in df.columns if c not in meta_cols])
    return df[meta_cols + feat_cols]
