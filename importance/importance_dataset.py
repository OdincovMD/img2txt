"""
Датасет для модели отбора важных признаков.
Загружает изображение (и опционально маску), строит целевой вектор по разметке эксперта.
"""

import json
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from config.importance_config import LABEL_NAMES, LABEL_TO_IDX, expert_string_to_label_key


def parse_expert_labels(row, label_columns: Optional[List[str]] = None) -> List[str]:
    """
    Из строки датасета извлекает список из 10 меток эксперта.
    Поддерживает:
    - Колонку important_labels (JSON-массив строк).
    - Колонки label_1, label_2, ..., label_10.
    - label_columns — явный список имён колонок (например ["label_1", ..., "label_10"]).
    """
    if label_columns:
        out = []
        for c in label_columns:
            v = row.get(c)
            if pd.isna(v) or v is None or str(v).strip() == "":
                continue
            out.append(str(v).strip())
        return out[:10]

    raw = row.get("important_labels")
    if raw is not None and not pd.isna(raw):
        if isinstance(raw, str):
            try:
                arr = json.loads(raw)
            except Exception:
                arr = [raw]
        else:
            arr = list(raw)
        return [str(x).strip() for x in arr if x][:10]

    out = []
    for i in range(1, 11):
        v = row.get(f"label_{i}")
        if v is not None and not pd.isna(v) and str(v).strip():
            out.append(str(v).strip())
    return out[:10]


def build_target_vector(expert_label_strings: List[str]) -> torch.Tensor:
    """
    По списку строк эксперта "признак:значение" строит бинарный вектор длины NUM_LABELS.
    Индекс i равен 1, если признак LABEL_NAMES[i] встречается среди экспертных меток.
    """
    keys_present = set()
    for s in expert_label_strings:
        key = expert_string_to_label_key(s)
        if key in LABEL_TO_IDX:
            keys_present.add(key)
    target = torch.zeros(len(LABEL_NAMES), dtype=torch.float32)
    for key in keys_present:
        target[LABEL_TO_IDX[key]] = 1.0
    return target


class ImportanceDataset(Dataset):
    """
    PyTorch Dataset: изображение (и опционально маска) + целевой вектор важности по разметке эксперта.
    """

    def __init__(
        self,
        table: pd.DataFrame,
        image_dir: Union[str, Path],
        image_col: str = "filename",
        image_id_col: str = "image_id",
        mask_dir: Optional[Union[str, Path]] = None,
        mask_suffix: str = "_mask.png",
        label_columns: Optional[List[str]] = None,
        transform=None,
        use_mask_channel: bool = False,
        preload_to_memory: bool = False,
        preload_size: int = 256,
    ):
        """
        table: DataFrame с колонками для пути к изображению и разметкой эксперта.
        image_dir: корень директории с изображениями (к колонке image_col подставляется путь).
        image_col: имя колонки с именем файла (например filename).
        image_id_col: имя колонки с id изображения (для маски: image_id + mask_suffix).
        mask_dir: если задано, маска ищется в mask_dir / (image_id + mask_suffix).
        label_columns: список колонок с метками эксперта (label_1 ... label_10 или свои).
        transform: torchvision transform (нормализация, ресайз и т.д.).
        use_mask_channel: если True, конкатенировать маску как 4-й канал к RGB (нужен mask_dir).
        preload_to_memory: если True, все картинки декодируются и ресайзятся один раз
            при инициализации и хранятся в RAM (ускоряет обучение на медленных ФС
            типа kaggle fuse mount). Маски также кэшируются.
        preload_size: размер короткой стороны при препроцессинге (картинки ресайзятся
            до preload_size на короткой стороне с сохранением пропорций).
        """
        self.table = table.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.image_col = image_col
        self.image_id_col = image_id_col
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.mask_suffix = mask_suffix
        self.label_columns = label_columns
        self.transform = transform
        self.use_mask_channel = use_mask_channel
        self.preload_size = preload_size
        self._cache: Optional[List[np.ndarray]] = None
        self._mask_cache: Optional[List[Optional[np.ndarray]]] = None

        if preload_to_memory:
            self._preload()

    def _resolve_image_path(self, row) -> Path:
        fname = row[self.image_col]
        image_id = row[self.image_id_col]
        image_path = self.image_dir / fname
        if not image_path.is_file():
            image_path = self.image_dir / image_id / fname
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")
        return image_path

    def _resize_square(self, img: np.ndarray) -> np.ndarray:
        """Ресайз в квадрат preload_size × preload_size (без сохранения пропорций).
        Аугментация при обучении делает RandomResizedCrop, так что точная пропорция не критична."""
        pil = Image.fromarray(img)
        pil = pil.resize((self.preload_size, self.preload_size), Image.BILINEAR)
        return np.array(pil)

    def _preload(self):
        from tqdm import tqdm
        n = len(self.table)
        # Один большой непрерывный массив — избегаем fork-copy миллионов мелких объектов
        cache = np.empty((n, self.preload_size, self.preload_size, 3), dtype=np.uint8)
        mask_cache = None
        if self.use_mask_channel and self.mask_dir:
            mask_cache = np.zeros((n, self.preload_size, self.preload_size), dtype=np.uint8)
            has_mask = np.zeros(n, dtype=bool)

        for i in tqdm(range(n), desc="Preloading images"):
            row = self.table.iloc[i]
            path = self._resolve_image_path(row)
            img = np.array(Image.open(path).convert("RGB"))
            cache[i] = self._resize_square(img)

            if mask_cache is not None:
                image_id = row[self.image_id_col]
                mask_path = self.mask_dir / f"{image_id}{self.mask_suffix}"
                if mask_path.is_file():
                    m_pil = Image.open(mask_path).convert("L").resize(
                        (self.preload_size, self.preload_size), Image.NEAREST
                    )
                    m = np.array(m_pil, dtype=np.uint8)
                    mask_cache[i] = (m > 0).astype(np.uint8) * 255
                    has_mask[i] = True

        self._cache = cache
        self._mask_cache = mask_cache
        if mask_cache is not None:
            self._has_mask = has_mask
        total_mb = cache.nbytes / 1e6
        if mask_cache is not None:
            total_mb += mask_cache.nbytes / 1e6
        print(f"Preloaded {n} images as contiguous array ({total_mb:.1f} MB in RAM)")

    def __len__(self) -> int:
        return len(self.table)

    def _load_image(self, path: Path) -> np.ndarray:
        img = Image.open(path).convert("RGB")
        return np.array(img)

    def _load_mask(self, image_id: str) -> Optional[np.ndarray]:
        if not self.mask_dir:
            return None
        mask_path = self.mask_dir / f"{image_id}{self.mask_suffix}"
        if not mask_path.is_file():
            return None
        mask = np.array(Image.open(mask_path).convert("L"))
        return (mask > 0).astype(np.uint8)

    def __getitem__(self, idx: int):
        row = self.table.iloc[idx]
        image_id = row[self.image_id_col]

        if self._cache is not None:
            # Копия из contiguous-кэша — augmentation не должен модифицировать кэш
            image = self._cache[idx].copy()
            if self.use_mask_channel:
                if self._mask_cache is not None and self._has_mask[idx]:
                    mask = self._mask_cache[idx]
                else:
                    mask = np.zeros(image.shape[:2], dtype=np.uint8)
                image = np.concatenate([image, mask[..., None]], axis=-1)
        else:
            image_path = self._resolve_image_path(row)
            image = self._load_image(image_path)
            if self.use_mask_channel:
                mask = self._load_mask(image_id) if self.mask_dir else None
                if mask is None:
                    mask = np.zeros(image.shape[:2], dtype=np.uint8)
                mask = np.expand_dims(mask, axis=-1)
                image = np.concatenate([image, mask], axis=-1)

        expert_strings = parse_expert_labels(row, self.label_columns)
        target = build_target_vector(expert_strings)

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "target": target,
            "image_id": image_id,
            "idx": idx,
        }
