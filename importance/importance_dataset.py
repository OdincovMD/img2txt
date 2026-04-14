"""
Датасет для модели отбора важных признаков.
Загружает изображение (и опционально маску), числовые признаки и целевой вектор.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

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


def _parse_labels_json(row_dict: dict) -> dict:
    """Извлекает бакетированный словарь меток из строки датафрейма."""
    raw = row_dict.get("labels_json")
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return {}
    return {}


def build_label_vocab(df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    """
    Собирает словарь возможных значений для каждого ключа из LABEL_NAMES по train-выборке.
    Возвращает {label_key: {value_str: index}}.
    Только строковые значения (для активных LABEL_NAMES они все строковые).
    """
    collected: Dict[str, set] = {k: set() for k in LABEL_NAMES}
    for _, row in df.iterrows():
        d = _parse_labels_json(row.to_dict())
        for k in LABEL_NAMES:
            v = d.get(k)
            if isinstance(v, str) and v.strip():
                collected[k].add(v.strip())
    vocab: Dict[str, Dict[str, int]] = {}
    for k in LABEL_NAMES:
        vocab[k] = {val: i for i, val in enumerate(sorted(collected[k]))}
    return vocab


def vocab_dim(vocab: Dict[str, Dict[str, int]]) -> int:
    return sum(len(v) for v in vocab.values())


def encode_bucket_vector(
    row_dict: dict,
    vocab: Dict[str, Dict[str, int]],
) -> torch.Tensor:
    """
    По бакетированному словарю меток строит one-hot вектор конкатенацией
    one-hot кодировок для каждого ключа из LABEL_NAMES в порядке LABEL_NAMES.
    Если значение отсутствует/незнакомо — сегмент остаётся нулевым.
    """
    d = _parse_labels_json(row_dict)
    segments = []
    for k in LABEL_NAMES:
        sub = vocab.get(k, {})
        seg = torch.zeros(len(sub), dtype=torch.float32)
        v = d.get(k)
        if isinstance(v, str):
            idx = sub.get(v.strip())
            if idx is not None:
                seg[idx] = 1.0
        segments.append(seg)
    if not segments:
        return torch.zeros(0, dtype=torch.float32)
    return torch.cat(segments, dim=0)


class ImportanceDataset(Dataset):
    """
    PyTorch Dataset: изображение (и опционально маска) + вектор числовых признаков
    + целевой вектор важности по разметке эксперта.
    """

    def __init__(
        self,
        table: pd.DataFrame,
        image_dir: Union[str, Path],
        image_col: str = "filename",
        image_id_col: str = "image_id",
        vocab: Optional[Dict[str, Dict[str, int]]] = None,
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
        image_dir: корень директории с изображениями.
        image_col: имя колонки с именем файла.
        image_id_col: имя колонки с id изображения.
        feat_mean / feat_std: статистики нормализации, вычисленные по train-выборке.
        mask_dir: если задано, маска ищется в mask_dir / (image_id + mask_suffix).
        label_columns: список колонок с метками эксперта.
        use_mask_channel: конкатенировать маску как 4-й канал к RGB.
        preload_to_memory: загрузить все изображения в RAM при инициализации.
        """
        self.table = table.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.image_col = image_col
        self.image_id_col = image_id_col
        self.vocab = vocab or {}
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.mask_suffix = mask_suffix
        self.label_columns = label_columns
        self.transform = transform
        self.use_mask_channel = use_mask_channel
        self.preload_size = preload_size
        self._cache: Optional[np.ndarray] = None
        self._mask_cache: Optional[np.ndarray] = None

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
        pil = Image.fromarray(img)
        pil = pil.resize((self.preload_size, self.preload_size), Image.BILINEAR)
        return np.array(pil)

    def _preload(self):
        from tqdm import tqdm
        n = len(self.table)
        cache = np.empty((n, self.preload_size, self.preload_size, 3), dtype=np.uint8)
        mask_cache = None
        has_mask = None
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
        self._has_mask = has_mask
        total_mb = cache.nbytes / 1e6
        if mask_cache is not None:
            total_mb += mask_cache.nbytes / 1e6
        print(f"Preloaded {n} images as contiguous array ({total_mb:.1f} MB in RAM)")

    def __len__(self) -> int:
        return len(self.table)

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
            image = self._cache[idx].copy()
            if self.use_mask_channel:
                if self._mask_cache is not None and self._has_mask[idx]:
                    mask = self._mask_cache[idx]
                else:
                    mask = np.zeros(image.shape[:2], dtype=np.uint8)
                image = np.concatenate([image, mask[..., None]], axis=-1)
        else:
            image = np.array(Image.open(self._resolve_image_path(row)).convert("RGB"))
            if self.use_mask_channel:
                mask = self._load_mask(image_id)
                if mask is None:
                    mask = np.zeros(image.shape[:2], dtype=np.uint8)
                image = np.concatenate([image, mask[..., None]], axis=-1)

        features = encode_bucket_vector(row.to_dict(), self.vocab)

        expert_strings = parse_expert_labels(row, self.label_columns)
        target = build_target_vector(expert_strings)

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "features": features,
            "target": target,
            "image_id": image_id,
            "idx": idx,
        }
