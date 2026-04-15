"""
Датасет v2.

Загружает:
  - RGB-изображение (+ опционально маску как 4-й канал)
  - бакетированные признаки → one-hot вектор
  - целевой бинарный вектор (24 метки) по экспертной разметке

Все парсинг-функции вынесены отдельно для удобства дебага.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from training_v2.config import LABEL_NAMES, LABEL_TO_IDX, NUM_LABELS


# ══════════════════════════════════════════════════════════════════════════════
# 1. Парсинг экспертных меток  →  целевой вектор
# ══════════════════════════════════════════════════════════════════════════════

def parse_expert_labels(raw: str) -> List[str]:
    """
    Из строки important_labels (JSON-массив "признак:значение")
    извлекает до 10 строк.

    >>> parse_expert_labels('[\"asymmetry:выраженная\", \"shape:округлая\"]')
    ['asymmetry:выраженная', 'shape:округлая']
    """
    if pd.isna(raw) or raw is None:
        return []
    if isinstance(raw, str):
        try:
            arr = json.loads(raw)
        except json.JSONDecodeError:
            return [raw.strip()] if raw.strip() else []
    else:
        arr = list(raw)
    return [str(x).strip() for x in arr if x][:10]


def label_key(s: str) -> str:
    """'asymmetry:выраженная' → 'asymmetry'"""
    return s.split(":", 1)[0].strip() if ":" in s else s.strip()


def build_target_vector(expert_strings: List[str]) -> torch.Tensor:
    """
    Бинарный вектор (NUM_LABELS,): target[i] = 1 если LABEL_NAMES[i]
    встречается среди экспертных меток.
    """
    target = torch.zeros(NUM_LABELS, dtype=torch.float32)
    for s in expert_strings:
        key = label_key(s)
        if key in LABEL_TO_IDX:
            target[LABEL_TO_IDX[key]] = 1.0
    return target


# ══════════════════════════════════════════════════════════════════════════════
# 2. Бакетированные признаки  →  one-hot вектор
# ══════════════════════════════════════════════════════════════════════════════

def parse_labels_json(raw) -> dict:
    """Извлекает словарь бакетированных меток из ячейки DataFrame."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return {}
    return {}


def build_label_vocab(df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    """
    По train-выборке собирает словарь возможных значений для каждого
    из 24 ключей LABEL_NAMES.  {label_key: {value_str: index}}.
    """
    collected: Dict[str, set] = {k: set() for k in LABEL_NAMES}
    for _, row in df.iterrows():
        d = parse_labels_json(row.get("labels_json"))
        for k in LABEL_NAMES:
            v = d.get(k)
            if isinstance(v, str) and v.strip():
                collected[k].add(v.strip())
    vocab: Dict[str, Dict[str, int]] = {}
    for k in LABEL_NAMES:
        vocab[k] = {val: i for i, val in enumerate(sorted(collected[k]))}
    return vocab


def vocab_total_dim(vocab: Dict[str, Dict[str, int]]) -> int:
    """Суммарная размерность one-hot вектора."""
    return sum(len(v) for v in vocab.values())


def encode_bucket_vector(
    labels_json_raw,
    vocab: Dict[str, Dict[str, int]],
) -> torch.Tensor:
    """
    Строит конкат one-hot сегментов по каждому ключу из LABEL_NAMES.
    Незнакомые/отсутствующие значения → нулевой сегмент.
    """
    d = parse_labels_json(labels_json_raw)
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
    return torch.cat(segments, dim=0) if segments else torch.zeros(0)


# ══════════════════════════════════════════════════════════════════════════════
# 3. Подготовка данных: merge annotations + features
# ══════════════════════════════════════════════════════════════════════════════

def prepare_dataframe(
    features_csv: str,
    annotations_csv: str,
) -> pd.DataFrame:
    """
    Объединяет features_dataset_bucket.csv с annotations.csv.
    Оставляет только строки, у которых есть экспертная разметка.

    Возвращает DataFrame с колонками:
        image_path, image_id, filename, labels_json, important_labels
    """
    feat = pd.read_csv(features_csv)
    ann = pd.read_csv(annotations_csv)

    # В annotations: image_path = "tmp/100.jpg"
    # В features:    image_path = "tmp/100.jpg", image_id = "100"
    # Объединяем по image_path
    merged = feat.merge(ann, on="image_path", how="inner")

    # Проверка
    n_ann = len(ann)
    n_merged = len(merged)
    print(f"[prepare_dataframe] Annotations: {n_ann}, "
          f"Features: {len(feat)}, Merged: {n_merged}")
    if n_merged == 0:
        raise ValueError("Нет совпадений между annotations и features!")
    if n_merged < n_ann:
        print(f"  ⚠ Потеряно {n_ann - n_merged} аннотированных изображений")

    return merged


# ══════════════════════════════════════════════════════════════════════════════
# 4. PyTorch Dataset
# ══════════════════════════════════════════════════════════════════════════════

class ImportanceDatasetV2(Dataset):
    """
    Элемент: {
        "image":    Tensor (C, H, W) — float, после transform,
        "features": Tensor (feat_dim,) — one-hot бакетированных признаков,
        "target":   Tensor (NUM_LABELS,) — бинарный вектор по разметке,
        "image_id": str,
    }
    """

    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: Union[str, Path],
        vocab: Dict[str, Dict[str, int]],
        transform=None,
        use_mask: bool = False,
        mask_dir: Optional[Union[str, Path]] = None,
        mask_suffix: str = "_mask.png",
        preload: bool = False,
        preload_size: int = 256,
    ):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.vocab = vocab
        self.transform = transform
        self.use_mask = use_mask
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.mask_suffix = mask_suffix
        self.preload_size = preload_size

        self._img_cache: Optional[np.ndarray] = None
        self._mask_cache: Optional[np.ndarray] = None

        if preload:
            self._preload_images()

    # ── Preload ──
    def _preload_images(self):
        from tqdm import tqdm
        n = len(self.df)
        cache = np.empty((n, self.preload_size, self.preload_size, 3), dtype=np.uint8)
        mask_cache = None
        if self.use_mask and self.mask_dir:
            mask_cache = np.zeros((n, self.preload_size, self.preload_size), dtype=np.uint8)

        for i in tqdm(range(n), desc="Preload images"):
            row = self.df.iloc[i]
            img = self._load_rgb(row)
            pil = Image.fromarray(img).resize(
                (self.preload_size, self.preload_size), Image.BILINEAR
            )
            cache[i] = np.array(pil)

            if mask_cache is not None:
                m = self._load_mask(row)
                if m is not None:
                    m_pil = Image.fromarray(m).resize(
                        (self.preload_size, self.preload_size), Image.NEAREST
                    )
                    mask_cache[i] = np.array(m_pil)

        self._img_cache = cache
        self._mask_cache = mask_cache
        mb = cache.nbytes / 1e6
        if mask_cache is not None:
            mb += mask_cache.nbytes / 1e6
        print(f"  Preloaded {n} images ({mb:.1f} MB)")

    # ── Загрузка одного изображения ──
    def _resolve_path(self, row) -> Path:
        p = self.image_dir / row["image_path"]
        if p.is_file():
            return p
        p2 = self.image_dir / row["filename"]
        if p2.is_file():
            return p2
        raise FileNotFoundError(f"Image not found: {p} / {p2}")

    def _load_rgb(self, row) -> np.ndarray:
        return np.array(Image.open(self._resolve_path(row)).convert("RGB"))

    def _load_mask(self, row) -> Optional[np.ndarray]:
        if not self.mask_dir:
            return None
        image_id = str(row["image_id"])
        mask_path = self.mask_dir / f"{image_id}{self.mask_suffix}"
        if not mask_path.is_file():
            return None
        m = np.array(Image.open(mask_path).convert("L"))
        return (m > 0).astype(np.uint8) * 255

    # ── __getitem__ ──
    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        # Изображение
        if self._img_cache is not None:
            image = self._img_cache[idx].copy()
        else:
            image = self._load_rgb(row)

        # Маска → 4-й канал
        if self.use_mask:
            if self._mask_cache is not None:
                mask = self._mask_cache[idx]
            else:
                mask = self._load_mask(row)
                if mask is None:
                    mask = np.zeros(image.shape[:2], dtype=np.uint8)
            image = np.concatenate([image, mask[..., None]], axis=-1)

        # Бакетированные признаки → one-hot
        features = encode_bucket_vector(row.get("labels_json"), self.vocab)

        # Целевой вектор
        expert_strings = parse_expert_labels(row.get("important_labels"))
        target = build_target_vector(expert_strings)

        # Трансформ
        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "features": features,
            "target": target,
            "image_id": str(row["image_id"]),
        }


# ══════════════════════════════════════════════════════════════════════════════
# 5. Debug-утилиты
# ══════════════════════════════════════════════════════════════════════════════

def debug_dataset_sample(ds: ImportanceDatasetV2, idx: int = 0):
    """Печатает подробности одного элемента — для дебага."""
    sample = ds[idx]
    print(f"\n{'='*60}")
    print(f"  Dataset sample #{idx}")
    print(f"{'='*60}")
    print(f"  image_id   : {sample['image_id']}")
    print(f"  image shape: {sample['image'].shape}")
    print(f"  image dtype: {sample['image'].dtype}")
    print(f"  features   : shape={sample['features'].shape}, "
          f"nonzero={sample['features'].nonzero().shape[0]}")
    print(f"  target     : shape={sample['target'].shape}, "
          f"sum={sample['target'].sum().item():.0f}")

    # Какие метки включены
    active_labels = [
        LABEL_NAMES[i] for i in range(NUM_LABELS) if sample["target"][i] > 0.5
    ]
    print(f"  active labels ({len(active_labels)}): {active_labels}")
    print(f"{'='*60}\n")
