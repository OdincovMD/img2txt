"""
Конфигурация пайплайна обучения v2.

Единый источник правды: 24 активных признака, пути, гиперпараметры.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional


# ── 24 активных признака (после сжатия пространства) ──────────────────────────
# Порядок фиксирован — определяет индексы выходов модели.
LABEL_NAMES: List[str] = [
    "asymmetry",
    "borders",
    "contrast",
    "palette",
    "texture",
    "elongation",
    "rim",
    "color_homogeneity",
    "shape",
    "dominant_hue",
    "structure_order",
    "perimeter",
    "fractal_dimension",
    "eccentricity",
    "color_distance_euclidean",
    "delta_H_center_periphery",
    "delta_S_center_periphery",
    "delta_V_center_periphery",
    "delta_V_inner_rim",
    "delta_V_left_right",
    "delta_V_top_bottom",
    "delta_S_left_right",
    "std_H_lesion",
    "glcm_energy",
]

NUM_LABELS: int = len(LABEL_NAMES)          # 24
LABEL_TO_IDX = {n: i for i, n in enumerate(LABEL_NAMES)}

# Сколько признаков выбираем как «важные» при инференсе
TOP_K: int = 10


# ── Гиперпараметры ────────────────────────────────────────────────────────────
@dataclass
class TrainConfig:
    """Все настройки обучения в одном месте."""

    # Данные
    features_csv: str = "features_dataset_bucket.csv"
    annotations_csv: str = "annotations.csv"
    image_dir: str = "."                      # корень, содержащий tmp/
    mask_dir: Optional[str] = None
    mask_suffix: str = "_mask.png"

    # Разбивка
    val_ratio: float = 0.15
    seed: int = 42

    # Модель
    backbone: Literal["efficientnet_b0", "resnet18", "resnet34"] = "efficientnet_b0"
    pretrained: bool = True
    use_mask: bool = False                    # 4-й канал = маска
    dropout: float = 0.3
    image_size: int = 224

    # Обучение
    epochs: int = 50
    batch_size: int = 16
    lr: float = 1e-4
    backbone_lr_factor: float = 0.1           # lr бэкбона = lr * factor
    weight_decay: float = 1e-3
    freeze_backbone_epochs: int = 5
    label_smoothing: float = 0.05

    # Лосс
    loss_type: Literal["bce", "focal"] = "bce"
    focal_gamma: float = 2.0
    use_pos_weight: bool = True
    pos_weight_cap: float = 10.0

    # Аугментации
    aug_strength: Literal["none", "mild", "strong"] = "mild"

    # Система
    num_workers: int = 0
    use_amp: bool = True
    preload_to_memory: bool = True
    preload_size: int = 256

    # Выход
    out_dir: str = "training_v2/checkpoints"

    def __post_init__(self):
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)
