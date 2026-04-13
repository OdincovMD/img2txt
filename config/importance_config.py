"""
Конфиг для модели отбора важных признаков (Этап 2).
Список скалярных меток и порядок выходов модели — единый источник правды.
"""

from pathlib import Path
from typing import FrozenSet, List

# Полный список всех скалярных меток, которые умеет вычислять features_to_labels().
# Не удалять — используется как справочник и при необходимости полного режима.
_ALL_LABEL_NAMES: List[str] = [
    "asymmetry",
    "borders",
    "contrast",
    "palette",
    "texture",
    "lobulation",
    "pigmentation",
    "elongation",
    "rim",
    "color_homogeneity",
    "texture_coarseness",
    "shape",
    "dominant_hue",
    "structure_order",
    "area",
    "perimeter",
    "convexity",
    "solidity",
    "extent",
    "radial_variance",
    "fractal_dimension",
    "eccentricity",
    "perimeter_area_ratio",
    "color_balance_R",
    "color_balance_G",
    "color_balance_B",
    "color_distance_euclidean",
    "delta_H_center_periphery",
    "delta_S_center_periphery",
    "delta_V_center_periphery",
    "delta_V_inner_rim",
    "delta_V_left_right",
    "delta_V_top_bottom",
    "delta_S_left_right",
    "delta_S_top_bottom",
    "entropy_H_lesion",
    "entropy_S_lesion",
    "entropy_V_lesion",
    "std_H_lesion",
    "std_S_lesion",
    "std_V_lesion",
    "mean_S_lesion",
    "mean_V_lesion",
    "glcm_energy",
    "glcm_entropy",
    "lbp_entropy",
    "lbp_mean",
    "lbp_std",
    "lbp_median",
    "percent_dark_pixels",
    "percent_white_pixels",
    "percent_red_pixels",
    "percent_blue_pixels",
    "percent_outlier_bright_pixels",
    "percent_outlier_dark_pixels",
]

# Признаки, прошедшие порог частоты ≥10% по 400 аннотированным изображениям.
# Только эти признаки попадают в labels / используются моделью.
# При изменении порога или переразметке — обновить этот список.
ACTIVE_LABELS: FrozenSet[str] = frozenset([
    "asymmetry",           # 62.7%
    "borders",             # 15.2%
    "contrast",            # 32.2%
    "palette",             # 16.2%
    "texture",             # 10.8%
    "elongation",          # 32.2%
    "rim",                 # 22.5%
    "color_homogeneity",   # 64.5%
    "shape",               # 85.0%
    "dominant_hue",        # 42.0%
    "structure_order",     # 21.5%
    "perimeter",           # 30.0%
    "fractal_dimension",   # 45.8%
    "eccentricity",        # 33.8%
    "color_distance_euclidean",    # 85.8%
    "delta_H_center_periphery",    # 36.0%
    "delta_S_center_periphery",    # 39.2%
    "delta_V_center_periphery",    # 24.2%
    "delta_V_inner_rim",           # 56.5%
    "delta_V_left_right",          # 18.5%
    "delta_V_top_bottom",          # 29.2%
    "delta_S_left_right",          # 37.5%
    "std_H_lesion",                # 41.2%
    "glcm_energy",                 # 13.8%
])

# Упорядоченный список активных меток — определяет порядок выходов модели.
LABEL_NAMES: List[str] = [l for l in _ALL_LABEL_NAMES if l in ACTIVE_LABELS]

NUM_LABELS = len(LABEL_NAMES)

# Маппинг имя_метки -> индекс (для целевого вектора и выхода модели)
LABEL_TO_IDX = {name: i for i, name in enumerate(LABEL_NAMES)}


def expert_string_to_label_key(s: str) -> str:
    """
    Из строки разметки эксперта "признак:значение" извлекает ключ признака.
    Пример: "asymmetry:умеренная" -> "asymmetry".
    """
    s = (s or "").strip()
    if ":" in s:
        return s.split(":", 1)[0].strip()
    return s
