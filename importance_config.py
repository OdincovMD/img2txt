"""
Конфиг для модели отбора важных признаков (Этап 2).
Список скалярных меток и порядок выходов модели — единый источник правды.
"""

from pathlib import Path
from typing import List

# Упорядоченный список ключей скалярных меток (без center_periphery и pigment_inclusions).
# Совпадает с ключами в features_to_labels(), дающими строку.
LABEL_NAMES: List[str] = [
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
