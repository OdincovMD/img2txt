"""Model-only config for feature-importance ranking."""

from typing import FrozenSet, List

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

NUM_LABELS = len(LABEL_NAMES)

# Табличный вход модели — one-hot кодирование бакетированных значений тех же LABEL_NAMES.
# Размер вектора вычисляется динамически из словаря (build_label_vocab по train-выборке)
# и сохраняется в чекпойнт. Ключей FEAT_KEYS/FEAT_DIM больше нет — они неявно равны LABEL_NAMES
# и сумме размеров словарей на каждый ключ соответственно.
