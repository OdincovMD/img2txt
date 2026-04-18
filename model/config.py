"""Configuration for the XGBoost importance model."""

from pathlib import Path
from typing import List

import torch

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
TOP_K = 10
SEED = 42
TRAIN_SIZE = 0.2
OPTUNA_TRIALS = 30
EARLY_STOPPING_ROUNDS = 50
DEFAULT_FEATURES_CSV = "features_dataset_bucket.csv"
DEFAULT_ANNOTATIONS_CSV = "annotations.csv"
DEFAULT_CHECKPOINT_PATH = Path("model/checkpoints/xgb_importance.pkl")


def get_xgb_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
