"""Map numeric features to scalar labels and flat bucket columns."""

import math
from typing import Any, Optional

from service.app.pipeline_steps.step1_features.derived_features import COMPOSITE_FEATURE_COLUMNS
from service.app.pipeline_steps.step1_features.feature_schema import FEATURE_COLUMNS
from service.app.pipeline_steps.step2_bucketing.bucket_schema import bucket_column_name
from service.app.pipeline_steps.step2_bucketing.threshold_config import THRESHOLDS, SCALAR, apply_threshold


def _get_num(values: dict[str, Any], key: str) -> Optional[float]:
    value = values.get(key)
    if value is None:
        return None
    if isinstance(value, (int, float)) and not (isinstance(value, float) and math.isnan(value)):
        return float(value)
    return None


def _label(features: dict[str, Any], key: str, default: str = "неопределено") -> str:
    if key not in THRESHOLDS:
        return default
    return apply_threshold(_get_num(features, key), THRESHOLDS[key], default) or default


def _asymmetry(features: dict[str, Any]) -> str:
    return apply_threshold(_get_num(features, "asymmetry_agg"), THRESHOLDS["asymmetry_agg"], "слабая")


def _borders(features: dict[str, Any]) -> str:
    circularity = _get_num(features, "circularity")
    perimeter_area_ratio = _get_num(features, "perimeter_area_ratio")
    circ_high, para_high = SCALAR["borders_circ_high"], SCALAR["borders_para_high"]
    circ_mid, para_mid = SCALAR["borders_circ_mid"], SCALAR["borders_para_mid"]

    if circularity is not None and perimeter_area_ratio is not None:
        if circularity >= circ_high and perimeter_area_ratio < para_high:
            return "ровные"
        if circularity >= circ_mid and perimeter_area_ratio < para_mid:
            return "умеренно неровные"
    if circularity is not None:
        if circularity >= circ_high:
            return "ровные"
        if circularity >= circ_mid:
            return "умеренно неровные"
    return "фестончатые"


def _palette(features: dict[str, Any]) -> str:
    return apply_threshold(_get_num(features, "palette_variety"), THRESHOLDS["palette_variety"], "монотонная")


def _pigmentation(features: dict[str, Any]) -> str:
    dark = _get_num(features, "percent_dark_pixels") or 0.0
    red = _get_num(features, "percent_red_pixels") or 0.0
    blue = _get_num(features, "percent_blue_pixels") or 0.0
    white = _get_num(features, "percent_white_pixels") or 0.0
    white_th, red_blue_th = SCALAR["pigmentation_white"], SCALAR["pigmentation_red_blue"]
    dark_high, dark_mid = SCALAR["pigmentation_dark_high"], SCALAR["pigmentation_dark_mid"]

    if white > white_th:
        return "с выраженными обесцвеченными участками"
    if red > red_blue_th or blue > red_blue_th:
        return "с примесью красного или синего"
    if dark > dark_high:
        return "преимущественно тёмная"
    if dark > dark_mid:
        return "умеренно пигментированная"
    return "слабо пигментированная"


def _elongation(features: dict[str, Any]) -> str:
    score = _get_num(features, "elongation_score")
    if score is None:
        return "округлая"
    low, mid = SCALAR["elongation_score_low"], SCALAR["elongation_score_mid"]
    if score < low:
        return "округлая"
    if score < mid:
        return "умеренно вытянутая"
    return "вытянутая"


def _rim(features: dict[str, Any]) -> str:
    delta_v = _get_num(features, "delta_V_inner_rim")
    threshold = SCALAR["rim_abs"]
    if delta_v is None or abs(delta_v) < threshold:
        return "без выраженного ободка"
    return "светлый ободок по краю" if delta_v > 0 else "тёмный ободок по краю"


def _color_homogeneity(features: dict[str, Any]) -> str:
    score = _get_num(features, "color_homogeneity_agg") or 0.0
    high, mid = SCALAR["color_homogeneity_agg_high"], SCALAR["color_homogeneity_agg_mid"]
    if score < high:
        return "цвет однородный"
    if score < mid:
        return "цвет умеренно неоднородный"
    return "цвет неоднородный"


def _dominant_hue(features: dict[str, Any]) -> str:
    mean_h = _get_num(features, "mean_H_lesion")
    if mean_h is None:
        return "неопределенный"
    label = apply_threshold(mean_h, THRESHOLDS["mean_H_lesion"], "неопределенный")
    if label == "коричневый":
        red = _get_num(features, "color_balance_R") or 0.0
        green = _get_num(features, "color_balance_G") or 0.0
        if green > red * 1.2:
            return "зеленоватый"
    return label or "неопределенный"


def _lobulation(features: dict[str, Any]) -> str:
    score = _get_num(features, "lobulation_score") or 0.0
    low, mid = SCALAR["lobulation_score_low"], SCALAR["lobulation_score_mid"]
    if score < low:
        return "лобуляции отсутствуют"
    if score < mid:
        return "лобуляции слабые"
    return "лобуляции выраженные"


def bucket_feature_values(features: dict[str, Any]) -> dict[str, str]:
    return {
        "asymmetry": _asymmetry(features),
        "borders": _borders(features),
        "contrast": _label(features, "color_distance_deltaE", "умеренный"),
        "palette": _palette(features),
        "texture": _label(features, "glcm_homogeneity", "умеренно неоднородная"),
        "lobulation": _lobulation(features),
        "pigmentation": _pigmentation(features),
        "elongation": _elongation(features),
        "rim": _rim(features),
        "color_homogeneity": _color_homogeneity(features),
        "texture_coarseness": _label(features, "glcm_contrast", "средняя"),
        "shape": _label(features, "circularity", "неопределенная"),
        "dominant_hue": _dominant_hue(features),
        "structure_order": _label(features, "lbp_uniformity", "неопределенная"),
        "area": _label(features, "area", "неопределенный"),
        "perimeter": _label(features, "perimeter", "неопределенный"),
        "convexity": _label(features, "convexity", "неопределенная"),
        "solidity": _label(features, "solidity", "неопределенная"),
        "extent": _label(features, "extent", "неопределенное"),
        "radial_variance": _label(features, "radial_variance", "неопределенная"),
        "fractal_dimension": _label(features, "fractal_dimension", "неопределенная"),
        "eccentricity": _label(features, "eccentricity", "неопределенная"),
        "perimeter_area_ratio": _label(features, "perimeter_area_ratio", "неопределенное"),
        "color_balance_R": _label(features, "color_balance_R", "неопределено"),
        "color_balance_G": _label(features, "color_balance_G", "неопределено"),
        "color_balance_B": _label(features, "color_balance_B", "неопределено"),
        "color_distance_euclidean": _label(features, "color_distance_euclidean", "неопределенный"),
        "delta_H_center_periphery": _label(features, "delta_H_center_periphery", "неопределено"),
        "delta_S_center_periphery": _label(features, "delta_S_center_periphery", "неопределено"),
        "delta_V_center_periphery": _label(features, "delta_V_center_periphery", "неопределено"),
        "delta_V_inner_rim": _label(features, "delta_V_inner_rim", "неопределено"),
        "delta_V_left_right": _label(features, "delta_V_left_right", "неопределено"),
        "delta_V_top_bottom": _label(features, "delta_V_top_bottom", "неопределено"),
        "delta_S_left_right": _label(features, "delta_S_left_right", "неопределено"),
        "delta_S_top_bottom": _label(features, "delta_S_top_bottom", "неопределено"),
        "entropy_H_lesion": _label(features, "entropy_H_lesion", "неопределенное"),
        "entropy_S_lesion": _label(features, "entropy_S_lesion", "неопределенное"),
        "entropy_V_lesion": _label(features, "entropy_V_lesion", "неопределенное"),
        "std_H_lesion": _label(features, "std_H_lesion", "неопределенная"),
        "std_S_lesion": _label(features, "std_S_lesion", "неопределенная"),
        "std_V_lesion": _label(features, "std_V_lesion", "неопределенная"),
        "mean_S_lesion": _label(features, "mean_S_lesion", "неопределенная"),
        "mean_V_lesion": _label(features, "mean_V_lesion", "неопределенная"),
        "glcm_energy": _label(features, "glcm_energy", "неопределенная"),
        "glcm_entropy": _label(features, "glcm_entropy", "неопределенная"),
        "lbp_entropy": _label(features, "lbp_entropy", "неопределенная"),
        "lbp_mean": _label(features, "lbp_mean", "неопределенное"),
        "lbp_std": _label(features, "lbp_std", "неопределенный"),
        "lbp_median": _label(features, "lbp_median", "неопределенная"),
        "percent_dark_pixels": _label(features, "percent_dark_pixels", "неопределено"),
        "percent_white_pixels": _label(features, "percent_white_pixels", "неопределено"),
        "percent_red_pixels": _label(features, "percent_red_pixels", "неопределено"),
        "percent_blue_pixels": _label(features, "percent_blue_pixels", "неопределено"),
        "percent_outlier_bright_pixels": _label(features, "percent_outlier_bright_pixels", "неопределено"),
        "percent_outlier_dark_pixels": _label(features, "percent_outlier_dark_pixels", "неопределено"),
    }


def bucket_columns_from_labels(labels: dict[str, str]) -> dict[str, str]:
    return {bucket_column_name(label_name): value for label_name, value in labels.items()}


def extract_raw_features(row_dict: dict[str, Any]) -> dict[str, Any]:
    features: dict[str, Any] = {}
    for key in (*FEATURE_COLUMNS, *COMPOSITE_FEATURE_COLUMNS):
        value = row_dict.get(key)
        if isinstance(value, (int, float)) and not (isinstance(value, float) and math.isnan(value)):
            features[key] = float(value)
        elif isinstance(value, (list, tuple)):
            features[key] = value
    return features


def row_to_labels(row_dict: dict[str, Any]) -> dict[str, str]:
    return bucket_feature_values(extract_raw_features(row_dict))


def row_to_bucket_columns(row_dict: dict[str, Any]) -> dict[str, str]:
    return bucket_columns_from_labels(row_to_labels(row_dict))
