"""Composite numeric features derived from extracted image features."""

import math
from typing import Any, Optional

from config.threshold_config import SCALAR

HUE_PERIOD = 180.0
COMPOSITE_FEATURE_COLUMNS = (
    "asymmetry_agg",
    "palette_variety",
    "elongation_score",
    "color_homogeneity_agg",
    "lobulation_score",
)


def _get_num(features: dict[str, Any], key: str) -> Optional[float]:
    value = features.get(key)
    if value is None:
        return None
    if isinstance(value, (int, float)) and not (isinstance(value, float) and math.isnan(value)):
        return float(value)
    return None


def compute_palette_variety(dominant_colors_lesion: Any) -> int:
    if dominant_colors_lesion is None or not isinstance(dominant_colors_lesion, (list, tuple)):
        return 1
    colors = list(dominant_colors_lesion)
    if not colors:
        return 1

    hues = []
    for color in colors:
        if isinstance(color, (list, tuple)) and len(color) >= 1:
            hues.append(float(color[0]) % HUE_PERIOD)
        elif isinstance(color, (int, float)):
            hues.append(float(color) % HUE_PERIOD)

    if len(hues) < 2:
        return min(len(colors), 4)

    hues = sorted(hues)
    clusters = 1
    for idx in range(1, len(hues)):
        if abs(hues[idx] - hues[idx - 1]) > 15 and abs(hues[idx] - hues[0]) > 15:
            clusters += 1
    return min(clusters, 6)


def compute_asymmetry_score(features: dict[str, Any]) -> float:
    deltas = [
        _get_num(features, "delta_H_center_periphery"),
        _get_num(features, "delta_S_center_periphery"),
        _get_num(features, "delta_V_center_periphery"),
        _get_num(features, "delta_V_left_right"),
        _get_num(features, "delta_S_left_right"),
        _get_num(features, "delta_V_top_bottom"),
        _get_num(features, "delta_S_top_bottom"),
    ]
    return float(sum(abs(value) for value in deltas if value is not None))


def compute_elongation_score(features: dict[str, Any]) -> float:
    aspect_ratio = _get_num(features, "aspect_ratio")
    eccentricity = _get_num(features, "eccentricity")

    score = 0.0
    if aspect_ratio is not None:
        score += max(0.0, (aspect_ratio - 1.0) * 0.5)
    if eccentricity is not None:
        score += eccentricity * 0.5
    return float(score)


def compute_color_homogeneity_score(features: dict[str, Any]) -> float:
    std_h = _get_num(features, "std_H_lesion") or 0.0
    std_s = _get_num(features, "std_S_lesion") or 0.0
    std_v = _get_num(features, "std_V_lesion") or 0.0
    entropy_h = _get_num(features, "entropy_H_lesion") or 0.0
    return float(std_h / 30.0 + std_s / 100.0 + std_v / 100.0 + entropy_h / 4.0)


def compute_lobulation_score(features: dict[str, Any]) -> float:
    radial_variance = _get_num(features, "radial_variance")
    convexity = _get_num(features, "convexity")
    perimeter_area_ratio = _get_num(features, "perimeter_area_ratio")

    score = 0.0
    if radial_variance is not None:
        score += min(radial_variance / SCALAR["lobulation_radial_norm"], 1.0) * 0.4
    if convexity is not None and convexity > 0:
        score += (1.0 - convexity) * 0.3
    if perimeter_area_ratio is not None:
        base = SCALAR["lobulation_para_base"]
        span = SCALAR["lobulation_para_span"]
        score += max(0.0, (perimeter_area_ratio - base) / span) * 0.3
    return float(score)


def compute_composite_features(features: dict[str, Any]) -> dict[str, float]:
    return {
        "asymmetry_agg": compute_asymmetry_score(features),
        "palette_variety": float(compute_palette_variety(features.get("dominant_colors_lesion"))),
        "elongation_score": compute_elongation_score(features),
        "color_homogeneity_agg": compute_color_homogeneity_score(features),
        "lobulation_score": compute_lobulation_score(features),
    }


def merge_with_composite_features(features: dict[str, Any]) -> dict[str, Any]:
    merged = dict(features)
    merged.update(compute_composite_features(merged))
    return merged
