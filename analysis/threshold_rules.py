"""Числа → метки. Пороговые правила (ABCD, CIEDE2000, GLCM)."""

import math
from typing import Optional

from config.config import FEATURE_ROUTING
from config.threshold_config import THRESHOLDS, SCALAR, apply_threshold


def _get_num(d: dict, key: str) -> Optional[float]:
    """Извлечь число из словаря."""
    v = d.get(key)
    if v is None:
        return None
    if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
        return float(v)
    return None


def _label(features: dict, key: str, default: str = "неопределено") -> str:
    """Один признак → одна метка по THRESHOLDS."""
    if key not in THRESHOLDS:
        return default
    return apply_threshold(_get_num(features, key), THRESHOLDS[key], default) or default


# ——— Составные правила (несколько признаков → одна метка) ———

def _asymmetry(features: dict) -> str:
    """Асимметрия по градиентам H/S/V (ABCD rule)."""
    deltas = [
        _get_num(features, "delta_H_center_periphery"),
        _get_num(features, "delta_S_center_periphery"),
        _get_num(features, "delta_V_center_periphery"),
        _get_num(features, "delta_V_left_right"),
        _get_num(features, "delta_S_left_right"),
        _get_num(features, "delta_V_top_bottom"),
        _get_num(features, "delta_S_top_bottom"),
    ]
    abs_vals = [abs(x) for x in deltas if x is not None]
    if not abs_vals:
        return "слабая"
    return apply_threshold(sum(abs_vals), THRESHOLDS["asymmetry_agg"], "слабая")


def _borders(features: dict) -> str:
    """Границы по circularity + perimeter_area_ratio."""
    circ = _get_num(features, "circularity")
    para = _get_num(features, "perimeter_area_ratio")
    ch, ph = SCALAR["borders_circ_high"], SCALAR["borders_para_high"]
    cm, pm = SCALAR["borders_circ_mid"], SCALAR["borders_para_mid"]
    if circ is not None and para is not None:
        if circ >= ch and para < ph:
            return "ровные"
        if circ >= cm and para < pm:
            return "умеренно неровные"
    if circ is not None:
        if circ >= ch:
            return "ровные"
        if circ >= cm:
            return "умеренно неровные"
    return "фестончатые"


def _palette(features: dict) -> str:
    """Палитра по dominant_colors_lesion → palette_variety."""
    dom = features.get("dominant_colors_lesion")
    pv = _compute_palette_variety(dom)
    return apply_threshold(float(pv), THRESHOLDS["palette_variety"], "монотонная")


def _compute_palette_variety(dominant_colors_lesion) -> int:
    if dominant_colors_lesion is None or not isinstance(dominant_colors_lesion, (list, tuple)):
        return 1
    colors = list(dominant_colors_lesion)
    if not colors:
        return 1
    hues = []
    for c in colors:
        if isinstance(c, (list, tuple)) and len(c) >= 1:
            hues.append(c[0] % 180)
        elif isinstance(c, (int, float)):
            hues.append(float(c) % 180)
    if len(hues) < 2:
        return min(len(colors), 4)
    hues = sorted(hues)
    n = 1
    for i in range(1, len(hues)):
        if abs(hues[i] - hues[i - 1]) > 15 and abs(hues[i] - hues[0]) > 15:
            n += 1
    return min(n, 6)


def _center_periphery(features: dict) -> dict:
    """Центр/периферия: delta_V и delta_S."""
    out = {}
    dv = _get_num(features, "delta_V_center_periphery")
    ds = _get_num(features, "delta_S_center_periphery")
    th_v = SCALAR["center_periphery_V_abs"]
    th_s = SCALAR["center_periphery_S_abs"]
    if dv is not None and abs(dv) > th_v:
        out["center_darker"] = "центр темнее периферии" if dv < 0 else "центр светлее периферии"
    if ds is not None and abs(ds) > th_s:
        out["saturation_center"] = "насыщенность выше в центре" if ds > 0 else "насыщенность ниже в центре"
    return out


def _pigmentation(features: dict) -> str:
    """Пигментация по долям тёмных, красных, синих, белых пикселей."""
    dark = _get_num(features, "percent_dark_pixels") or 0
    red = _get_num(features, "percent_red_pixels") or 0
    blue = _get_num(features, "percent_blue_pixels") or 0
    white = _get_num(features, "percent_white_pixels") or 0
    w, rb = SCALAR["pigmentation_white"], SCALAR["pigmentation_red_blue"]
    dh, dm = SCALAR["pigmentation_dark_high"], SCALAR["pigmentation_dark_mid"]
    if white > w:
        return "с выраженными обесцвеченными участками"
    if red > rb or blue > rb:
        return "с примесью красного или синего"
    if dark > dh:
        return "преимущественно тёмная"
    if dark > dm:
        return "умеренно пигментированная"
    return "слабо пигментированная"


def _elongation(features: dict) -> str:
    """Вытянутость: aspect_ratio + eccentricity."""
    ar = _get_num(features, "aspect_ratio")
    ecc = _get_num(features, "eccentricity")
    if ar is None and ecc is None:
        return "округлая"
    score = 0.0
    if ar is not None:
        score += max(0, (ar - 1) * 0.5)
    if ecc is not None:
        score += ecc * 0.5
    low, mid = SCALAR["elongation_score_low"], SCALAR["elongation_score_mid"]
    if score < low:
        return "округлая"
    if score < mid:
        return "умеренно вытянутая"
    return "вытянутая"


def _rim(features: dict) -> str:
    """Ободок: delta_V_inner_rim."""
    dv = _get_num(features, "delta_V_inner_rim")
    th = SCALAR["rim_abs"]
    if dv is None or abs(dv) < th:
        return "без выраженного ободка"
    return "светлый ободок по краю" if dv > 0 else "тёмный ободок по краю"


def _color_homogeneity(features: dict) -> str:
    """Однородность цвета: std_H/S/V + entropy."""
    std_h = _get_num(features, "std_H_lesion") or 0
    std_s = _get_num(features, "std_S_lesion") or 0
    std_v = _get_num(features, "std_V_lesion") or 0
    ent_h = _get_num(features, "entropy_H_lesion") or 0
    agg = std_h / 30 + std_s / 100 + std_v / 100 + ent_h / 4
    high, mid = SCALAR["color_homogeneity_agg_high"], SCALAR["color_homogeneity_agg_mid"]
    if agg < high:
        return "цвет однородный"
    if agg < mid:
        return "цвет умеренно неоднородный"
    return "цвет неоднородный"


def _dominant_hue(features: dict) -> str:
    """Преобладающий оттенок по mean_H + color_balance."""
    mean_h = _get_num(features, "mean_H_lesion")
    if mean_h is None:
        return "неопределенный"
    base = apply_threshold(mean_h, THRESHOLDS["mean_H_lesion"], "неопределенный")
    if base == "коричневый":
        r = _get_num(features, "color_balance_R") or 0
        g = _get_num(features, "color_balance_G") or 0
        if g > r * 1.2:
            return "зеленоватый"
    return base or "неопределенный"


def _pigment_inclusions(features: dict) -> list:
    """Цветовые включения (красные/синие/белые участки)."""
    th = SCALAR["pigment_inclusion_threshold"]
    inclusions = []
    if (_get_num(features, "percent_red_pixels") or 0) > th:
        inclusions.append("красные вкрапления")
    if (_get_num(features, "percent_blue_pixels") or 0) > th:
        inclusions.append("синеватые участки")
    if (_get_num(features, "percent_white_pixels") or 0) > th:
        inclusions.append("обесцвеченные участки")
    return inclusions


def _lobulation(features: dict) -> str:
    """Лобуляции: radial_variance + convexity + perimeter_area_ratio."""
    rvar = _get_num(features, "radial_variance")
    conv = _get_num(features, "convexity")
    para = _get_num(features, "perimeter_area_ratio")
    rn = SCALAR["lobulation_radial_norm"]
    pb, ps = SCALAR["lobulation_para_base"], SCALAR["lobulation_para_span"]
    score = 0.0
    if rvar is not None:
        score += min(rvar / rn, 1.0) * 0.4
    if conv is not None and conv > 0:
        score += (1 - conv) * 0.3
    if para is not None:
        score += max(0, (para - pb) / ps) * 0.3
    low, mid = SCALAR["lobulation_score_low"], SCALAR["lobulation_score_mid"]
    if score < low:
        return "лобуляции отсутствуют"
    if score < mid:
        return "лобуляции слабые"
    return "лобуляции выраженные"


# ——— Агрегатор ———

def features_to_labels(features: dict) -> dict:
    """Преобразует словарь признаков в словарь категориальных меток."""
    labels = {
        # Составные правила
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
        "pigment_inclusions": _pigment_inclusions(features),
        "structure_order": _label(features, "lbp_uniformity", "неопределенная"),
        "center_periphery": _center_periphery(features),
        # Простые признаки — прямой вызов _label
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
    return labels


def row_to_labels(row_dict: dict) -> dict:
    """Для словаря строки df возвращает словарь меток."""
    features = {}
    for key in FEATURE_ROUTING:
        value = row_dict.get(key)
        if isinstance(value, (int, float)) and not (isinstance(value, float) and math.isnan(value)):
            features[key] = float(value)
        elif isinstance(value, (list, tuple)):
            features[key] = value
    return features_to_labels(features)
