"""Шаг 1: Числа → метки. Пороговые правила (ABCD, CIEDE2000, GLCM)."""

import json
import math
from typing import Optional

import pandas as pd

from config.threshold_config import THRESHOLDS, SCALAR, apply_threshold


def _get_num(d: dict, key: str) -> Optional[float]:
    """Извлечь число из значения: число, (val, desc) или [name, val, unit]."""
    v = d.get(key)
    if v is None:
        return None
    if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
        return float(v)
    if isinstance(v, (list, tuple)) and len(v) >= 2:
        x = v[1]
        if isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x)):
            return float(x)
    if isinstance(v, (list, tuple)) and len(v) >= 1:
        x = v[0]
        if isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x)):
            return float(x)
    return None


def compute_palette_variety(dominant_colors_lesion) -> int:
    """Число устойчивых тонов из dominant_colors_lesion (список HSV-троек)."""
    if dominant_colors_lesion is None:
        return 1
    if not isinstance(dominant_colors_lesion, (list, tuple)):
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


def number_to_label_asymmetry(features: dict) -> str:
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
    agg = sum(abs_vals)
    return apply_threshold(agg, THRESHOLDS["asymmetry_agg"], "слабая")


def number_to_label_borders(features: dict) -> str:
    """Границы по circularity, perimeter_area_ratio."""
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


def number_to_label_contrast(features: dict) -> str:
    """Контраст очаг↔кожа по ΔE2000 (CIEDE2000)."""
    de = _get_num(features, "color_distance_deltaE")
    return apply_threshold(de, THRESHOLDS["color_distance_deltaE"], "умеренный")


def number_to_label_palette(features: dict) -> str:
    """Палитра по palette_variety."""
    dom = features.get("dominant_colors_lesion")
    pv = compute_palette_variety(dom)
    return apply_threshold(float(pv), THRESHOLDS["palette_variety"], "монотонная")


def number_to_label_center_periphery(features: dict) -> dict:
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


def number_to_label_texture(features: dict) -> str:
    """Текстура по glcm_homogeneity."""
    hom = _get_num(features, "glcm_homogeneity")
    return apply_threshold(hom, THRESHOLDS["glcm_homogeneity"], "умеренно неоднородная")


def number_to_label_pigmentation(features: dict) -> str:
    """Пигментация: по долям тёмных, красных, синих, белых пикселей."""
    dark = _get_num(features, "percent_dark_pixels") or 0
    red = _get_num(features, "percent_red_pixels") or 0
    blue = _get_num(features, "percent_blue_pixels") or 0
    white = _get_num(features, "percent_white_pixels") or 0
    w, rb, dh, dm = (
        SCALAR["pigmentation_white"],
        SCALAR["pigmentation_red_blue"],
        SCALAR["pigmentation_dark_high"],
        SCALAR["pigmentation_dark_mid"],
    )
    if white > w:
        return "с выраженными обесцвеченными участками"
    if red > rb or blue > rb:
        return "с примесью красного или синего"
    if dark > dh:
        return "преимущественно тёмная"
    if dark > dm:
        return "умеренно пигментированная"
    return "слабо пигментированная"


def number_to_label_elongation(features: dict) -> str:
    """Вытянутость формы: aspect_ratio, eccentricity."""
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


def number_to_label_rim(features: dict) -> str:
    """Ободок: delta_V_inner_rim (светлый/тёмный ободок по краю)."""
    dv = _get_num(features, "delta_V_inner_rim")
    th = SCALAR["rim_abs"]
    if dv is None or abs(dv) < th:
        return "без выраженного ободка"
    if dv > 0:
        return "светлый ободок по краю"
    return "тёмный ободок по краю"


def number_to_label_color_homogeneity(features: dict) -> str:
    """Однородность цвета: std_H/S/V, entropy."""
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


def number_to_label_texture_coarseness(features: dict) -> str:
    """Грубость текстуры: glcm_contrast (высокий = грубая, пятнистая)."""
    c = _get_num(features, "glcm_contrast")
    return apply_threshold(c, THRESHOLDS["glcm_contrast"], "средняя")


def number_to_label_shape(features: dict) -> str:
    """Форма по circularity."""
    circ = _get_num(features, "circularity")
    return apply_threshold(circ, THRESHOLDS["circularity"], "неопределенная")


def number_to_label_dominant_hue(features: dict) -> str:
    """Преобладающий оттенок по mean_H и color_balance (H в HSV 0–180 в OpenCV)."""
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


def number_to_label_pigment_inclusions(features: dict) -> list:
    """Список цветовых включений (красные/синие/белые участки)."""
    th = SCALAR["pigment_inclusion_threshold"]
    inclusions = []
    red = _get_num(features, "percent_red_pixels") or 0
    if red > th:
        inclusions.append("красные вкрапления")
    blue = _get_num(features, "percent_blue_pixels") or 0
    if blue > th:
        inclusions.append("синеватые участки")
    white = _get_num(features, "percent_white_pixels") or 0
    if white > th:
        inclusions.append("обесцвеченные участки")
    return inclusions


def number_to_label_structure_order(features: dict) -> str:
    """Упорядоченность структуры по lbp_uniformity."""
    uniform = _get_num(features, "lbp_uniformity")
    return apply_threshold(uniform, THRESHOLDS["lbp_uniformity"], "неопределенная")


def number_to_label_lobulation(features: dict) -> str:
    """Лобуляции: суррогат из radial_variance, convexity, perimeter_area_ratio."""
    rvar = _get_num(features, "radial_variance")
    conv = _get_num(features, "convexity")
    para = _get_num(features, "perimeter_area_ratio")
    rn, pb, ps = (
        SCALAR["lobulation_radial_norm"],
        SCALAR["lobulation_para_base"],
        SCALAR["lobulation_para_span"],
    )
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


def _label_by_feature(
    features: dict,
    feature_key: str,
    default: str = "неопределено",
) -> str:
    """Универсальное преобразование одного признака в метку по конфигу THRESHOLDS."""
    val = _get_num(features, feature_key)
    if feature_key not in THRESHOLDS:
        return default
    return apply_threshold(val, THRESHOLDS[feature_key], default) or default


# ——— Функции для признаков, ранее не охваченных составными правилами ———

def number_to_label_area(features: dict) -> str:
    """Размер очага по площади маски."""
    return _label_by_feature(features, "area", "неопределенный")


def number_to_label_perimeter(features: dict) -> str:
    """Периметр контура."""
    return _label_by_feature(features, "perimeter", "неопределенный")


def number_to_label_convexity(features: dict) -> str:
    """Выпуклость контура."""
    return _label_by_feature(features, "convexity", "неопределенная")


def number_to_label_solidity(features: dict) -> str:
    """Плотность к выпуклой оболочке."""
    return _label_by_feature(features, "solidity", "неопределенная")


def number_to_label_extent(features: dict) -> str:
    """Заполнение bbox."""
    return _label_by_feature(features, "extent", "неопределенное")


def number_to_label_radial_variance(features: dict) -> str:
    """Вариация радиуса границы."""
    return _label_by_feature(features, "radial_variance", "неопределенная")


def number_to_label_fractal_dimension(features: dict) -> str:
    """Фрактальная размерность границы."""
    return _label_by_feature(features, "fractal_dimension", "неопределенная")


def number_to_label_eccentricity(features: dict) -> str:
    """Эксцентриситет формы."""
    return _label_by_feature(features, "eccentricity", "неопределенная")


def number_to_label_perimeter_area_ratio(features: dict) -> str:
    """Отношение периметр/площадь."""
    return _label_by_feature(features, "perimeter_area_ratio", "неопределенное")


def number_to_label_color_balance_R(features: dict) -> str:
    """Доля красного в среднем цвете."""
    return _label_by_feature(features, "color_balance_R", "неопределено")


def number_to_label_color_balance_G(features: dict) -> str:
    """Доля зелёного в среднем цвете."""
    return _label_by_feature(features, "color_balance_G", "неопределено")


def number_to_label_color_balance_B(features: dict) -> str:
    """Доля синего в среднем цвете."""
    return _label_by_feature(features, "color_balance_B", "неопределено")


def number_to_label_color_distance_euclidean(features: dict) -> str:
    """LAB-контраст очаг–кожа (евклидов)."""
    return _label_by_feature(features, "color_distance_euclidean", "неопределенный")


def number_to_label_delta_H_center_periphery(features: dict) -> str:
    """Разница оттенка центр–периферия."""
    return _label_by_feature(features, "delta_H_center_periphery", "неопределено")


def number_to_label_delta_S_center_periphery(features: dict) -> str:
    """Разница насыщенности центр–периферия."""
    return _label_by_feature(features, "delta_S_center_periphery", "неопределено")


def number_to_label_delta_V_center_periphery(features: dict) -> str:
    """Разница яркости центр–периферия."""
    return _label_by_feature(features, "delta_V_center_periphery", "неопределено")


def number_to_label_delta_V_inner_rim(features: dict) -> str:
    """Разница яркости центр–ободок."""
    return _label_by_feature(features, "delta_V_inner_rim", "неопределено")


def number_to_label_delta_V_left_right(features: dict) -> str:
    """Асимметрия яркости лево–право."""
    return _label_by_feature(features, "delta_V_left_right", "неопределено")


def number_to_label_delta_V_top_bottom(features: dict) -> str:
    """Асимметрия яркости верх–низ."""
    return _label_by_feature(features, "delta_V_top_bottom", "неопределено")


def number_to_label_delta_S_left_right(features: dict) -> str:
    """Асимметрия насыщенности лево–право."""
    return _label_by_feature(features, "delta_S_left_right", "неопределено")


def number_to_label_delta_S_top_bottom(features: dict) -> str:
    """Асимметрия насыщенности верх–низ."""
    return _label_by_feature(features, "delta_S_top_bottom", "неопределено")


def number_to_label_entropy_H_lesion(features: dict) -> str:
    """Разнообразие оттенка в очаге."""
    return _label_by_feature(features, "entropy_H_lesion", "неопределенное")


def number_to_label_entropy_S_lesion(features: dict) -> str:
    """Разнообразие насыщенности в очаге."""
    return _label_by_feature(features, "entropy_S_lesion", "неопределенное")


def number_to_label_entropy_V_lesion(features: dict) -> str:
    """Разнообразие яркости в очаге."""
    return _label_by_feature(features, "entropy_V_lesion", "неопределенное")


def number_to_label_std_H_lesion(features: dict) -> str:
    """Вариабельность оттенка в очаге."""
    return _label_by_feature(features, "std_H_lesion", "неопределенная")


def number_to_label_std_S_lesion(features: dict) -> str:
    """Вариабельность насыщенности в очаге."""
    return _label_by_feature(features, "std_S_lesion", "неопределенная")


def number_to_label_std_V_lesion(features: dict) -> str:
    """Вариабельность яркости в очаге."""
    return _label_by_feature(features, "std_V_lesion", "неопределенная")


def number_to_label_mean_S_lesion(features: dict) -> str:
    """Средняя насыщенность очага."""
    return _label_by_feature(features, "mean_S_lesion", "неопределенная")


def number_to_label_mean_V_lesion(features: dict) -> str:
    """Средняя яркость очага."""
    return _label_by_feature(features, "mean_V_lesion", "неопределенная")


def number_to_label_glcm_energy(features: dict) -> str:
    """GLCM-энергия текстуры."""
    return _label_by_feature(features, "glcm_energy", "неопределенная")


def number_to_label_glcm_entropy(features: dict) -> str:
    """GLCM-энтропия текстуры."""
    return _label_by_feature(features, "glcm_entropy", "неопределенная")


def number_to_label_lbp_entropy(features: dict) -> str:
    """LBP-энтропия."""
    return _label_by_feature(features, "lbp_entropy", "неопределенная")


def number_to_label_lbp_mean(features: dict) -> str:
    """LBP-среднее."""
    return _label_by_feature(features, "lbp_mean", "неопределенное")


def number_to_label_lbp_std(features: dict) -> str:
    """LBP-разброс."""
    return _label_by_feature(features, "lbp_std", "неопределенный")


def number_to_label_lbp_median(features: dict) -> str:
    """LBP-медиана."""
    return _label_by_feature(features, "lbp_median", "неопределенная")


def number_to_label_percent_dark_pixels(features: dict) -> str:
    """Доля тёмных пикселей."""
    return _label_by_feature(features, "percent_dark_pixels", "неопределено")


def number_to_label_percent_white_pixels(features: dict) -> str:
    """Доля белых пикселей."""
    return _label_by_feature(features, "percent_white_pixels", "неопределено")


def number_to_label_percent_red_pixels(features: dict) -> str:
    """Доля красных пикселей."""
    return _label_by_feature(features, "percent_red_pixels", "неопределено")


def number_to_label_percent_blue_pixels(features: dict) -> str:
    """Доля синих пикселей."""
    return _label_by_feature(features, "percent_blue_pixels", "неопределено")


def number_to_label_percent_outlier_bright_pixels(features: dict) -> str:
    """Доля аномально ярких пикселей."""
    return _label_by_feature(features, "percent_outlier_bright_pixels", "неопределено")


def number_to_label_percent_outlier_dark_pixels(features: dict) -> str:
    """Доля аномально тёмных пикселей."""
    return _label_by_feature(features, "percent_outlier_dark_pixels", "неопределено")


def features_to_labels(features: dict) -> dict:
    """Преобразует словарь признаков в словарь категориальных меток."""
    labels = {
        "asymmetry": number_to_label_asymmetry(features),
        "borders": number_to_label_borders(features),
        "contrast": number_to_label_contrast(features),
        "palette": number_to_label_palette(features),
        "texture": number_to_label_texture(features),
        "lobulation": number_to_label_lobulation(features),
        "pigmentation": number_to_label_pigmentation(features),
        "elongation": number_to_label_elongation(features),
        "rim": number_to_label_rim(features),
        "color_homogeneity": number_to_label_color_homogeneity(features),
        "texture_coarseness": number_to_label_texture_coarseness(features),
        "shape": number_to_label_shape(features),
        "dominant_hue": number_to_label_dominant_hue(features),
        "pigment_inclusions": number_to_label_pigment_inclusions(features),
        "structure_order": number_to_label_structure_order(features),
        # Метки по одному признаку из конфига
        "area": number_to_label_area(features),
        "perimeter": number_to_label_perimeter(features),
        "convexity": number_to_label_convexity(features),
        "solidity": number_to_label_solidity(features),
        "extent": number_to_label_extent(features),
        "radial_variance": number_to_label_radial_variance(features),
        "fractal_dimension": number_to_label_fractal_dimension(features),
        "eccentricity": number_to_label_eccentricity(features),
        "perimeter_area_ratio": number_to_label_perimeter_area_ratio(features),
        "color_balance_R": number_to_label_color_balance_R(features),
        "color_balance_G": number_to_label_color_balance_G(features),
        "color_balance_B": number_to_label_color_balance_B(features),
        "color_distance_euclidean": number_to_label_color_distance_euclidean(features),
        "delta_H_center_periphery": number_to_label_delta_H_center_periphery(features),
        "delta_S_center_periphery": number_to_label_delta_S_center_periphery(features),
        "delta_V_center_periphery": number_to_label_delta_V_center_periphery(features),
        "delta_V_inner_rim": number_to_label_delta_V_inner_rim(features),
        "delta_V_left_right": number_to_label_delta_V_left_right(features),
        "delta_V_top_bottom": number_to_label_delta_V_top_bottom(features),
        "delta_S_left_right": number_to_label_delta_S_left_right(features),
        "delta_S_top_bottom": number_to_label_delta_S_top_bottom(features),
        "entropy_H_lesion": number_to_label_entropy_H_lesion(features),
        "entropy_S_lesion": number_to_label_entropy_S_lesion(features),
        "entropy_V_lesion": number_to_label_entropy_V_lesion(features),
        "std_H_lesion": number_to_label_std_H_lesion(features),
        "std_S_lesion": number_to_label_std_S_lesion(features),
        "std_V_lesion": number_to_label_std_V_lesion(features),
        "mean_S_lesion": number_to_label_mean_S_lesion(features),
        "mean_V_lesion": number_to_label_mean_V_lesion(features),
        "glcm_energy": number_to_label_glcm_energy(features),
        "glcm_entropy": number_to_label_glcm_entropy(features),
        "lbp_entropy": number_to_label_lbp_entropy(features),
        "lbp_mean": number_to_label_lbp_mean(features),
        "lbp_std": number_to_label_lbp_std(features),
        "lbp_median": number_to_label_lbp_median(features),
        "percent_dark_pixels": number_to_label_percent_dark_pixels(features),
        "percent_white_pixels": number_to_label_percent_white_pixels(features),
        "percent_red_pixels": number_to_label_percent_red_pixels(features),
        "percent_blue_pixels": number_to_label_percent_blue_pixels(features),
        "percent_outlier_bright_pixels": number_to_label_percent_outlier_bright_pixels(features),
        "percent_outlier_dark_pixels": number_to_label_percent_outlier_dark_pixels(features),
    }
    labels["center_periphery"] = number_to_label_center_periphery(features)
    return labels


def parse_features_json(features_json) -> dict:
    """Парсит features_json (str или dict) в плоский словарь feature_name -> value."""
    if isinstance(features_json, str):
        try:
            features_json = json.loads(features_json)
        except Exception:
            return {}
    out = {}
    blocks = [
        features_json.get("color", {}).get("local", {}),
        features_json.get("color", {}).get("global", {}),
        features_json.get("shape", {}),
        features_json.get("border", {}),
        features_json.get("texture", {}),
    ]
    for block in blocks:
        if not isinstance(block, dict):
            continue
        for k, v in block.items():
            val = v[1] if isinstance(v, (list, tuple)) and len(v) >= 2 else v
            out[k] = val
    return out


def row_to_labels(row) -> dict:
    """Для строки df с колонкой features_json возвращает словарь меток."""
    raw = row.get("features_json", {})
    if pd.isna(raw) or raw is None or raw == "":
        return features_to_labels({})
    feats = parse_features_json(raw)
    return features_to_labels(feats)

