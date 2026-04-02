"""
Псевдо-разметка важности без участия эксперта.

Метод 1 — статистика по датасету: для скалярных признаков, совпадающих с ключами
из importance_config.LABEL_NAMES, считается робастный z-score; чем |z| выше,
тем «важнее» соответствующая метка шага 1 (строка ``ключ:значение``).

Метод 2 — слабый приор по полям diagnosis / structure / properties (ключевые слова
на русском → веса на ключи меток). Суммируется с |z| из метода 1.

Итог: до top_k строк формата, совместимых с importance_dataset / expert_string_to_label_key.
"""

from __future__ import annotations

import ast
import json
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config.config import FEATURE_ROUTING
from config.importance_config import LABEL_NAMES
from analysis.threshold_rules import parse_features_json

# Ключи меток, для которых в features_json есть одноимённый скаляр (робастный z-score).
_NUMERIC_LABEL_KEYS: Tuple[str, ...] = tuple(
    k for k in LABEL_NAMES if k in FEATURE_ROUTING
)

# (подстроки в нижнем регистре) → веса на ключи LABEL_NAMES.
# Подстраивается под типичные формулировки классификатора; при необходимости расширяйте.
_METADATA_RULES: Tuple[Tuple[Tuple[str, ...], Mapping[str, float]], ...] = (
    (("ретикуляр",), {"texture": 1.0, "pigmentation": 0.9, "texture_coarseness": 0.7, "structure_order": 0.6}),
    (("симметрич",), {"asymmetry": 1.2}),
    (("линия", "линии"), {"borders": 0.9, "structure_order": 0.5}),
    (("бесструктур",), {"structure_order": 1.1, "texture": 0.7, "contrast": 0.5}),
    (("невус", "родинк"), {"shape": 0.4, "borders": 0.25, "pigmentation": 0.25}),
)


def _is_scalar_number(x: Any) -> bool:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return False
    if isinstance(x, (bool, np.bool_)):
        return False
    if isinstance(x, (int, np.integer, float, np.floating)):
        return np.isfinite(x)
    return False


def flatten_numeric_feature_frame(df: pd.DataFrame, features_col: str = "features_json") -> pd.DataFrame:
    """Плоская таблица скалярных фичей по строкам датафрейма (только числовые значения)."""
    rows: List[Dict[str, float]] = []
    for _, row in df.iterrows():
        raw = parse_features_json(row.get(features_col))
        clean: Dict[str, float] = {}
        for k, v in raw.items():
            if _is_scalar_number(v):
                clean[k] = float(v)
        rows.append(clean)
    return pd.DataFrame(rows)


def fit_robust_z_params(X: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Медиана и робастный масштаб (1.4826 * MAD) по каждому столбцу."""
    med = X.median(axis=0, numeric_only=True)
    mad = (X - med).abs().median(axis=0, numeric_only=True)
    scale = 1.4826 * mad + 1e-9
    return med, scale


def row_abs_robust_z(
    feat_row: Mapping[str, Any],
    med: pd.Series,
    scale: pd.Series,
    keys: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    """|robust z| для одной строки; отсутствующие ключи → 0.0."""
    want = set(keys) if keys is not None else set(med.index) & set(scale.index)
    out: Dict[str, float] = {}
    for k in want:
        if k not in med.index or k not in scale.index:
            continue
        v = feat_row.get(k)
        if not _is_scalar_number(v):
            out[k] = 0.0
            continue
        z = (float(v) - float(med[k])) / float(scale[k])
        out[k] = abs(z)
    return out


def _meta_field_to_str(x: Any) -> str:
    """NaN / None из pandas и прочие не-строки → пустая строка (для join и .lower())."""
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(x, str):
        return x
    return str(x)


def metadata_label_key_boost(
    diagnosis: Any,
    structure: Any,
    properties: Any,
    rules: Sequence[Tuple[Tuple[str, ...], Mapping[str, float]]] = _METADATA_RULES,
) -> Dict[str, float]:
    """
    Метод 2: агрегированные веса по ключам меток из текстов клиники.
    Подстроки ищутся в объединённом тексте (lower).
    """
    parts = [
        _meta_field_to_str(diagnosis),
        _meta_field_to_str(structure),
        _meta_field_to_str(properties),
    ]
    text = " ".join(parts).lower()
    boost: Dict[str, float] = {}
    for keywords, weights in rules:
        if not any(kw in text for kw in keywords):
            continue
        for lk, w in weights.items():
            boost[lk] = boost.get(lk, 0.0) + float(w)
    return boost


def _candidate_tags(labels_dict: Mapping[str, Any]) -> List[Tuple[str, str]]:
    """Пары (ключ, полная строка key:value) для всех скалярных строковых меток из LABEL_NAMES."""
    out: List[Tuple[str, str]] = []
    for k in LABEL_NAMES:
        v = labels_dict.get(k)
        if not isinstance(v, str) or not v.strip():
            continue
        out.append((k, f"{k}:{v.strip()}"))
    return out


def pseudo_important_labels_statistical(
    labels_dict: Mapping[str, Any],
    z_abs: Mapping[str, float],
    top_k: int = 10,
) -> List[str]:
    """
    Метод 1: ранжирование по |z| для ключей из _NUMERIC_LABEL_KEYS; прочие метки с score 0
    (при равенстве — лексикографически по строке тега).
    """
    scored: List[Tuple[float, str]] = []
    for k, tag in _candidate_tags(labels_dict):
        base = z_abs.get(k, 0.0) if k in _NUMERIC_LABEL_KEYS else 0.0
        scored.append((base, tag))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [t for _, t in scored[:top_k]]


def pseudo_important_labels_combined(
    labels_dict: Mapping[str, Any],
    z_abs: Mapping[str, float],
    meta_boost: Mapping[str, float],
    top_k: int = 10,
    z_weight: float = 1.0,
    prior_weight: float = 1.0,
) -> List[str]:
    """
    Метод 1 + 2: score = z_weight * |z_k| + prior_weight * meta_boost[k]
    (для составных меток без скаляра k только приор).
    """
    scored: List[Tuple[float, str]] = []
    for k, tag in _candidate_tags(labels_dict):
        z_part = z_abs.get(k, 0.0) if k in _NUMERIC_LABEL_KEYS else 0.0
        p_part = float(meta_boost.get(k, 0.0))
        score = z_weight * z_part + prior_weight * p_part
        scored.append((score, tag))
    scored.sort(key=lambda x: (-x[0], x[1]))
    out: List[str] = []
    seen = set()
    for s, tag in scored:
        if tag in seen:
            continue
        seen.add(tag)
        out.append(tag)
        if len(out) >= top_k:
            break
    return out


def add_pseudo_important_labels(
    df: pd.DataFrame,
    labels_step1_col: str = "labels_step1",
    features_col: str = "features_json",
    diagnosis_col: str = "diagnosis",
    structure_col: str = "structure",
    properties_col: str = "properties",
    out_col: str = "pseudo_important_labels",
    top_k: int = 10,
    mode: str = "combined",
    z_weight: float = 1.0,
    prior_weight: float = 1.0,
) -> pd.DataFrame:
    """
    Добавляет колонку ``out_col`` — JSON-массив строк (как important_labels).

    mode: ``"statistical"`` | ``"combined"``.
    """
    if mode not in ("statistical", "combined"):
        raise ValueError('mode must be "statistical" or "combined"')

    X = flatten_numeric_feature_frame(df, features_col=features_col)
    med, scale = fit_robust_z_params(X)

    def _parse_labels(val: Any) -> Dict[str, Any]:
        if isinstance(val, dict):
            return val
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return {}
        if isinstance(val, str):
            try:
                return ast.literal_eval(val)
            except Exception:
                try:
                    return json.loads(val)
                except Exception:
                    return {}
        return {}

    tags: List[str] = []
    for pos in range(len(df)):
        row = df.iloc[pos]
        labels_dict = _parse_labels(row.get(labels_step1_col))
        feat_row = X.iloc[pos]
        z_abs = row_abs_robust_z(
            feat_row.to_dict(),
            med,
            scale,
            keys=_NUMERIC_LABEL_KEYS,
        )
        if mode == "statistical":
            lst = pseudo_important_labels_statistical(labels_dict, z_abs, top_k=top_k)
        else:
            boost = metadata_label_key_boost(
                row.get(diagnosis_col),
                row.get(structure_col),
                row.get(properties_col),
            )
            lst = pseudo_important_labels_combined(
                labels_dict,
                z_abs,
                boost,
                top_k=top_k,
                z_weight=z_weight,
                prior_weight=prior_weight,
            )
        tags.append(json.dumps(lst, ensure_ascii=False))

    out = df.copy()
    out[out_col] = tags
    return out


def example_nevus_profile_boost() -> Dict[str, float]:
    """Демонстрация приора для указанного профиля (можно вызвать вручную)."""
    return metadata_label_key_boost(
        diagnosis="Невус (родинка)",
        structure="Линии, Бесструктурная область",
        properties="Ретикулярные; Симметричные",
    )
