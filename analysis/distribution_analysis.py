"""
Анализ распределений числовых признаков и оценка текущих порогов.

Использование:
    python -m analysis.distribution_analysis --csv features_dataset_bucket.csv
    python -m analysis.distribution_analysis --csv features.csv --output analysis/results

Результаты:
    - analysis/results/distribution_report.json  — статистика по каждому признаку
    - analysis/results/label_distribution.json    — распределение меток по бакетам
    - analysis/results/plots/                     — гистограммы с порогами
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.config import FEATURE_ROUTING
from config.threshold_config import THRESHOLDS, SCALAR, INF

# Признаки, для которы�� пороги задаются через THRESHOLDS (интервальные)
THRESHOLD_FEATURES = set(THRESHOLDS.keys()) - {"asymmetry_agg", "palette_variety"}

DEFAULT_OUTPUT = Path("analysis/results")


# ——————————————————————————————————————————————
#  Статистика по одному признаку
# ——————————————————————————————————————————————

def _feature_stats(values: np.ndarray) -> dict:
    """Базовая описательная статистика для массива значений."""
    if len(values) == 0:
        return {"count": 0}
    return {
        "count": int(len(values)),
        "mean": round(float(np.mean(values)), 4),
        "std": round(float(np.std(values)), 4),
        "min": round(float(np.min(values)), 4),
        "p5": round(float(np.percentile(values, 5)), 4),
        "p25": round(float(np.percentile(values, 25)), 4),
        "median": round(float(np.percentile(values, 50)), 4),
        "p75": round(float(np.percentile(values, 75)), 4),
        "p95": round(float(np.percentile(values, 95)), 4),
        "max": round(float(np.max(values)), 4),
        "iqr": round(float(np.percentile(values, 75) - np.percentile(values, 25)), 4),
    }


def _bucket_counts(values: np.ndarray, thresholds: list) -> dict:
    """Считает сколько значений попадает в каждый бакет порогов."""
    counts = {}
    for low, high, label in thresholds:
        h = high if high != INF else float("inf")
        l = low if low != -INF else float("-inf")
        n = int(np.sum((values >= l) & (values < h)))
        counts[label] = {
            "range": f"[{low}, {high})",
            "count": n,
            "percent": round(n / len(values) * 100, 1) if len(values) > 0 else 0,
        }
    return counts


def _suggested_thresholds(values: np.ndarray, n_buckets: int = 3) -> list:
    """Пороги на основе квантилей (для сравнения с текущими)."""
    if len(values) < n_buckets:
        return []
    quantiles = np.linspace(0, 100, n_buckets + 1)[1:-1]
    return [round(float(np.percentile(values, q)), 4) for q in quantiles]


# ——————————————————————————————————————————————
#  Основной анализ
# ——————————————————————————————————————————————

def analyze_distributions(df: pd.DataFrame) -> dict:
    """
    Анализирует распределения всех числовых признаков из FEATURE_ROUTING.

    Returns:
        dict с ключами — имена признаков, значения — статистик�� + бакеты.
    """
    report = {}

    for feature_name in FEATURE_ROUTING:
        if feature_name not in df.columns:
            continue

        col = df[feature_name]
        numeric = pd.to_numeric(col, errors="coerce").dropna()
        values = numeric.values

        if len(values) == 0:
            continue

        entry: dict[str, Any] = {
            "category": FEATURE_ROUTING[feature_name][0],
            "description": FEATURE_ROUTING[feature_name][1],
            "unit": FEATURE_ROUTING[feature_name][2],
            "stats": _feature_stats(values),
        }

        # Если есть пороги — считаем распределение по бакетам
        if feature_name in THRESHOLDS:
            th = THRESHOLDS[feature_name]
            entry["current_thresholds"] = [
                {"low": low, "high": high, "label": label}
                for low, high, label in th
            ]
            entry["bucket_distribution"] = _bucket_counts(values, th)
            entry["quantile_thresholds"] = _suggested_thresholds(values, n_buckets=len(th))

            # Предупрежде��ия
            warnings = []
            for label, info in entry["bucket_distribution"].items():
                if info["count"] == 0:
                    warnings.append(f"бакет '{label}' пустой")
                elif info["percent"] > 90:
                    warnings.append(f"бакет '{label}' содержит {info['percent']}% данных — порог может быть бесполезен")
            if warnings:
                entry["warnings"] = warnings

        report[feature_name] = entry

    return report


def analyze_label_distribution(df: pd.DataFrame) -> dict:
    """
    Анализирует распределение категориальных меток (после bucketing).
    Требует колонку 'labels' (dict) или 'labels_json' (str).
    """
    if "labels" not in df.columns and "labels_json" not in df.columns:
        return {}

    label_stats = {}

    for _, row in df.iterrows():
        if "labels" in df.columns and isinstance(row.get("labels"), dict):
            labels = row["labels"]
        elif "labels_json" in df.columns and isinstance(row.get("labels_json"), str):
            try:
                labels = json.loads(row["labels_json"])
            except (json.JSONDecodeError, TypeError):
                continue
        else:
            continue

        for key, value in labels.items():
            if isinstance(value, dict):
                # center_periphery и подобные
                for sub_key, sub_val in value.items():
                    full_key = f"{key}.{sub_key}"
                    label_stats.setdefault(full_key, {})
                    label_stats[full_key][sub_val] = label_stats[full_key].get(sub_val, 0) + 1
            elif isinstance(value, list):
                # pigment_inclusions
                for item in value:
                    label_stats.setdefault(key, {})
                    label_stats[key][item] = label_stats[key].get(item, 0) + 1
                if not value:
                    label_stats.setdefault(key, {})
                    label_stats[key]["нет"] = label_stats[key].get("нет", 0) + 1
            elif isinstance(value, str):
                label_stats.setdefault(key, {})
                label_stats[key][value] = label_stats[key].get(value, 0) + 1

    total = len(df)
    result = {}
    for key, counts in sorted(label_stats.items()):
        result[key] = {
            val: {"count": cnt, "percent": round(cnt / total * 100, 1)}
            for val, cnt in sorted(counts.items(), key=lambda x: -x[1])
        }

    return result


# ——————————————————————————————————————————————
#  Графики
# ——————————————————————————————————————————————

def plot_distributions(df: pd.DataFrame, output_dir: Path, report: dict = None):
    """
    Строит гистограмму для каждого признака с вертикальными линиями на текущих порогах.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if report is None:
        report = analyze_distributions(df)

    for feature_name, entry in report.items():
        col = df.get(feature_name)
        if col is None:
            continue
        values = pd.to_numeric(col, errors="coerce").dropna().values
        if len(values) < 5:
            continue

        fig, ax = plt.subplots(figsize=(10, 5))

        # Гистограмма
        n_bins = min(80, max(20, len(values) // 10))
        ax.hist(values, bins=n_bins, alpha=0.7, color="#4a90d9", edgecolor="white", linewidth=0.5)

        # Текущие пороги — вертикальные линии
        if feature_name in THRESHOLDS:
            seen = set()
            for low, high, label in THRESHOLDS[feature_name]:
                for boundary in [low, high]:
                    if boundary in seen or abs(boundary) == INF:
                        continue
                    seen.add(boundary)
                    ax.axvline(boundary, color="red", linestyle="--", linewidth=1.5, alpha=0.8)
                    ax.text(
                        boundary, ax.get_ylim()[1] * 0.95, f" {boundary}",
                        color="red", fontsize=8, rotation=90, va="top",
                    )

        # Квантильные пороги — пунктир
        quantile_th = entry.get("quantile_thresholds", [])
        for qt in quantile_th:
            ax.axvline(qt, color="green", linestyle=":", linewidth=1.2, alpha=0.7)

        # Перцентили p5, p95
        stats = entry.get("stats", {})
        for pct_key, pct_color in [("p5", "#888"), ("p95", "#888")]:
            pct_val = stats.get(pct_key)
            if pct_val is not None:
                ax.axvline(pct_val, color=pct_color, linestyle="-.", linewidth=0.8, alpha=0.5)

        desc = entry.get("description", feature_name)
        ax.set_title(f"{feature_name}\n{desc}", fontsize=11)
        ax.set_xlabel(entry.get("unit", ""))
        ax.set_ylabel("count")

        # Легенда
        from matplotlib.lines import Line2D
        legend_items = [
            Line2D([0], [0], color="red", linestyle="--", label="текущие пороги"),
        ]
        if quantile_th:
            legend_items.append(Line2D([0], [0], color="green", linestyle=":", label="квантильные пороги"))
        legend_items.append(Line2D([0], [0], color="#888", linestyle="-.", label="p5 / p95"))
        ax.legend(handles=legend_items, fontsize=8, loc="upper right")

        plt.tight_layout()
        fig.savefig(plots_dir / f"{feature_name}.png", dpi=120)
        plt.close(fig)

    # Сводный график — бакеты с предупреждениями
    warning_features = [f for f, e in report.items() if e.get("warnings")]
    if warning_features:
        _plot_bucket_balance(df, report, warning_features, plots_dir)


def _plot_bucket_balance(
    df: pd.DataFrame, report: dict, features: list, plots_dir: Path,
):
    """Горизонтальная столбчатая диаграмма баланса бакетов для проблемных признаков."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(features)
    fig, axes = plt.subplots(n, 1, figsize=(10, max(3 * n, 5)))
    if n == 1:
        axes = [axes]

    for ax, fname in zip(axes, features):
        buckets = report[fname].get("bucket_distribution", {})
        labels = list(buckets.keys())
        pcts = [buckets[l]["percent"] for l in labels]
        colors = ["#e74c3c" if p > 90 or p == 0 else "#4a90d9" for p in pcts]
        ax.barh(labels, pcts, color=colors, edgecolor="white")
        ax.set_xlim(0, 105)
        ax.set_title(fname, fontsize=10)
        ax.set_xlabel("%")
        for i, p in enumerate(pcts):
            ax.text(p + 1, i, f"{p}%", va="center", fontsize=8)

    plt.tight_layout()
    fig.savefig(plots_dir / "_bucket_balance_warnings.png", dpi=120)
    plt.close(fig)


# ——————————————————————————————————————————————
#  Скалярные пороги
# ——————————————————————————————————————————————

def analyze_scalar_thresholds(df: pd.DataFrame) -> dict:
    """
    Анализирует скалярные пороги из SCALAR — показывает, какой процент данных
    выше/ниже каждого порога.
    """
    result = {}

    scalar_feature_map = {
        "borders_circ_high": "circularity",
        "borders_circ_mid": "circularity",
        "borders_para_high": "perimeter_area_ratio",
        "borders_para_mid": "perimeter_area_ratio",
        "center_periphery_V_abs": "delta_V_center_periphery",
        "center_periphery_S_abs": "delta_S_center_periphery",
        "rim_abs": "delta_V_inner_rim",
        "pigmentation_white": "percent_white_pixels",
        "pigmentation_red_blue": "percent_red_pixels",
        "pigmentation_dark_high": "percent_dark_pixels",
        "pigmentation_dark_mid": "percent_dark_pixels",
        "pigment_inclusion_threshold": "percent_red_pixels",
    }

    for scalar_name, threshold_value in SCALAR.items():
        feature_name = scalar_feature_map.get(scalar_name)
        if feature_name is None or feature_name not in df.columns:
            continue

        values = pd.to_numeric(df[feature_name], errors="coerce").dropna().values
        if len(values) == 0:
            continue

        # Для абсолютных порогов сравниваем |value|
        use_abs = "_abs" in scalar_name
        cmp_values = np.abs(values) if use_abs else values

        above = int(np.sum(cmp_values > threshold_value))
        below = int(np.sum(cmp_values <= threshold_value))
        total = len(cmp_values)

        result[scalar_name] = {
            "feature": feature_name,
            "threshold": threshold_value,
            "use_abs": use_abs,
            "above_count": above,
            "above_percent": round(above / total * 100, 1),
            "below_count": below,
            "below_percent": round(below / total * 100, 1),
            "feature_stats": _feature_stats(cmp_values),
        }

    return result


# ——————————————————————————————————————————————
#  Запуск
# ——————————————————————————————————————————————

def run_full_analysis(
    df: pd.DataFrame,
    output_dir: Path = DEFAULT_OUTPUT,
    plots: bool = True,
) -> dict:
    """
    Полный анализ: распределения, метки, скалярные пороги, графики.
    Сохраняет результаты в output_dir.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Распределения признаков
    dist_report = analyze_distributions(df)
    (output_dir / "distribution_report.json").write_text(
        json.dumps(dist_report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    # 2. Распределение меток (если есть)
    label_report = analyze_label_distribution(df)
    if label_report:
        (output_dir / "label_distribution.json").write_text(
            json.dumps(label_report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # 3. Скалярные пороги
    scalar_report = analyze_scalar_thresholds(df)
    if scalar_report:
        (output_dir / "scalar_thresholds.json").write_text(
            json.dumps(scalar_report, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    # 4. Графики
    if plots:
        plot_distributions(df, output_dir, report=dist_report)

    # 5. Сводка предупреждений
    warnings_summary = {}
    for fname, entry in dist_report.items():
        if entry.get("warnings"):
            warnings_summary[fname] = entry["warnings"]
    if warnings_summary:
        (output_dir / "warnings.json").write_text(
            json.dumps(warnings_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return {
        "features_analyzed": len(dist_report),
        "labels_analyzed": len(label_report),
        "scalars_analyzed": len(scalar_report),
        "warnings": sum(len(w) for w in warnings_summary.values()),
        "output_dir": str(output_dir),
    }


def main():
    parser = argparse.ArgumentParser(description="Анализ распределений признаков")
    parser.add_argument("--csv", required=True, help="CSV с признаками (после extraction или bucketing)")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Папка для результатов")
    parser.add_argument("--no-plots", action="store_true", help="Не строить графики")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    print(f"Загружено {len(df)} строк, {len(df.columns)} колонок")

    summary = run_full_analysis(
        df,
        output_dir=Path(args.output),
        plots=not args.no_plots,
    )

    print(f"\nАнализ завершён:")
    print(f"  Признаков проанализировано: {summary['features_analyzed']}")
    print(f"  Меток проанализировано:     {summary['labels_analyzed']}")
    print(f"  Скалярных порогов:          {summary['scalars_analyzed']}")
    print(f"  Предупреждений:             {summary['warnings']}")
    print(f"  Результаты в: {summary['output_dir']}")


if __name__ == "__main__":
    main()
