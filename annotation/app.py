"""
Annotation mini-app for labeling important dermoscopic features.
Run: python annotation/app.py --csv <bucketed_features.csv> [--output annotations.csv] [--port 5050]
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config import FEATURE_ROUTING
from config.importance_config import LABEL_NAMES

# ---------------------------------------------------------------------------
# Label metadata: category + Russian description for each of 56 labels
# ---------------------------------------------------------------------------

# Compound labels not in FEATURE_ROUTING
_COMPOUND_LABELS = {
    "asymmetry":          ("general",      "Асимметрия"),
    "borders":            ("border",       "Границы"),
    "contrast":           ("color.global", "Контраст"),
    "palette":            ("color.global", "Палитра"),
    "texture":            ("texture",      "Текстура"),
    "lobulation":         ("shape",        "Лобуляция"),
    "pigmentation":       ("color.global", "Пигментация"),
    "elongation":         ("shape",        "Вытянутость"),
    "rim":                ("color.local",  "Ободок"),
    "color_homogeneity":  ("color.global", "Однородность цвета"),
    "texture_coarseness": ("texture",      "Грубость текстуры"),
    "shape":              ("shape",        "Форма"),
    "dominant_hue":       ("color.global", "Доминантный оттенок"),
    "structure_order":    ("texture",      "Упорядоченность структуры"),
}

CATEGORY_DISPLAY = {
    "general":     "Общие",
    "color.global": "Цвет (глобальный)",
    "color.local":  "Цвет (локальный)",
    "shape":        "Форма",
    "border":       "Граница",
    "texture":      "Текстура",
}

CATEGORY_ORDER = ["general", "color.global", "color.local", "shape", "border", "texture"]


def _build_label_meta():
    """Build {label_key: {category, description}} for all LABEL_NAMES."""
    meta = {}
    for name in LABEL_NAMES:
        if name in _COMPOUND_LABELS:
            cat, desc = _COMPOUND_LABELS[name]
        elif name in FEATURE_ROUTING:
            cat, desc, _unit = FEATURE_ROUTING[name]
        else:
            cat, desc = "general", name
        meta[name] = {"category": cat, "description": desc}
    return meta


LABEL_META = _build_label_meta()

# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------

app = Flask(__name__)

INPUT_DF: pd.DataFrame = pd.DataFrame()
OUTPUT_PATH: Path = Path("annotations.csv")
ANNOTATED: dict = {}  # image_path -> list of "key:value" strings


def _load_annotations():
    """Load existing annotations from output CSV."""
    global ANNOTATED
    if OUTPUT_PATH.exists():
        df = pd.read_csv(OUTPUT_PATH)
        for _, row in df.iterrows():
            try:
                labels = json.loads(row["important_labels"])
            except (json.JSONDecodeError, KeyError, TypeError):
                labels = []
            ANNOTATED[row["image_path"]] = labels


def _save_annotations():
    """Atomically write all annotations to output CSV."""
    rows = [
        {"image_path": path, "important_labels": json.dumps(labels, ensure_ascii=False)}
        for path, labels in ANNOTATED.items()
    ]
    df = pd.DataFrame(rows)
    fd, tmp = tempfile.mkstemp(suffix=".csv", dir=OUTPUT_PATH.parent)
    os.close(fd)
    try:
        df.to_csv(tmp, index=False)
        os.replace(tmp, OUTPUT_PATH)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _get_labels_for_row(idx: int) -> dict:
    """Parse labels_json for a given row index."""
    raw = INPUT_DF.iloc[idx].get("labels_json", "{}")
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}
    if isinstance(raw, dict):
        return raw
    return {}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    first_unannotated = None
    for i in range(len(INPUT_DF)):
        if INPUT_DF.iloc[i]["image_path"] not in ANNOTATED:
            first_unannotated = i
            break
    return jsonify({
        "total": len(INPUT_DF),
        "annotated": len(ANNOTATED),
        "first_unannotated": first_unannotated if first_unannotated is not None else 0,
    })


@app.route("/api/image/<int:idx>")
def api_image(idx: int):
    if idx < 0 or idx >= len(INPUT_DF):
        return jsonify({"error": "Index out of range"}), 404

    row = INPUT_DF.iloc[idx]
    image_path = row["image_path"]
    labels = _get_labels_for_row(idx)

    # Group labels by category
    categories = {}
    for key in LABEL_NAMES:
        value = labels.get(key)
        if value is None:
            continue
        # Skip non-scalar values (dicts, lists of dicts)
        if isinstance(value, dict):
            continue
        if isinstance(value, list):
            value = ", ".join(str(v) for v in value)

        # Strip prefix before colon if present (e.g. "разнообразие_оттенка:низкое" → "низкое")
        value_str = str(value)
        if ":" in value_str:
            value_str = value_str.split(":", 1)[1]

        meta = LABEL_META.get(key, {"category": "general", "description": key})
        cat = meta["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append({
            "key": key,
            "description": meta["description"],
            "value": value_str,
        })

    # Order categories
    ordered_categories = []
    for cat_key in CATEGORY_ORDER:
        if cat_key in categories:
            ordered_categories.append({
                "key": cat_key,
                "name": CATEGORY_DISPLAY.get(cat_key, cat_key),
                "labels": categories[cat_key],
            })

    saved = ANNOTATED.get(image_path, None)

    return jsonify({
        "index": idx,
        "image_path": image_path,
        "categories": ordered_categories,
        "saved_labels": saved,
        "is_annotated": saved is not None,
    })


@app.route("/api/image/<int:idx>/file")
def api_image_file(idx: int):
    if idx < 0 or idx >= len(INPUT_DF):
        return "Not found", 404
    image_path = INPUT_DF.iloc[idx]["image_path"]
    p = Path(image_path)
    if not p.is_absolute():
        p = Path(INPUT_DF.attrs.get("csv_dir", ".")) / p
    if not p.exists():
        return f"Image not found: {p}", 404
    return send_file(p)


@app.route("/api/image/<int:idx>/save", methods=["POST"])
def api_save(idx: int):
    if idx < 0 or idx >= len(INPUT_DF):
        return jsonify({"error": "Index out of range"}), 404

    data = request.get_json(force=True)
    important_labels = data.get("important_labels", [])

    if len(important_labels) > 10:
        return jsonify({"error": "Maximum 10 labels"}), 400

    image_path = INPUT_DF.iloc[idx]["image_path"]
    ANNOTATED[image_path] = important_labels
    _save_annotations()

    # Find next unannotated
    next_idx = None
    for i in range(idx + 1, len(INPUT_DF)):
        if INPUT_DF.iloc[i]["image_path"] not in ANNOTATED:
            next_idx = i
            break

    return jsonify({"saved": True, "next_index": next_idx})


@app.route("/api/next_unannotated")
def api_next_unannotated():
    for i in range(len(INPUT_DF)):
        if INPUT_DF.iloc[i]["image_path"] not in ANNOTATED:
            return jsonify({"index": i})
    return jsonify({"index": None})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global INPUT_DF, OUTPUT_PATH

    parser = argparse.ArgumentParser(description="Annotation mini-app for important features")
    parser.add_argument("--csv", required=True, help="Path to bucketed features CSV (needs image_path + labels_json)")
    parser.add_argument("--output", default="annotations.csv", help="Output CSV path (default: annotations.csv)")
    parser.add_argument("--port", type=int, default=5050, help="Port (default: 5050)")
    args = parser.parse_args()

    csv_path = Path(args.csv).resolve()
    if not csv_path.exists():
        print(f"Error: CSV not found: {csv_path}")
        sys.exit(1)

    OUTPUT_PATH = Path(args.output).resolve()

    INPUT_DF = pd.read_csv(csv_path)
    INPUT_DF.attrs["csv_dir"] = str(csv_path.parent)

    if "image_path" not in INPUT_DF.columns:
        print("Error: CSV must have 'image_path' column")
        sys.exit(1)
    if "labels_json" not in INPUT_DF.columns:
        print("Error: CSV must have 'labels_json' column")
        sys.exit(1)

    _load_annotations()

    print(f"Loaded {len(INPUT_DF)} images, {len(ANNOTATED)} already annotated")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Open http://localhost:{args.port}")

    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()
