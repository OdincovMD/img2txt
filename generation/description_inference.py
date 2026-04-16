"""
Step 4: Clinical description generation.
Converts top-k important features + optional classification → Russian clinical text via Qwen2.5-7B.
Public API: generate_descriptions_batch(df, ...) → df with 'description' column.
"""

from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import torch
from tqdm import tqdm

from generation.classification_types import ClassificationResult


# Global model cache
_model_cache = {"model": None, "tokenizer": None, "device": None}


# ---------------------------------------------------------------------------
# Clinical descriptions for all 24 active labels.
# Keys match LABEL_NAMES from config/importance_config.py.
# Each entry: (category, russian_name, {value: clinical_text})
# ---------------------------------------------------------------------------
_LABEL_DESCRIPTIONS: Dict[str, Tuple[str, str, Dict[str, str]]] = {
    # --- ФОРМА ---
    "shape": ("форма", "Форма образования", {
        "округлая": "образование имеет правильную округлую форму",
        "овальная": "образование имеет овальную форму",
        "неправильная": "образование имеет неправильную, асимметричную форму",
    }),
    "elongation": ("форма", "Вытянутость", {
        "округлая": "контур приближен к кругу, без вытянутости",
        "умеренно вытянутая": "образование умеренно вытянуто по одной из осей",
        "вытянутая": "образование значительно вытянуто, соотношение осей выражено непропорционально",
    }),
    "eccentricity": ("форма", "Эксцентриситет", {
        "форма:округлая": "форма близка к кругу",
        "форма:умеренно_вытянутая": "форма умеренно эллиптическая",
        "форма:сильно_вытянутая": "форма резко эллиптическая, значительно вытянута",
    }),
    "perimeter": ("форма", "Размер (периметр контура)", {
        "периметр:маленький": "образование мелкое",
        "периметр:средний": "образование среднего размера",
        "периметр:большой": "образование крупное",
        "периметр:очень_большой": "образование очень крупное",
    }),
    # --- ГРАНИЦЫ ---
    "borders": ("границы", "Характер границ", {
        "ровные": "границы ровные, чёткие",
        "умеренно неровные": "границы умеренно неровные, местами нечёткие",
        "фестончатые": "границы фестончатые, выражено неровные",
    }),
    "fractal_dimension": ("границы", "Изрезанность контура", {
        "граница:слабо_изрезанная": "контур гладкий, без существенных зубцов",
        "граница:умеренно_изрезанная": "контур умеренно зубчатый",
        "граница:сильно_изрезанная": "контур сильно изрезанный, неровный",
    }),
    "asymmetry": ("границы", "Симметрия образования", {
        "слабая": "образование достаточно симметрично",
        "умеренная": "присутствует умеренная асимметрия по цвету или форме",
        "выраженная": "выраженная асимметрия по нескольким осям",
    }),
    "rim": ("границы", "Краевой ободок", {
        "без выраженного ободка": "краевой ободок не определяется",
        "светлый ободок по краю": "по периферии визуализируется светлый ободок",
        "тёмный ободок по краю": "по периферии визуализируется тёмный пигментированный ободок",
    }),
    # --- ЦВЕТ ---
    "color_homogeneity": ("цвет", "Однородность окраски", {
        "цвет однородный": "окраска равномерная, без значимых вариаций",
        "цвет умеренно неоднородный": "окраска умеренно неоднородная, присутствуют зоны различной интенсивности",
        "цвет неоднородный": "окраска выражено неоднородная, отмечается мозаичность цветового паттерна",
    }),
    "dominant_hue": ("цвет", "Преобладающий оттенок", {
        "красноватый": "преобладает красновато-розовый оттенок",
        "желтовато-коричневый": "преобладает желтовато-коричневый оттенок",
        "коричневый": "преобладает коричневый оттенок",
        "синеватый": "преобладает синевато-серый оттенок",
        "фиолетовый": "преобладает фиолетовый оттенок",
        "зеленоватый": "преобладает зеленоватый оттенок",
        "неопределенный": "доминирующий оттенок не определяется чётко",
    }),
    "contrast": ("цвет", "Контраст образования с кожей", {
        "низкий": "образование слабо контрастирует с окружающей кожей",
        "умеренный": "образование умеренно контрастирует с окружающей кожей",
        "выраженный": "образование резко контрастирует с окружающей кожей",
    }),
    "palette": ("цвет", "Цветовая палитра", {
        "монотонная": "цветовая палитра однотонная",
        "умеренно разнообразная": "присутствуют 2–3 различных оттенка",
        "полихромная": "палитра полихромная, присутствуют множественные оттенки",
    }),
    "color_distance_euclidean": ("цвет", "Цветовое отличие от кожи (LAB)", {
        "контраст:очень_слабый": "цвет образования практически не отличается от окружающей кожи",
        "контраст:умеренный": "цвет образования умеренно отличается от окружающей кожи",
        "контраст:выраженный": "цвет образования выражено отличается от окружающей кожи",
        "контраст:сильный": "цвет образования резко отличается от окружающей кожи",
        "контраст:очень_сильный": "цвет образования очень резко отличается от окружающей кожи",
    }),
    "delta_H_center_periphery": ("цвет", "Различие оттенка центр–периферия", {
        "центр:краснее_периферии": "центральная зона имеет более красный оттенок, чем периферия",
        "центр:оттенок_как_периферия": "оттенок в центре и на периферии сопоставим",
        "центр:синее_периферии": "центральная зона имеет более синеватый оттенок, чем периферия",
    }),
    "delta_S_center_periphery": ("цвет", "Различие насыщенности центр–периферия", {
        "насыщенность_центра:ниже": "центр менее насыщен по цвету, чем периферия",
        "насыщенность_центра:равна": "насыщенность цвета в центре и на периферии сопоставима",
        "насыщенность_центра:умеренно_выше": "центр умеренно насыщеннее периферии",
        "насыщенность_центра:выше": "центр заметно насыщеннее периферии",
        "насыщенность_центра:значительно_выше": "центр значительно насыщеннее периферии",
    }),
    "delta_V_center_periphery": ("цвет", "Различие яркости центр–периферия", {
        "яркость_центра:значительно_темнее": "центр значительно темнее периферии",
        "яркость_центра:темнее": "центр темнее периферии",
        "яркость_центра:равная": "яркость центра и периферии сопоставимы",
        "яркость_центра:светлее": "центр светлее периферии",
        "яркость_центра:значительно_светлее": "центр значительно светлее периферии",
    }),
    "delta_V_inner_rim": ("цвет", "Яркость ободка относительно центра", {
        "ободок:темнее_центра": "краевая зона значительно темнее центральной части",
        "ободок:слегка_темнее": "краевая зона слегка темнее центральной части",
        "ободок:одинаковый": "яркость краевой и центральной зон сопоставима",
        "ободок:светлее": "краевая зона светлее центральной части",
        "ободок:значительно_светлее": "краевая зона значительно светлее центральной части",
    }),
    "delta_V_left_right": ("цвет", "Асимметрия яркости лево–право", {
        "яркость_асимметрия:сильная_лево_право": "выраженная асимметрия яркости между левой и правой половинами",
        "яркость_асимметрия:слабая": "яркость распределена симметрично слева и справа",
        "яркость_асимметрия:сильная_право_лево": "выраженная асимметрия яркости между правой и левой половинами",
    }),
    "delta_V_top_bottom": ("цвет", "Асимметрия яркости верх–низ", {
        "яркость_асимметрия:сильная_верх_низ": "верхняя часть значительно отличается по яркости от нижней",
        "яркость_асимметрия:слабая": "яркость распределена симметрично по вертикали",
        "яркость_асимметрия:сильная_низ_верх": "нижняя часть значительно отличается по яркости от верхней",
    }),
    "delta_S_left_right": ("цвет", "Асимметрия насыщенности лево–право", {
        "насыщенность_асимметрия:сильная_лево_право": "выраженная асимметрия насыщенности цвета между левой и правой половинами",
        "насыщенность_асимметрия:слабая": "насыщенность цвета распределена симметрично",
        "насыщенность_асимметрия:сильная_право_лево": "выраженная асимметрия насыщенности между правой и левой половинами",
    }),
    "std_H_lesion": ("цвет", "Вариабельность оттенка", {
        "вариабельность_оттенка:низкая": "оттенок в пределах образования однородный",
        "вариабельность_оттенка:средняя": "присутствует умеренная вариабельность оттенка",
        "вариабельность_оттенка:высокая": "выраженная вариабельность оттенка, мультихромность",
    }),
    # --- ТЕКСТУРА ---
    "texture": ("текстура", "Текстура поверхности", {
        "преимущественно однородная": "текстура поверхности однородная",
        "умеренно неоднородная": "текстура умеренно неоднородная, с участками различной плотности",
        "неоднородная": "текстура выражено неоднородная, гетерогенная",
    }),
    "structure_order": ("текстура", "Упорядоченность структуры", {
        "упорядоченная": "внутренняя структура упорядоченная, регулярная",
        "средне упорядоченная": "внутренняя структура умеренно упорядочена",
        "хаотичная": "внутренняя структура хаотичная, нерегулярная",
    }),
    "glcm_energy": ("текстура", "Энергия текстуры (GLCM)", {
        "энергия_текстуры:низкая": "текстура сложная, с выраженной гетерогенностью",
        "энергия_текстуры:средняя": "текстура умеренной сложности",
        "энергия_текстуры:высокая": "текстура простая, однородная",
    }),
}

# Category display order for grouped output
_CATEGORY_ORDER = ["форма", "границы", "цвет", "текстура"]


def _parse_feature_entries(
    important_features: List[str],
) -> Dict[str, List[Tuple[str, str]]]:
    """Parse 'feature:value' strings into grouped {category: [(name, clinical_text)]} dict."""
    from config.config import FEATURE_ROUTING

    grouped: Dict[str, List[Tuple[str, str]]] = {c: [] for c in _CATEGORY_ORDER}

    for entry in important_features:
        parts = entry.split(":", 1)
        feature_key = parts[0]
        label_value = parts[1] if len(parts) > 1 else ""

        desc_entry = _LABEL_DESCRIPTIONS.get(feature_key)
        if desc_entry:
            category, rus_name, value_map = desc_entry
            clinical_text = value_map.get(label_value, label_value)
            grouped.setdefault(category, []).append((rus_name, clinical_text))
        else:
            routing = FEATURE_ROUTING.get(feature_key)
            if routing:
                _cat_path, rus_name, _unit = routing
            else:
                rus_name = feature_key
            cat = "цвет" if "color" in feature_key or "delta" in feature_key else "текстура"
            grouped.setdefault(cat, []).append((rus_name, label_value))

    return grouped


_SYSTEM_PROMPT = (
    "Ты — медицинский ассистент, составляющий клинические дерматоскопические описания новообразований кожи. "
    "Пиши ТОЛЬКО на русском языке.\n\n"
    "ПРАВИЛА СОСТАВЛЕНИЯ ОПИСАНИЯ:\n"
    "1. Ровно 3–5 предложений. Профессиональный клинический стиль.\n"
    "2. Структура текста: начни с формы и размера → границы и контур → цветовой паттерн → текстура.\n"
    "3. Используй ТОЛЬКО предоставленные признаки. Не добавляй информацию от себя.\n"
    "4. Признаки уже расшифрованы на русском — перефразируй их в связный клинический текст, "
    "НЕ перечисляй их списком.\n"
    "5. Если предоставлена КЛАССИФИКАЦИЯ — она задаёт общий контекст описания: "
    "упомяни структурный тип и свойства в тексте, но НЕ ставь диагноз.\n"
    "6. ЗАПРЕЩЕНО: ставить диагноз, предполагать диагноз, упоминать названия заболеваний, "
    "давать рекомендации, добавлять комментарии или преамбулы.\n"
    "7. Отвечай сразу текстом описания."
)

_FEW_SHOT_EXAMPLES = [
    # Example 1: typical benign-looking lesion
    {
        "features": (
            "ФОРМА И РАЗМЕР:\n"
            "- Форма образования: образование имеет правильную округлую форму\n"
            "- Размер (периметр контура): образование среднего размера\n\n"
            "ГРАНИЦЫ И КОНТУР:\n"
            "- Характер границ: границы ровные, чёткие\n"
            "- Изрезанность контура: контур гладкий, без существенных зубцов\n"
            "- Симметрия образования: образование достаточно симметрично\n\n"
            "ЦВЕТОВОЙ ПАТТЕРН:\n"
            "- Однородность окраски: окраска равномерная, без значимых вариаций\n"
            "- Преобладающий оттенок: преобладает коричневый оттенок\n"
            "- Контраст образования с кожей: образование умеренно контрастирует с окружающей кожей\n\n"
            "ТЕКСТУРА:\n"
            "- Текстура поверхности: текстура поверхности однородная"
        ),
        "classification": None,
        "output": (
            "Образование округлой формы, среднего размера, с чёткими ровными границами и симметричным контуром. "
            "Окраска равномерная, преобладает коричневый оттенок, умеренно контрастирующий с окружающей кожей. "
            "Текстура поверхности однородная, без участков структурной неоднородности."
        ),
    },
    # Example 2: suspicious lesion with classification
    {
        "features": (
            "ФОРМА И РАЗМЕР:\n"
            "- Форма образования: образование имеет неправильную, асимметричную форму\n"
            "- Вытянутость: образование умеренно вытянуто по одной из осей\n\n"
            "ГРАНИЦЫ И КОНТУР:\n"
            "- Характер границ: границы фестончатые, выражено неровные\n"
            "- Изрезанность контура: контур сильно изрезанный, неровный\n"
            "- Симметрия образования: выраженная асимметрия по нескольким осям\n"
            "- Краевой ободок: по периферии визуализируется тёмный пигментированный ободок\n\n"
            "ЦВЕТОВОЙ ПАТТЕРН:\n"
            "- Однородность окраски: окраска выражено неоднородная, отмечается мозаичность цветового паттерна\n"
            "- Различие яркости центр–периферия: центр значительно темнее периферии\n"
            "- Вариабельность оттенка: выраженная вариабельность оттенка, мультихромность\n"
            "- Цветовая палитра: палитра полихромная, присутствуют множественные оттенки\n\n"
            "ТЕКСТУРА:\n"
            "- Упорядоченность структуры: внутренняя структура хаотичная, нерегулярная"
        ),
        "classification": (
            "КЛАССИФИКАЦИЯ ОБРАЗОВАНИЯ:\n"
            "- Структура: Комки\n"
            "- Свойства: асимметричные, полихромные\n"
            "- Итоговый класс: Подозрительное"
        ),
        "output": (
            "Образование неправильной формы, умеренно вытянутое, с выраженной асимметрией по нескольким осям. "
            "Границы фестончатые, сильно изрезанные, по периферии определяется тёмный пигментированный ободок. "
            "Цветовой паттерн мозаичный, полихромный — центральная часть значительно темнее периферии, "
            "отмечается выраженная вариабельность оттенков. "
            "При дерматоскопии определяются асимметрично расположенные комки. "
            "Текстура неоднородная, внутренняя структура хаотичная, нерегулярная."
        ),
    },
]


def _format_grouped_features(grouped: Dict[str, List[Tuple[str, str]]]) -> str:
    """Format grouped features into structured text for the prompt."""
    category_titles = {
        "форма": "ФОРМА И РАЗМЕР",
        "границы": "ГРАНИЦЫ И КОНТУР",
        "цвет": "ЦВЕТОВОЙ ПАТТЕРН",
        "текстура": "ТЕКСТУРА",
    }
    sections = []
    for cat in _CATEGORY_ORDER:
        items = grouped.get(cat, [])
        if not items:
            continue
        title = category_titles.get(cat, cat.upper())
        lines = [f"- {name}: {text}" for name, text in items]
        sections.append(f"{title}:\n" + "\n".join(lines))
    return "\n\n".join(sections)


def _format_classification(classification: ClassificationResult) -> str:
    """Format classification block for the prompt."""
    props_str = ", ".join(classification.properties) if classification.properties else "\u2014"
    lines = ["КЛАССИФИКАЦИЯ ОБРАЗОВАНИЯ:"]
    lines.append(f"- Структура: {classification.structure.value}")
    lines.append(f"- Свойства: {props_str}")
    if classification.final_class:
        lines.append(f"- Итоговый класс: {classification.final_class}")
    return "\n".join(lines)


def _build_messages(
    grouped_features: Dict[str, List[Tuple[str, str]]],
    classification: Optional[ClassificationResult],
) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = [{"role": "system", "content": _SYSTEM_PROMPT}]

    # Few-shot examples
    for ex in _FEW_SHOT_EXAMPLES:
        user_text = f"Составь клиническое дерматоскопическое описание по следующим данным:\n\n{ex['features']}"
        if ex.get("classification"):
            user_text += f"\n\n{ex['classification']}"
        messages.append({"role": "user", "content": user_text})
        messages.append({"role": "assistant", "content": ex["output"]})

    # Actual request
    features_text = _format_grouped_features(grouped_features)

    classification_block = ""
    if classification is not None:
        classification_block = "\n\n" + _format_classification(classification)

    user_content = (
        f"Составь клиническое дерматоскопическое описание по следующим данным:\n\n"
        f"{features_text}{classification_block}"
    )

    messages.append({"role": "user", "content": user_content})
    return messages


def _load_model(
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    device: Optional[torch.device] = None,
) -> Tuple[Any, Any, torch.device]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading {model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if device.type == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu")

    model.eval()
    print(f"Model loaded on {device}")
    return model, tokenizer, device


def _generate_text(
    messages: List[Dict[str, str]], model, tokenizer, device: torch.device, max_tokens: int = 300,
) -> str:
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.3, top_p=0.85,
            repetition_penalty=1.15, do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()


def _generate_single(
    important_features: List[str],
    classification: Optional[ClassificationResult],
    model, tokenizer, device: torch.device,
    max_tokens: int = 256,
) -> str:
    """Generate description for one set of features. Returns text or error string."""
    if not important_features:
        return "Недостаточно данных для описания."
    grouped = _parse_feature_entries(important_features)
    messages = _build_messages(grouped, classification)
    return _generate_text(messages, model, tokenizer, device, max_tokens)


def _ensure_model(
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    device: Optional[torch.device] = None,
    use_cache: bool = True,
) -> Tuple[Any, Any, torch.device]:
    """Load model or return cached instance."""
    global _model_cache
    if use_cache and _model_cache["model"] is not None:
        return _model_cache["model"], _model_cache["tokenizer"], _model_cache["device"]

    model, tokenizer, device_used = _load_model(model_name, device)
    if use_cache:
        _model_cache["model"] = model
        _model_cache["tokenizer"] = tokenizer
        _model_cache["device"] = device_used
    return model, tokenizer, device_used


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_descriptions_batch(
    df: pd.DataFrame,
    classification_col: Optional[str] = "classification",
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    device: Optional[torch.device] = None,
    max_tokens: int = 300,
    use_cache: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Step 4: Generate Russian clinical descriptions for all rows in df.

    Args:
        df: DataFrame with 'important_labels' column (from step 3).
            Optional: column with ClassificationResult objects (see classification_col).
        classification_col: Column name containing ClassificationResult objects.
            If column doesn't exist or value is None, classification block is omitted.
        model_name: HuggingFace model ID
        device: torch device (defaults to cuda if available)
        max_tokens: Max tokens per description
        use_cache: Cache model in memory for subsequent calls
        verbose: Show progress bar

    Returns:
        df with added column:
        - description: Russian clinical text (3-5 sentences)
    """
    df = df.copy()
    has_classification = classification_col in df.columns

    try:
        model, tokenizer, device_used = _ensure_model(model_name, device, use_cache)

        descriptions = []
        iterator = tqdm(df.itertuples(index=False), total=len(df)) if verbose else df.itertuples(index=False)

        for row in iterator:
            row_dict = row._asdict()
            important_features = row_dict.get("important_labels", [])
            if not isinstance(important_features, list):
                important_features = []

            classification = row_dict.get(classification_col) if has_classification else None
            if classification is not None and not isinstance(classification, ClassificationResult):
                classification = None

            text = _generate_single(important_features, classification, model, tokenizer, device_used, max_tokens)
            descriptions.append(text)

        df["description"] = descriptions

    except ImportError:
        print("Warning: transformers module not installed.")
        df["description"] = "Модуль transformers не установлен."

    except Exception as e:
        print(f"Warning: Description generation failed: {e}")
        df["description"] = f"Ошибка генерации описания: {e}"

    return df
