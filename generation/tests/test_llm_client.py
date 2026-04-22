import re

from generation.classification_types import ClassificationResult, FeatureType, Structure
from generation.llm_client import generate_description


def _sentence_count(text: str) -> int:
    return len(re.findall(r"[^.!?]+[.!?]", text))


def test_generate_description_returns_three_to_five_sentences():
    features_text = (
        "ФОРМА И РАЗМЕР:\n"
        "- Форма образования: образование имеет овальную форму\n"
        "- Размер (периметр контура): образование среднего размера\n\n"
        "ГРАНИЦЫ И КОНТУР:\n"
        "- Характер границ: границы умеренно неровные, местами нечёткие\n"
        "- Симметрия образования: присутствует умеренная асимметрия по цвету или форме\n\n"
        "ЦВЕТОВОЙ ПАТТЕРН:\n"
        "- Однородность окраски: окраска умеренно неоднородная, присутствуют зоны различной интенсивности\n"
        "- Преобладающий оттенок: преобладает коричневый оттенок\n\n"
        "ТЕКСТУРА:\n"
        "- Текстура поверхности: текстура умеренно неоднородная, с участками различной плотности"
    )

    classification = ClassificationResult(
        feature_type=FeatureType.MULTIPLE,
        structure=Structure.GLOBULES,
        properties=["асимметричные", "полихромные"],
        final_class="Меланома",
    )

    text = generate_description(features_text, classification)
    print(f"\nGroq response:\n{text}\n")

    assert text.strip()
    assert 3 <= _sentence_count(text) <= 5
    assert "меланом" in text.lower()
