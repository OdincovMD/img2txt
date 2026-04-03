"""
Типы классификации дерматоскопических образований для текстового описания.
Используется в модуле generation/description_inference.py для структурирования
входных данных для LLM.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List


class FeatureType(Enum):
    """Тип выявленных признаков в образовании."""
    SINGLE = "Один признак"
    MULTIPLE = "Несколько признаков"


class Structure(Enum):
    """Основная структура образования."""
    STRUCTURELESS = "Бесструктурная область"
    GLOBULES = "Комки"
    LINES = "Линии"
    DOTS = "Точки"
    CIRCLES = "Круги"
    PSEUDOPODIA = "Псевдоподии"


class LineType(Enum):
    """Тип линий (если структура = LINES)."""
    CURVED = "Изогнутые"
    PARALLEL = "Параллельные"
    RETICULAR = "Ретикулярные"
    BRANCHED = "Разветвленные"


class CountColor(Enum):
    """Количество цветов в образовании."""
    ONE = "Один цвет"
    MANY = "Несколько цветов"


class PigmentType(Enum):
    """Тип пигмента."""
    MELANIN = "Меланин"
    OTHER = "Другой пигмент"


class Symmetry(Enum):
    """Симметричность образования."""
    SYMMETRIC = "Симметричные"
    ASYMMETRIC = "Асимметричные"


@dataclass
class ClassificationResult:
    """
    Результат классификации дерматоскопического образования.

    Используется как опциональный входной параметр для generate_description().
    Если не предоставлен, описание генерируется только на основе важных признаков.
    """
    feature_type: FeatureType
    structure: Structure
    properties: List[str] = field(default_factory=list)
    final_class: str = ""
