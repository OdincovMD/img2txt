"""Typed classification payloads for clinical description generation."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable


class DisplayEnum(str, Enum):
    """Enum whose string value is already user-facing display text."""

    def __str__(self) -> str:
        return self.value


class FeatureType(DisplayEnum):
    """Тип выявленных признаков в образовании."""
    SINGLE = "Один признак"
    MULTIPLE = "Несколько признаков"


class Structure(DisplayEnum):
    """Основная структура образования."""
    STRUCTURELESS = "Бесструктурная область"
    GLOBULES = "Комки"
    LINES = "Линии"
    DOTS = "Точки"
    CIRCLES = "Круги"
    PSEUDOPODIA = "Псевдоподии"


class LineType(DisplayEnum):
    """Тип линий (если структура = LINES)."""
    CURVED = "Изогнутые"
    PARALLEL = "Параллельные"
    RETICULAR = "Ретикулярные"
    BRANCHED = "Разветвленные"


class CountColor(DisplayEnum):
    """Количество цветов в образовании."""
    ONE = "Один цвет"
    MANY = "Несколько цветов"


class PigmentType(DisplayEnum):
    """Тип пигмента."""
    MELANIN = "Меланин"
    OTHER = "Другой пигмент"


class Symmetry(DisplayEnum):
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
    properties: list[str] = field(default_factory=list)
    final_class: str = ""

    def __post_init__(self) -> None:
        self.properties = [str(item).strip() for item in self.properties if str(item).strip()]
        self.final_class = self.final_class.strip()

    def has_properties(self) -> bool:
        return bool(self.properties)

    def property_text(self, fallback: str = "—") -> str:
        return ", ".join(self.properties) if self.properties else fallback

    def iter_prompt_lines(self) -> Iterable[str]:
        yield f"- Тип признаков: {self.feature_type.value}"
        yield f"- Структура: {self.structure.value}"
        yield f"- Свойства: {self.property_text()}"
        if self.final_class:
            yield f"- Финальная классификация: {self.final_class}"
