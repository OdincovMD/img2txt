"""
Конфигурация пайплайна обучения v2 (XGBoost).

Реэкспортирует константы из config.importance_config — единого источника правды.
"""

from config.importance_config import (
    LABEL_NAMES,
    NUM_LABELS,
    LABEL_TO_IDX,
)

# Сколько признаков выбираем как «важные» при инференсе
TOP_K: int = 10
