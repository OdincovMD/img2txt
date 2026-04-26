# img2txt — исследовательская ветка

Эта ветка (`main`) содержит **исследовательский и экспериментальный код**
пайплайна автоматического описания дерматоскопических изображений кожных
новообразований.

Рабочий сервис, интегрированный в основное приложение
[skin-cancer-ai](https://github.com/OdincovMD/skin-cancer-ai), живёт в ветке
[`feature-description-service`](https://github.com/OdincovMD/img2txt/tree/feature-description-service).

---

## Что такое img2txt

Система превращает дерматоскопическое изображение в короткое русскоязычное
клиническое описание. Описание строится по структурированным признакам — не
по «свободной интерпретации» нейросети, а по измеренным и выбранным свойствам
очага.

Пайплайн состоит из четырёх последовательных шагов:

```
изображение + маска
    → (1) извлечение ~60 численных признаков
    → (2) перевод в категориальные клинические метки
    → (3) ранжирование: top-k важных признаков
    → (4) генерация клинического текста по top-k
```

Финальный текст — короткое наблюдательное описание (3–5 предложений) на
русском языке. Диагноз модель не ставит.

---

## Структура репозитория

```
.
├── extraction/          # Шаг 1: извлечение признаков (цвет, форма, граница, текстура)
│   ├── features.py
│   ├── derived_features.py
│   ├── segmentation.py      # YOLO + UNet fallback
│   └── feature_extraction_batch.py
│
├── bucketing/           # Шаг 2: численные признаки → клинические метки
│   ├── threshold_rules.py
│   ├── schema.py
│   └── feature_bucketing_batch.py
│
├── model/               # Шаг 3: ранжирование важных признаков
│   ├── model.py             # EfficientNet/ResNet backbone
│   ├── mlp.py               # MLP на табличных признаках
│   ├── run.py               # обучение CNN-ветки
│   ├── mlp_run.py           # обучение MLP-ветки
│   ├── inference.py
│   ├── mlp_inference.py
│   └── checkpoints/
│
├── generation/          # Шаг 4: генерация описания через LLM
│   ├── description_inference.py   # Qwen2.5-7B-Instruct
│   ├── description_templates.py
│   ├── classification_types.py
│   └── llm_client.py
│
├── annotation/          # Инструмент ручной разметки (Flask + vanilla JS)
│   ├── app.py
│   ├── templates/
│   └── static/
│
├── config/              # Конфиги словаря меток и порогов
│   ├── threshold_config.py
│   └── __init__.py
│
├── analysis/            # Анализ распределений, калибровка порогов
│   ├── distribution_analysis.py
│   ├── feature_metadata.py
│   └── results/
│       ├── distribution_report.json
│       ├── scalar_thresholds.json
│       └── label_distribution.json
│
├── notebooks/           # Эксперименты
│   └── img2txt.ipynb
│
├── docs/                # Документация по каждому шагу
│   ├── research_step1_extraction.md
│   ├── research_step2_bucketing.md
│   ├── research_step3_ranking.md
│   └── research_step4_generation.md
│
├── weights/             # Веса YOLO и UNet для сегментации
├── images/              # Датасет изображений
├── annotations.csv      # Экспертная разметка важных признаков
├── features.csv         # Извлечённые признаки
├── bucketed_features.csv
└── requirements.txt
```

---

## Шаги пайплайна

### Шаг 1 — Извлечение признаков

Двухступенчатая сегментация: YOLO (основной) → UNet (fallback). Извлекается
около 60 численных признаков в пяти группах:

| Группа | Ключевые признаки |
|---|---|
| Глобальный цвет | mean/std HSV, entropy, color_distance_euclidean / ΔE2000, dominant_colors, percent_dark/white/red/blue |
| Локальный цвет | delta_H/S/V центр↔периферия, лево↔право, верх↔низ, inner_rim |
| Форма | area, circularity, aspect_ratio, eccentricity, solidity, extent |
| Граница | radial_variance, convexity, fractal_dimension |
| Текстура | GLCM (contrast, homogeneity, energy, entropy), LBP (uniformity, entropy, mean, std) |

Подробнее: [docs/research_step1_extraction.md](docs/research_step1_extraction.md)

### Шаг 2 — Бакетизация

Численные признаки → категориальные метки (`shape:неправильная`,
`borders:фестончатые`, ...). Часть меток строится прямыми пороговыми правилами,
часть — составными (asymmetry, palette, pigmentation, lobulation и др.).

Подробнее: [docs/research_step2_bucketing.md](docs/research_step2_bucketing.md)

### Шаг 3 — Ранжирование важных признаков

Multi-label ranking: по признакам предсказываются top-k меток, которые эксперт
счёл бы диагностически значимыми. Метрика: Precision@10 = Recall@10.

Пройденные направления:

- **image-only (EfficientNet)** — плато ~0.51 P@10 на 400 аннотациях;
- **multimodal (CNN + tabular)** — ~0.61;
- **XGBoost (independent)** — ~0.62, стал основным baseline;
- **xgb_classifier_chain** — учитывает зависимости между метками через OOF;
- **MLP на табличных признаках** — альтернативная нейросетевая ветка.

Ключевые уроки: pos_weight вреден для top-k порядка; hue shift в аугментациях
ломает цветовой сигнал; псевдо-разметка на z-score несовместима с экспертной;
сжатие словаря с 56 до 24 реально используемых меток дало один из сильнейших
приростов.

Подробнее: [docs/research_step3_ranking.md](docs/research_step3_ranking.md)

### Шаг 4 — Генерация описания

Модель: **Qwen2.5-7B-Instruct** (`torch.float16`, `device_map="auto"`).
Генерация жёстко управляется структурированным промптом: top-k признаки
раскрываются в клинические формулировки, группируются по разделам (форма →
границы → цвет → текстура), стиль стабилизируется few-shot примерами.
Запрещено: диагноз, рекомендации, информация сверх переданных признаков.

Параметры: `temperature=0.4`, `top_p=0.85`, `repetition_penalty=1.15`,
`max_new_tokens=256`.

Подробнее: [docs/research_step4_generation.md](docs/research_step4_generation.md)

---

## Инструмент ручной разметки

Мини-приложение для сбора экспертных аннотаций (Flask + vanilla JS):

```bash
python annotation/app.py \
    --csv bucketed_features.csv \
    --output annotations.csv \
    --port 5050
```

Эксперту показывается изображение и список признаков, сгруппированных по
категориям. Лимит — 10 чекбоксов (технически enforced). Навигация: Enter =
сохранить, стрелки = предыдущий/следующий. Атомарная запись через
`os.replace`, резюмирование при перезапуске.

---

## Быстрый старт

### Установка зависимостей

```bash
pip install -r requirements.txt
```

### Запуск полного пайплайна

```python
from extraction.feature_extraction_batch import extract_features_batch, images_to_df
from bucketing.feature_bucketing_batch import bucket_features_batch
from model.inference import rank_important_labels
from generation.description_inference import generate_descriptions_batch

df = images_to_df("images/")
df = extract_features_batch(df, yolo_weights="weights/mask_builder_yolo.pt",
                             unet_weights="weights/unet.pt")
df = bucket_features_batch(df)
df = rank_important_labels(df, checkpoint="model/checkpoints/best.pt")
df = generate_descriptions_batch(df)
```

### Запуск экспериментов

Основной ноутбук: [`notebooks/img2txt.ipynb`](notebooks/img2txt.ipynb)

---

## Связь с рабочим сервисом

| | Эта ветка (`main`) | Сервисная ветка (`feature-description-service`) |
|---|---|---|
| Назначение | Исследования, эксперименты, разметка | Production-сервис с HTTP API |
| Запуск | Jupyter / скрипты | Docker Compose |
| Интеграция | Автономно | Через skin-cancer-ai backend |
| Документация | `docs/research_*.md` | `docs/api.md`, `docs/architecture.md` |

---

## Документация

- [Шаг 1: извлечение признаков](docs/research_step1_extraction.md)
- [Шаг 2: бакетизация](docs/research_step2_bucketing.md)
- [Шаг 3: ранжирование](docs/research_step3_ranking.md)
- [Шаг 4: генерация](docs/research_step4_generation.md)

---

## Лицензия

См. [LICENSE](LICENSE).
