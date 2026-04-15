#!/usr/bin/env python3
"""
Точка входа для обучения v2.

Можно запускать целиком:
    python -m training_v2.run

Или пошагово для дебага — раскомментируй нужные блоки.
"""

import sys
from pathlib import Path

# Добавляем корень проекта в sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from training_v2.config import TrainConfig
from training_v2.train import (
    load_data,
    split_data,
    build_vocab,
    create_datasets,
    create_model,
    train_loop,
    set_seed,
)
from training_v2.dataset import debug_dataset_sample
from training_v2.model import debug_model


def main():
    # ── Конфигурация ──────────────────────────────────────────────────────
    cfg = TrainConfig(
        features_csv="features_dataset_bucket.csv",
        annotations_csv="annotations.csv",
        image_dir=".",
        backbone="efficientnet_b0",
        epochs=80,
        batch_size=16,
        lr=1e-3,
        image_size=224,
        use_mask=False,
        freeze_backbone_epochs=10,    # было 25 — head переобучалась
        label_smoothing=0.05,         # помогает от overfit на шумных метках
        aug_strength="medical",       # ColorJitter + RandomErasing (без flip/crop)
        dropout=0.3,                  # было 0.1 — усиливаем регуляризацию
        use_pos_weight=False,
        early_stopping_patience=15,
        preload_to_memory=True,
        out_dir="training_v2/checkpoints",
    )
    set_seed(cfg.seed)

    # ── Шаг 1: Загрузка данных ───────────────────────────────────────────
    print("\n" + "="*70)
    print("  ШАГ 1: Загрузка и объединение данных")
    print("="*70)
    df = load_data(cfg)
    print(f"  Колонки: {list(df.columns)[:10]}...")
    print(f"  Первые image_path: {df['image_path'].iloc[:3].tolist()}")
    # input("  [Enter] для продолжения...")   # ← раскомментируй для дебага

    # ── Шаг 2: Разбивка train/val ────────────────────────────────────────
    print("\n" + "="*70)
    print("  ШАГ 2: Разбивка train / val")
    print("="*70)
    train_df, val_df = split_data(df, cfg)
    # input("  [Enter] для продолжения...")

    # ── Шаг 3: Словарь one-hot ───────────────────────────────────────────
    print("\n" + "="*70)
    print("  ШАГ 3: Построение словаря one-hot")
    print("="*70)
    vocab, feat_dim = build_vocab(train_df, cfg)
    # input("  [Enter] для продолжения...")

    # ── Шаг 4: Создание Dataset'ов ───────────────────────────────────────
    print("\n" + "="*70)
    print("  ШАГ 4: Создание Dataset и DataLoader")
    print("="*70)
    train_ds, val_ds, train_loader, val_loader = create_datasets(
        train_df, val_df, vocab, cfg
    )

    # Дебаг одного элемента
    debug_dataset_sample(train_ds, idx=0)
    # input("  [Enter] для продолжения...")

    # ── Шаг 5: Создание модели ───────────────────────────────────────────
    print("\n" + "="*70)
    print("  ШАГ 5: Создание модели")
    print("="*70)
    model, optimizer, scheduler, criterion, device, backbone_params = create_model(
        feat_dim, train_df, cfg
    )

    # Дебаг модели
    debug_model(model, in_channels=(4 if cfg.use_mask else 3), feat_dim=feat_dim)
    # input("  [Enter] для продолжения...")

    # ── Шаг 6: Обучение ─────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  ШАГ 6: Обучение")
    print("="*70)
    best_score = train_loop(
        model, optimizer, scheduler, criterion,
        train_loader, val_loader,
        backbone_params, vocab, feat_dim, device, cfg,
    )

    print(f"\n{'='*70}")
    print(f"  ГОТОВО. Лучший score: {best_score:.4f}")
    print(f"  Чекпоинт: {cfg.out_dir}/best.pt")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
