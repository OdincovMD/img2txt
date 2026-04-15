#!/usr/bin/env python3
"""
Диагностика: почему модель не учится.

Тесты:
  1. Статистика таргетов
  2. Частотный baseline (без модели вообще)
  3. Overfit на 1 батч (можно ли вообще учиться?)
  4. Градиенты
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import numpy as np

from training_v2.config import TrainConfig, LABEL_NAMES, NUM_LABELS
from training_v2.train import (
    load_data, split_data, build_vocab, create_datasets,
    create_model, set_seed, _smooth_targets, _collate_fn,
)
from training_v2.dataset import (
    build_target_vector, parse_expert_labels, debug_dataset_sample,
)
from training_v2.metrics import precision_recall_at_k


def test_target_stats(train_ds, val_ds):
    """Тест 1: статистика таргетов."""
    print("\n" + "="*70)
    print("  ТЕСТ 1: Статистика таргетов")
    print("="*70)

    for name, ds in [("Train", train_ds), ("Val", val_ds)]:
        all_targets = torch.stack([ds[i]["target"] for i in range(len(ds))])
        freq = all_targets.mean(dim=0)
        n_pos = all_targets.sum(dim=1)

        print(f"\n  {name} ({len(ds)} samples):")
        print(f"    Среднее кол-во positive labels на sample: {n_pos.mean():.1f} "
              f"(min={n_pos.min():.0f}, max={n_pos.max():.0f})")
        print(f"    Частоты по меткам:")
        for i, label in enumerate(LABEL_NAMES):
            print(f"      {label:30s}: {freq[i]:.3f} ({int(freq[i]*len(ds)):3d}/{len(ds)})")

    return all_targets, freq


def test_frequency_baseline(train_ds, val_ds):
    """Тест 2: что даёт частотный baseline (всегда предсказываем top-10 частых)."""
    print("\n" + "="*70)
    print("  ТЕСТ 2: Частотный baseline")
    print("="*70)

    # Частоты по train
    train_targets = torch.stack([train_ds[i]["target"] for i in range(len(train_ds))])
    freq = train_targets.mean(dim=0)

    # Top-10 самых частых
    top10_idx = freq.argsort(descending=True)[:10]
    top10_labels = [LABEL_NAMES[i] for i in top10_idx]
    print(f"\n  Top-10 частых меток (по train):")
    for i, idx in enumerate(top10_idx):
        print(f"    {i+1}. {LABEL_NAMES[idx]:30s} freq={freq[idx]:.3f}")

    # Создаём "логиты" baseline: freq как логиты (высокий freq = высокий логит)
    fake_logits_freq = freq.unsqueeze(0)  # (1, 24)

    # Оцениваем на val
    val_targets = torch.stack([val_ds[i]["target"] for i in range(len(val_ds))])
    fake_logits_batch = fake_logits_freq.expand(len(val_ds), -1)

    p, r = precision_recall_at_k(fake_logits_batch, val_targets, k=10)
    print(f"\n  Frequency baseline на Val:")
    print(f"    P@10={p:.4f}  R@10={r:.4f}  score={(p+r)/2:.4f}")

    # Random baseline
    rand_logits = torch.randn(len(val_ds), NUM_LABELS)
    p_r, r_r = precision_recall_at_k(rand_logits, val_targets, k=10)
    print(f"  Random baseline на Val:")
    print(f"    P@10={p_r:.4f}  R@10={r_r:.4f}  score={(p_r+r_r)/2:.4f}")


def test_overfit_one_batch(train_loader, feat_dim, train_df, cfg):
    """Тест 3: может ли модель переобучиться на 1 батче? Если нет — баг."""
    print("\n" + "="*70)
    print("  ТЕСТ 3: Overfit на 1 батч (200 шагов)")
    print("="*70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from training_v2.model import ImportanceModelV2

    model = ImportanceModelV2(
        backbone_name=cfg.backbone,
        num_labels=NUM_LABELS,
        pretrained=cfg.pretrained,
        in_channels=(4 if cfg.use_mask else 3),
        dropout=0.0,           # БЕЗ dropout для overfitting теста
        feat_input_dim=feat_dim,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(reduction="mean")  # БЕЗ pos_weight
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # Высокий lr

    # Берём 1 батч
    batch = next(iter(train_loader))
    images = batch["image"].to(device)
    feats = batch["features"].to(device)
    targets = batch["target"].to(device)

    print(f"\n  Батч: images={images.shape}, feats={feats.shape}, targets={targets.shape}")
    print(f"  Target sum per sample: {targets.sum(dim=1).tolist()}")

    model.train()
    for step in range(200):
        optimizer.zero_grad()
        logits = model(images, feats)
        loss = criterion(logits, targets)
        loss.backward()

        # Проверяем градиенты
        if step == 0:
            grad_norms = {}
            for name, p in model.named_parameters():
                if p.grad is not None:
                    grad_norms[name] = p.grad.norm().item()
            print(f"\n  Градиенты (шаг 0):")
            for name in sorted(grad_norms, key=grad_norms.get, reverse=True)[:5]:
                print(f"    {name}: grad_norm={grad_norms[name]:.6f}")
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
            print(f"  Total grad norm: {total_norm:.4f}")

        optimizer.step()

        if step % 50 == 0 or step == 199:
            with torch.no_grad():
                preds = (torch.sigmoid(logits) > 0.5).float()
                acc = (preds == targets).float().mean()
                p, r = precision_recall_at_k(logits, targets, k=10)
            print(f"  Step {step:3d}: loss={loss.item():.4f}  "
                  f"acc={acc:.4f}  P@10={p:.4f}  R@10={r:.4f}")

    print("\n  Если loss→0 и P@10→1.0 — код верный, проблема в гиперпараметрах.")
    print("  Если loss не падает — баг в модели или данных.")


def test_gradient_flow_production(train_loader, feat_dim, train_df, cfg):
    """Тест 4: градиенты в условиях production конфига."""
    print("\n" + "="*70)
    print("  ТЕСТ 4: Градиенты с production конфигом")
    print("="*70)

    model, optimizer, scheduler, criterion, device, backbone_params = create_model(
        feat_dim, train_df, cfg
    )

    batch = next(iter(train_loader))
    images = batch["image"].to(device)
    feats = batch["features"].to(device)
    targets = batch["target"].to(device)

    model.train()
    optimizer.zero_grad(set_to_none=True)

    amp_enabled = cfg.use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    with torch.amp.autocast("cuda", enabled=amp_enabled):
        logits = model(images, feats)
        smooth = _smooth_targets(targets, cfg.label_smoothing)
        loss = criterion(logits, smooth).mean()

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)

    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
    print(f"\n  Loss: {loss.item():.4f}")
    print(f"  Total gradient norm (до clip): {total_norm:.4f}")
    print(f"  Max norm в конфиге: 1.0")
    if total_norm > 1.0:
        print(f"  ⚠ Градиенты обрезаются в {total_norm:.1f}x раз!")
        print(f"    Эффективный lr = {cfg.lr} / {total_norm:.1f} = {cfg.lr/total_norm:.2e}")

    # Отдельно backbone vs head
    head_grads = []
    branch_grads = []
    backbone_grads = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            g = p.grad.norm().item()
            if "head" in name:
                head_grads.append((name, g))
            elif "feature_branch" in name:
                branch_grads.append((name, g))
            elif "backbone" in name:
                backbone_grads.append((name, g))

    print(f"\n  Head gradients:")
    for n, g in head_grads:
        print(f"    {n}: {g:.6f}")
    print(f"  Feature branch gradients:")
    for n, g in branch_grads:
        print(f"    {n}: {g:.6f}")
    if backbone_grads:
        bb_norms = [g for _, g in backbone_grads]
        print(f"  Backbone gradients: {len(backbone_grads)} params, "
              f"mean={np.mean(bb_norms):.6f}")
    else:
        print(f"  Backbone: frozen (нет градиентов)")


def main():
    cfg = TrainConfig(
        features_csv="features_dataset_bucket.csv",
        annotations_csv="annotations.csv",
        image_dir=".",
        backbone="efficientnet_b0",
        epochs=50,
        batch_size=16,
        lr=1e-4,
        image_size=224,
        use_mask=False,
        freeze_backbone_epochs=5,
        label_smoothing=0.05,
        aug_strength="mild",
        dropout=0.3,
        preload_to_memory=True,
        out_dir="training_v2/checkpoints",
    )
    set_seed(cfg.seed)

    df = load_data(cfg)
    train_df, val_df = split_data(df, cfg)
    vocab, feat_dim = build_vocab(train_df, cfg)
    train_ds, val_ds, train_loader, val_loader = create_datasets(
        train_df, val_df, vocab, cfg
    )

    # Тест 1
    test_target_stats(train_ds, val_ds)

    # Тест 2
    test_frequency_baseline(train_ds, val_ds)

    # Тест 3
    test_overfit_one_batch(train_loader, feat_dim, train_df, cfg)

    # Тест 4
    test_gradient_flow_production(train_loader, feat_dim, train_df, cfg)

    print("\n" + "="*70)
    print("  ДИАГНОСТИКА ЗАВЕРШЕНА")
    print("="*70)


if __name__ == "__main__":
    main()
