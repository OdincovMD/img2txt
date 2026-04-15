"""
Обучение модели важности признаков v2.

Каждый этап отделён функцией для дебага:
  1. load_data()       — загрузка и merge CSV
  2. split_data()      — train/val split
  3. build_vocab()     — словарь one-hot
  4. create_datasets() — PyTorch Dataset'ы
  5. create_model()    — модель + оптимизатор + лосс
  6. train_loop()      — цикл обучения
"""

import gc
import json
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, default_collate
from tqdm import tqdm

from training_v2.config import LABEL_NAMES, NUM_LABELS, TrainConfig
from training_v2.dataset import (
    ImportanceDatasetV2,
    build_label_vocab,
    build_target_vector,
    parse_expert_labels,
    vocab_total_dim,
    prepare_dataframe,
)
from training_v2.metrics import precision_recall_at_k
from training_v2.model import ImportanceModelV2
from training_v2.transforms import get_transforms


# ══════════════════════════════════════════════════════════════════════════════
# Утилиты
# ══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _collate_fn(batch):
    result = {}
    for key in batch[0]:
        vals = [b[key] for b in batch]
        if isinstance(vals[0], str):
            result[key] = vals
            continue
        try:
            result[key] = default_collate(vals)
        except (TypeError, RuntimeError, ValueError):
            result[key] = vals
    return result


def _smooth_targets(t: torch.Tensor, eps: float) -> torch.Tensor:
    return t * (1.0 - eps) + eps * 0.5


# ══════════════════════════════════════════════════════════════════════════════
# 1. Загрузка данных
# ══════════════════════════════════════════════════════════════════════════════

def load_data(cfg: TrainConfig):
    """Merge features + annotations → DataFrame."""
    import pandas as pd
    df = prepare_dataframe(cfg.features_csv, cfg.annotations_csv)
    print(f"\n[1/6 load_data] OK — {len(df)} samples")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. Разбивка train / val
# ══════════════════════════════════════════════════════════════════════════════

def split_data(df, cfg: TrainConfig):
    """Случайная стратифицированная разбивка."""
    import pandas as pd
    n_val = max(1, int(len(df) * cfg.val_ratio))
    rng = np.random.RandomState(cfg.seed)
    perm = rng.permutation(len(df))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()

    # Сохраняем сплиты
    out_dir = Path(cfg.out_dir)
    splits = {
        "train_idx": train_idx.tolist(),
        "val_idx": val_idx.tolist(),
    }
    with open(out_dir / "splits.json", "w") as f:
        json.dump(splits, f, indent=2)

    print(f"[2/6 split_data] Train: {len(train_df)}, Val: {len(val_df)}")
    return train_df, val_df


# ══════════════════════════════════════════════════════════════════════════════
# 3. Словарь one-hot
# ══════════════════════════════════════════════════════════════════════════════

def build_vocab(train_df, cfg: TrainConfig):
    """Строит one-hot словарь по train-выборке."""
    vocab = build_label_vocab(train_df)
    feat_dim = vocab_total_dim(vocab)
    print(f"[3/6 build_vocab] {len(vocab)} keys, total one-hot dim = {feat_dim}")
    for k in LABEL_NAMES:
        n_vals = len(vocab.get(k, {}))
        if n_vals > 0:
            print(f"  {k}: {n_vals} values")
    return vocab, feat_dim


# ══════════════════════════════════════════════════════════════════════════════
# 4. Создание Dataset + DataLoader
# ══════════════════════════════════════════════════════════════════════════════

def create_datasets(train_df, val_df, vocab, cfg: TrainConfig):
    """Собирает Dataset'ы и DataLoader'ы."""
    in_ch = 4 if cfg.use_mask else 3
    t_train = get_transforms(cfg.image_size, True, in_ch, cfg.aug_strength)
    t_val = get_transforms(cfg.image_size, False, in_ch, cfg.aug_strength)

    ds_kwargs = dict(
        image_dir=cfg.image_dir,
        vocab=vocab,
        use_mask=cfg.use_mask,
        mask_dir=cfg.mask_dir,
        mask_suffix=cfg.mask_suffix,
        preload=cfg.preload_to_memory,
        preload_size=cfg.preload_size,
    )
    train_ds = ImportanceDatasetV2(train_df, transform=t_train, **ds_kwargs)
    val_ds = ImportanceDatasetV2(val_df, transform=t_val, **ds_kwargs)

    nw = 0 if cfg.preload_to_memory else cfg.num_workers
    loader_kw = dict(
        num_workers=nw,
        pin_memory=(nw > 0),
        collate_fn=_collate_fn,
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, **loader_kw
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, **loader_kw
    )

    print(f"[4/6 create_datasets] Train batches: {len(train_loader)}, "
          f"Val batches: {len(val_loader)}")
    return train_ds, val_ds, train_loader, val_loader


# ══════════════════════════════════════════════════════════════════════════════
# 5. Модель + оптимизатор + лосс
# ══════════════════════════════════════════════════════════════════════════════

def create_model(feat_dim: int, train_df, cfg: TrainConfig):
    """Создаёт модель, оптимизатор, лосс-функцию."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_ch = 4 if cfg.use_mask else 3
    skip_image = getattr(cfg, 'skip_image', False)

    model = ImportanceModelV2(
        backbone_name=cfg.backbone,
        num_labels=NUM_LABELS,
        pretrained=cfg.pretrained,
        in_channels=in_ch,
        dropout=cfg.dropout,
        feat_input_dim=feat_dim,
        skip_image=skip_image,
    ).to(device)

    # pos_weight
    pos_weight_tensor = None
    if cfg.use_pos_weight:
        pos_counts = torch.zeros(NUM_LABELS, dtype=torch.float32)
        for _, row in train_df.iterrows():
            strs = parse_expert_labels(row.get("important_labels"))
            t = build_target_vector(strs)
            pos_counts += t
        neg_counts = len(train_df) - pos_counts
        raw_pw = neg_counts / pos_counts.clamp(min=1.0)
        pos_weight_tensor = raw_pw.clamp(max=cfg.pos_weight_cap).to(device)
        print(f"  pos_weight: min={pos_weight_tensor.min():.2f}, "
              f"max={pos_weight_tensor.max():.2f}, "
              f"mean={pos_weight_tensor.mean():.2f}")

    # Лосс
    if cfg.loss_type == "bce":
        criterion = nn.BCEWithLogitsLoss(
            reduction="none", pos_weight=pos_weight_tensor
        )
    elif cfg.loss_type == "focal":
        def criterion(logits, targets):
            bce = nn.functional.binary_cross_entropy_with_logits(
                logits, targets, reduction="none", pos_weight=pos_weight_tensor
            )
            p = torch.sigmoid(logits)
            p_t = p * targets + (1 - p) * (1 - targets)
            return (1 - p_t).pow(cfg.focal_gamma) * bce
    else:
        raise ValueError(f"Unknown loss_type: {cfg.loss_type}")

    # Оптимизатор: группы параметров с разным lr
    param_groups = []
    head_params = list(model.head.parameters())
    param_groups.append({"params": head_params, "lr": cfg.lr})

    backbone_params = []
    if not skip_image:
        img_proj_params = list(model.img_projection.parameters())
        backbone_params = list(model.backbone.parameters())
        param_groups.append({"params": img_proj_params, "lr": cfg.lr})
        param_groups.append({"params": backbone_params, "lr": cfg.lr * cfg.backbone_lr_factor})

    if model.feature_branch is not None:
        branch_params = list(model.feature_branch.parameters())
        param_groups.append(
            {"params": branch_params, "lr": cfg.lr * cfg.feature_branch_lr_factor}
        )

    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs
    )

    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    mode = "TABULAR-ONLY" if skip_image else cfg.backbone
    print(f"[5/6 create_model] {mode} on {device}")
    print(f"  Parameters: {total_p:,} total, {train_p:,} trainable")
    print(f"  feat_input_dim={feat_dim}, in_channels={in_ch}, skip_image={skip_image}")

    return model, optimizer, scheduler, criterion, device, backbone_params


# ══════════════════════════════════════════════════════════════════════════════
# 6. Цикл обучения
# ══════════════════════════════════════════════════════════════════════════════

def train_loop(
    model: ImportanceModelV2,
    optimizer,
    scheduler,
    criterion,
    train_loader: DataLoader,
    val_loader: DataLoader,
    backbone_params: list,
    vocab: dict,
    feat_dim: int,
    device: torch.device,
    cfg: TrainConfig,
) -> float:
    """Основной цикл. Возвращает лучший score."""

    amp_enabled = cfg.use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    if amp_enabled:
        print("  AMP enabled")
    torch.backends.cudnn.benchmark = True

    # Заморозка бэкбона
    def set_backbone_grad(on: bool):
        for p in backbone_params:
            p.requires_grad = on

    if cfg.freeze_backbone_epochs > 0:
        set_backbone_grad(False)
        print(f"  Backbone frozen for first {cfg.freeze_backbone_epochs} epochs")

    best_score = 0.0
    epochs_no_improve = 0
    run_config = {
        "backbone": cfg.backbone,
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "image_size": cfg.image_size,
        "use_mask": cfg.use_mask,
        "freeze_backbone_epochs": cfg.freeze_backbone_epochs,
        "label_smoothing": cfg.label_smoothing,
        "aug_strength": cfg.aug_strength,
        "loss_type": cfg.loss_type,
        "seed": cfg.seed,
    }

    print(f"\n[6/6 train_loop] Starting training for {cfg.epochs} epochs\n")

    for epoch in range(cfg.epochs):
        # Разморозка
        if cfg.freeze_backbone_epochs > 0 and epoch == cfg.freeze_backbone_epochs:
            set_backbone_grad(True)
            for p in backbone_params:
                optimizer.state.pop(p, None)
            # Сбрасываем scheduler на оставшиеся эпохи
            remaining = cfg.epochs - epoch
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=remaining
            )
            print(f"  ── Epoch {epoch+1}: backbone unfrozen, scheduler reset for {remaining} epochs ──")

        # ── Train ──
        model.train()
        running_loss = 0.0
        for batch in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs} [train]", leave=False
        ):
            images = batch["image"].to(device)
            feats = batch["features"].to(device)
            targets = batch["target"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = model(images, feats)
                smooth = _smooth_targets(targets, cfg.label_smoothing)
                loss = criterion(logits, smooth).mean()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip_max_norm)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        scheduler.step()
        train_loss = running_loss / max(len(train_loader), 1)

        # ── Val ──
        model.eval()
        val_loss_sum = 0.0
        all_logits, all_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                feats = batch["features"].to(device)
                targets = batch["target"].to(device)
                with torch.amp.autocast("cuda", enabled=amp_enabled):
                    logits = model(images, feats)
                    loss = criterion(logits, targets).mean()
                val_loss_sum += loss.item()
                all_logits.append(logits.float().cpu())
                all_targets.append(targets.cpu())

        val_loss = val_loss_sum / max(len(val_loader), 1)
        logits_cat = torch.cat(all_logits)
        targets_cat = torch.cat(all_targets)

        prec10, rec10 = precision_recall_at_k(logits_cat, targets_cat, k=10)
        score = (prec10 + rec10) / 2.0

        frozen = epoch < cfg.freeze_backbone_epochs
        marker = "*" if frozen else " "
        print(
            f"Epoch {epoch+1:3d}{marker}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"P@10={prec10:.4f}  R@10={rec10:.4f}  score={score:.4f}"
        )

        # Сохранение лучшего
        if score > best_score:
            best_score = score
            out_dir = Path(cfg.out_dir)
            state = (
                model.module.state_dict()
                if isinstance(model, nn.DataParallel)
                else model.state_dict()
            )
            ckpt = {
                "epoch": epoch + 1,
                "model_state_dict": state,
                "optimizer_state_dict": optimizer.state_dict(),
                "score": score,
                "prec10": prec10,
                "rec10": rec10,
                "args": run_config,
                "label_names": LABEL_NAMES,
                "vocab": vocab,
                "feat_input_dim": feat_dim,
            }
            torch.save(ckpt, out_dir / "best.pt")
            print(f"  → saved best.pt (score={score:.4f})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.early_stopping_patience:
                print(f"\n  Early stopping: {epochs_no_improve} epochs без улучшения")
                break

        gc.collect()

    print(f"\nDone. Best score: {best_score:.4f}")
    return best_score


# ══════════════════════════════════════════════════════════════════════════════
# Главная функция
# ══════════════════════════════════════════════════════════════════════════════

def train(cfg: TrainConfig = None) -> float:
    """
    Полный пайплайн обучения — вызывается из run.py или ноутбука.
    """
    if cfg is None:
        cfg = TrainConfig()

    set_seed(cfg.seed)

    # Пошагово — каждый шаг можно дебажить отдельно
    df = load_data(cfg)
    train_df, val_df = split_data(df, cfg)
    vocab, feat_dim = build_vocab(train_df, cfg)
    train_ds, val_ds, train_loader, val_loader = create_datasets(
        train_df, val_df, vocab, cfg
    )
    model, optimizer, scheduler, criterion, device, backbone_params = create_model(
        feat_dim, train_df, cfg
    )

    return train_loop(
        model, optimizer, scheduler, criterion,
        train_loader, val_loader,
        backbone_params, vocab, feat_dim, device, cfg,
    )
