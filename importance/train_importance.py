"""
Обучение модели отбора важных признаков.
Запуск: train_importance(data_csv=..., image_dir=...) из ноутбука или скрипта.

Поддерживает:
  - freeze_backbone_epochs: сколько эпох обучать только голову (warmup)
  - label_smoothing: сглаживание меток (уменьшает overfit на шумные метки)
"""

import ctypes
import ctypes.util
import gc
import json
import random
from pathlib import Path

try:
    _libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6")
    _libc.malloc_trim.argtypes = [ctypes.c_size_t]
    _libc.malloc_trim.restype = ctypes.c_int
    def _malloc_trim():
        try:
            _libc.malloc_trim(0)
        except Exception:
            pass
except Exception:
    def _malloc_trim():
        pass

from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, default_collate
from tqdm import tqdm

try:
    from torchvision import transforms
except ImportError:
    transforms = None

from config.importance_config import FEAT_KEYS, LABEL_NAMES, NUM_LABELS
from importance.importance_dataset import ImportanceDataset, parse_expert_labels, build_target_vector
from importance.importance_metrics import precision_recall_at_k_batch
from importance.importance_model import ImportanceModel

RANDOM_STATE = 42

BackboneName = Literal["efficientnet_b0", "resnet18", "resnet34"]


def _collate_fn(batch):
    """Robust collate: tries default_collate per field, falls back to plain list."""
    result = {}
    for key in batch[0]:
        vals = [b[key] for b in batch]
        if any(isinstance(v, str) for v in vals):
            result[key] = vals
            continue
        try:
            result[key] = default_collate(vals)
        except (TypeError, RuntimeError, ValueError):
            result[key] = vals
    return result


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_transform(image_size: int = 224, is_training: bool = True, in_channels: int = 3, aug_strength: str = "mild"):
    if transforms is None:
        raise ImportError("torchvision required for transforms.")
    if aug_strength == "mild":
        crop_scale = (0.92, 1.0)
        rot_deg = 15
        cj_brightness, cj_contrast, cj_sat, cj_hue = 0.15, 0.15, 0.10, 0.0
    elif aug_strength == "strong":
        crop_scale = (0.8, 1.0)
        rot_deg = 30
        cj_brightness, cj_contrast, cj_sat, cj_hue = 0.3, 0.3, 0.2, 0.05
    else:
        raise ValueError(f"Unknown aug_strength: {aug_strength}")

    if in_channels == 3:
        if is_training:
            base = [
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(image_size, scale=crop_scale, ratio=(0.95, 1.05)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=rot_deg),
                transforms.ColorJitter(brightness=cj_brightness, contrast=cj_contrast, saturation=cj_sat, hue=cj_hue),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        else:
            base = [
                transforms.ToPILImage(),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        return transforms.Compose(base)

    from PIL import Image as PILImage

    def transform_4ch(img):
        rgb = img[..., :3]
        mask = img[..., 3:4].squeeze(-1)
        if mask.max() <= 1:
            mask = (mask * 255).astype(np.uint8)

        rgb_pil = PILImage.fromarray(rgb)
        mask_pil = PILImage.fromarray(mask)

        if is_training:
            if random.random() > 0.5:
                rgb_pil = rgb_pil.transpose(PILImage.FLIP_LEFT_RIGHT)
                mask_pil = mask_pil.transpose(PILImage.FLIP_LEFT_RIGHT)
            if random.random() > 0.5:
                rgb_pil = rgb_pil.transpose(PILImage.FLIP_TOP_BOTTOM)
                mask_pil = mask_pil.transpose(PILImage.FLIP_TOP_BOTTOM)
            angle = random.uniform(-30, 30)
            rgb_pil = rgb_pil.rotate(angle, resample=PILImage.BILINEAR)
            mask_pil = mask_pil.rotate(angle, resample=PILImage.NEAREST)
            rgb_pil = transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05
            )(rgb_pil)

        rgb_pil = rgb_pil.resize((image_size, image_size))
        mask_pil = mask_pil.resize((image_size, image_size))

        rgb_t = transforms.ToTensor()(rgb_pil)
        mask_t = torch.from_numpy(np.array(mask_pil)).float().unsqueeze(0) / 255.0
        x = torch.cat([rgb_t, mask_t], dim=0)
        x[0:3] = (x[0:3] - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) / torch.tensor(
            [0.229, 0.224, 0.225]
        ).view(3, 1, 1)
        return x

    return transform_4ch


def _smooth_targets(targets: torch.Tensor, smoothing: float) -> torch.Tensor:
    """Label smoothing: 1 → 1-ε, 0 → ε."""
    return targets * (1.0 - smoothing) + smoothing * 0.5


def compute_feat_stats(df: pd.DataFrame) -> tuple:
    """
    Вычисляет mean и std для каждого ключа FEAT_KEYS по плоским колонкам df.
    Возвращает (feat_mean, feat_std) — словари {key: float}.
    """
    accum = {k: [] for k in FEAT_KEYS}
    for k in FEAT_KEYS:
        if k not in df.columns:
            continue
        vals = pd.to_numeric(df[k], errors="coerce").dropna()
        vals = vals[np.isfinite(vals)]
        accum[k] = vals.tolist()

    feat_mean = {}
    feat_std = {}
    for k in FEAT_KEYS:
        vals = accum[k]
        if vals:
            feat_mean[k] = float(np.mean(vals))
            feat_std[k] = max(float(np.std(vals)), 1e-6)
        else:
            feat_mean[k] = 0.0
            feat_std[k] = 1.0
    return feat_mean, feat_std


def train_importance(
    data_csv: Union[str, Path],
    image_dir: Union[str, Path],
    *,
    mask_dir: Optional[Union[str, Path]] = None,
    image_col: str = "filename",
    image_id_col: str = "image_id",
    val_ratio: float = 0.15,
    splits_file: Optional[Union[str, Path]] = None,
    backbone: BackboneName = "efficientnet_b0",
    epochs: int = 50,
    batch_size: int = 16,
    lr: float = 1e-4,
    image_size: int = 224,
    use_mask: bool = False,
    freeze_backbone_epochs: int = 5,
    label_smoothing: float = 0.05,
    use_pos_weight: bool = True,
    pos_weight_cap: float = 10.0,
    aug_strength: Literal["mild", "strong"] = "mild",
    loss_type: Literal["bce", "focal"] = "bce",
    focal_gamma: float = 2.0,
    weight_decay: float = 1e-3,
    dropout: float = 0.4,
    num_workers: int = 4,
    use_amp: bool = True,
    multi_gpu: bool = True,
    preload_to_memory: bool = True,
    preload_size: int = 256,
    out_dir: Union[str, Path] = "importance_checkpoints",
    seed: int = RANDOM_STATE,
) -> float:
    """
    Обучает модель; сохраняет лучший чекпоинт в ``out_dir / best.pt``.

    Args:
        data_csv: CSV с реальной разметкой (колонки: image_id, filename,
                  important_labels, features_json).
        image_dir: Корень директории с изображениями.
        freeze_backbone_epochs: Число эпох warmup с замороженным бэкбоном.
        label_smoothing: Сглаживание меток (0.0 = выкл, 0.05–0.1 рекомендуется).
        epochs: Полное число эпох (включая freeze_backbone_epochs).

    Returns:
        Лучший score на валидации ((P@10 + R@10) / 2).
    """
    if backbone not in ("efficientnet_b0", "resnet18", "resnet34"):
        raise ValueError(f"Unknown backbone: {backbone}")

    set_seed(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_csv = Path(data_csv)
    image_dir = str(image_dir)
    mask_dir_s = str(mask_dir) if mask_dir is not None else None

    # ---- Загрузка и разбивка данных ----
    df = pd.read_csv(data_csv)
    print(f"Loaded {len(df)} samples from {data_csv}")

    if image_id_col not in df.columns and "image_id" in df.columns:
        image_id_col = "image_id"
    if image_col not in df.columns and "filename" in df.columns:
        image_col = "filename"

    if splits_file and Path(splits_file).is_file():
        with open(splits_file) as f:
            splits = json.load(f)
        if "train_idx" in splits:
            val_ids = set(df.iloc[splits["val_idx"]][image_id_col].tolist())
        else:
            val_ids = set(splits.get("val", []))
    else:
        n_val = max(1, int(len(df) * val_ratio))
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(df))
        val_ids = set(df.iloc[perm[:n_val]][image_id_col].tolist())
        splits = {"val_ids": list(val_ids)}
        with open(out_dir / "splits.json", "w") as f:
            json.dump(splits, f, indent=2)

    val_mask = df[image_id_col].isin(val_ids)
    val_df = df[val_mask].copy()
    train_df = df[~val_mask].copy()
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")

    # ---- Статистики нормализации признаков (только по train) ----
    feat_mean, feat_std = compute_feat_stats(train_df)
    print(f"Feature stats computed from {len(train_df)} train samples "
          f"({sum(1 for v in feat_mean.values() if v != 0.0)} non-zero means)")

    # ---- Трансформы ----
    in_channels = 4 if use_mask else 3
    transform_train = get_transform(image_size, is_training=True, in_channels=in_channels, aug_strength=aug_strength)
    transform_val = get_transform(image_size, is_training=False, in_channels=in_channels, aug_strength=aug_strength)
    print(f"Augmentation: {aug_strength}")

    ds_kwargs = dict(
        image_col=image_col,
        image_id_col=image_id_col,
        feat_mean=feat_mean,
        feat_std=feat_std,
        mask_dir=mask_dir_s,
        use_mask_channel=use_mask,
        preload_to_memory=preload_to_memory,
        preload_size=preload_size,
    )
    train_ds = ImportanceDataset(train_df, image_dir=image_dir, transform=transform_train, **ds_kwargs)
    val_ds = ImportanceDataset(val_df, image_dir=image_dir, transform=transform_val, **ds_kwargs)

    if preload_to_memory and num_workers > 0:
        print("preload_to_memory=True → forcing num_workers=0")
        num_workers = 0

    pin_memory = num_workers > 0
    loader_kwargs = dict(
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_collate_fn,
        persistent_workers=(num_workers > 0),
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 4

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)

    # ---- Модель ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImportanceModel(
        backbone_name=backbone,
        num_labels=NUM_LABELS,
        pretrained=True,
        in_channels=in_channels,
        dropout=dropout,
    ).to(device)

    # ---- pos_weight по частоте меток в train ----
    pos_weight_tensor = None
    if use_pos_weight:
        n = len(train_df)
        if n > 0:
            pos_counts = torch.zeros(NUM_LABELS, dtype=torch.float32)
            for _, row in train_df.iterrows():
                strs = parse_expert_labels(row, None)
                t = build_target_vector(strs)
                pos_counts += t
            neg_counts = n - pos_counts
            raw_pw = neg_counts / pos_counts.clamp(min=1.0)
            pos_weight_tensor = raw_pw.clamp(max=pos_weight_cap).to(device)
            print(f"pos_weight: min={pos_weight_tensor.min():.2f} "
                  f"max={pos_weight_tensor.max():.2f} "
                  f"mean={pos_weight_tensor.mean():.2f} (capped at {pos_weight_cap})")

    if loss_type == "bce":
        criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight_tensor)
    elif loss_type == "focal":
        def criterion(logits, targets):
            bce = nn.functional.binary_cross_entropy_with_logits(
                logits, targets, reduction="none", pos_weight=pos_weight_tensor
            )
            p = torch.sigmoid(logits)
            p_t = p * targets + (1 - p) * (1 - targets)
            focal_w = (1 - p_t).pow(focal_gamma)
            return focal_w * bce
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    head_params = list(model.head.parameters()) + list(model.feature_branch.parameters())
    backbone_params = list(model.backbone.parameters())
    optimizer = torch.optim.AdamW([
        {"params": head_params, "lr": lr},
        {"params": backbone_params, "lr": lr * 0.1},
    ], weight_decay=weight_decay)

    def set_backbone_grad(requires_grad: bool):
        for p in backbone_params:
            p.requires_grad = requires_grad

    if freeze_backbone_epochs > 0:
        set_backbone_grad(False)
        print(f"Backbone frozen for first {freeze_backbone_epochs} epochs (warmup)")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if multi_gpu and n_gpus > 1:
        model = nn.DataParallel(model)
        print(f"Using DataParallel on {n_gpus} GPUs")
    else:
        print(f"Using single device: {device}")

    amp_enabled = use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    if amp_enabled:
        print("Mixed precision (AMP) enabled")

    torch.backends.cudnn.benchmark = True

    run_config = {
        "data_csv": str(data_csv.resolve()),
        "image_dir": image_dir,
        "mask_dir": mask_dir_s,
        "backbone": backbone,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "image_size": image_size,
        "use_mask": use_mask,
        "freeze_backbone_epochs": freeze_backbone_epochs,
        "label_smoothing": label_smoothing,
        "out_dir": str(out_dir),
        "seed": seed,
    }

    best_score = 0.0
    non_blocking = pin_memory

    for epoch in range(epochs):
        if freeze_backbone_epochs > 0 and epoch == freeze_backbone_epochs:
            set_backbone_grad(True)
            for p in backbone_params:
                optimizer.state.pop(p, None)
            print(f"Epoch {epoch+1}: backbone unfrozen, Adam state reset")

        model.train()
        running_loss = 0.0
        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)):
            images = batch["image"].to(device, non_blocking=non_blocking)
            feats = batch["features"].to(device, non_blocking=non_blocking)
            targets = batch["target"].to(device, non_blocking=non_blocking)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = model(images, feats)
                smooth_targets = _smooth_targets(targets, label_smoothing)
                loss = criterion(logits, smooth_targets).mean()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

            del images, feats, targets, logits, smooth_targets, loss, batch
            if (step + 1) % 50 == 0:
                gc.collect()
                _malloc_trim()

        gc.collect()
        _malloc_trim()
        scheduler.step()
        train_loss = running_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        all_logits = []
        all_targets = []
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device, non_blocking=non_blocking)
                feats = batch["features"].to(device, non_blocking=non_blocking)
                targets = batch["target"].to(device, non_blocking=non_blocking)
                with torch.amp.autocast("cuda", enabled=amp_enabled):
                    logits = model(images, feats)
                    loss = criterion(logits, targets).mean()
                val_loss += loss.item()
                all_logits.append(logits.float().cpu())
                all_targets.append(targets.cpu())
                del images, feats, targets, logits, loss, batch

        val_loss /= len(val_loader)
        logits_cat = torch.cat(all_logits, dim=0)
        targets_cat = torch.cat(all_targets, dim=0)
        del all_logits, all_targets
        gc.collect()
        _malloc_trim()

        prec10, rec10 = precision_recall_at_k_batch(logits_cat, targets_cat, k=10)
        score = (prec10 + rec10) / 2.0

        backbone_frozen = epoch < freeze_backbone_epochs
        print(
            f"Epoch {epoch+1}{'*' if backbone_frozen else ' '} "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"P@10={prec10:.4f}  R@10={rec10:.4f}  score={score:.4f}"
        )

        if score > best_score:
            best_score = score
            state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            ckpt = {
                "epoch": epoch + 1,
                "model_state_dict": state_dict,
                "optimizer_state_dict": optimizer.state_dict(),
                "score": score,
                "prec10": prec10,
                "rec10": rec10,
                "args": run_config,
                "label_names": LABEL_NAMES,
                "feat_stats": {"mean": feat_mean, "std": feat_std},
            }
            torch.save(ckpt, out_dir / "best.pt")
            print(f"  -> saved best.pt (score={score:.4f})")

    print(f"Done. Best score: {best_score:.4f}. Checkpoints in {out_dir}")
    return float(best_score)


if __name__ == "__main__":
    train_importance(
        data_csv="features_dataset_for_importance.csv",
        image_dir=".",
    )
