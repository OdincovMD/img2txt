"""
Обучение модели отбора важных признаков.
Запуск: train_importance(data_csv=..., image_dir=...) из ноутбука или скрипта.

Поддерживает смешивание реальной разметки (500+) с псевдо-разметкой:
  - pseudo_csv: CSV с псевдо-метками (те же колонки, что и data_csv)
  - pseudo_weight: вес псевдо-примеров в функции потерь (по умолчанию 0.3)
  - freeze_backbone_epochs: сколько эпох обучать только голову (warmup)
  - label_smoothing: сглаживание меток (уменьшает overfit на шумные метки)
"""

import json
import random
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler, default_collate
from tqdm import tqdm

try:
    from torchvision import transforms
except ImportError:
    transforms = None

from config.importance_config import LABEL_NAMES, NUM_LABELS
from importance.importance_dataset import ImportanceDataset
from importance.importance_metrics import precision_recall_at_k_batch
from importance.importance_model import ImportanceModel

RANDOM_STATE = 42

BackboneName = Literal["efficientnet_b0", "resnet18", "resnet34"]


def _collate_fn(batch):
    """Robust collate: tries default_collate per field, falls back to plain list."""
    result = {}
    for key in batch[0]:
        vals = [b[key] for b in batch]
        # Для строковых полей сразу пропускаем default_collate
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


def get_transform(image_size: int = 224, is_training: bool = True, in_channels: int = 3):
    if transforms is None:
        raise ImportError("torchvision required for transforms.")
    if in_channels == 3:
        if is_training:
            base = [
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=30),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
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
            # Spatial transforms — применяем одинаково к RGB и маске
            if random.random() > 0.5:
                rgb_pil = rgb_pil.transpose(PILImage.FLIP_LEFT_RIGHT)
                mask_pil = mask_pil.transpose(PILImage.FLIP_LEFT_RIGHT)
            if random.random() > 0.5:
                rgb_pil = rgb_pil.transpose(PILImage.FLIP_TOP_BOTTOM)
                mask_pil = mask_pil.transpose(PILImage.FLIP_TOP_BOTTOM)
            angle = random.uniform(-30, 30)
            rgb_pil = rgb_pil.rotate(angle, resample=PILImage.BILINEAR)
            mask_pil = mask_pil.rotate(angle, resample=PILImage.NEAREST)
            # ColorJitter только для RGB
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


def train_importance(
    data_csv: Union[str, Path],
    image_dir: Union[str, Path],
    *,
    pseudo_csv: Optional[Union[str, Path]] = None,
    pseudo_weight: float = 0.3,
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
    out_dir: Union[str, Path] = "importance_checkpoints",
    seed: int = RANDOM_STATE,
) -> float:
    """
    Обучает модель; сохраняет лучший чекпоинт в ``out_dir / best.pt``.

    Args:
        data_csv: CSV с реальной разметкой (колонки: image_id, filename, important_labels).
        image_dir: Корень директории с изображениями.
        pseudo_csv: CSV с псевдо-разметкой (те же колонки). Будет смешан с data_csv.
        pseudo_weight: Вес псевдо-примеров в функции потерь (0..1). Реальные всегда 1.0.
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

    # ---- Загрузка данных ----
    df_real = pd.read_csv(data_csv)
    df_real["_is_pseudo"] = False

    if pseudo_csv is not None:
        df_pseudo = pd.read_csv(pseudo_csv)
        df_pseudo["_is_pseudo"] = True
        df_all = pd.concat([df_real, df_pseudo], ignore_index=True)
        print(f"Real: {len(df_real)}, Pseudo: {len(df_pseudo)}, Total: {len(df_all)}")
    else:
        df_all = df_real.copy()
        print(f"Real: {len(df_real)} (no pseudo-labels)")

    # Нормализуем имена колонок
    if image_id_col not in df_all.columns and "image_id" in df_all.columns:
        image_id_col = "image_id"
    if image_col not in df_all.columns and "filename" in df_all.columns:
        image_col = "filename"

    # Валидация — только из реальной разметки
    real_mask = ~df_all["_is_pseudo"]
    df_real_only = df_all[real_mask].copy()

    if splits_file and Path(splits_file).is_file():
        with open(splits_file) as f:
            splits = json.load(f)
        if "train_idx" in splits:
            val_real_ids = set(df_real_only.iloc[splits["val_idx"]][image_id_col].tolist())
        else:
            val_real_ids = set(splits.get("val", []))
    else:
        n_real = len(df_real_only)
        n_val = max(1, int(n_real * val_ratio))
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n_real)
        val_ids_local = perm[:n_val]
        val_real_ids = set(df_real_only.iloc[val_ids_local][image_id_col].tolist())
        splits = {
            "val_ids": list(val_real_ids),
            "note": "val is always from real-labeled data only",
        }
        with open(out_dir / "splits.json", "w") as f:
            json.dump(splits, f, indent=2)

    val_mask = df_all[image_id_col].isin(val_real_ids)
    val_df = df_all[val_mask & real_mask].copy()
    train_df = df_all[~val_mask].copy()

    print(f"Train: {len(train_df)} ({(~train_df['_is_pseudo']).sum()} real + {train_df['_is_pseudo'].sum()} pseudo)")
    print(f"Val:   {len(val_df)} (real only)")

    # Веса для взвешенной функции потерь
    train_weights = torch.tensor(
        [pseudo_weight if p else 1.0 for p in train_df["_is_pseudo"].tolist()],
        dtype=torch.float32,
    )

    # ---- Трансформы ----
    in_channels = 4 if use_mask else 3
    transform_train = get_transform(image_size, is_training=True, in_channels=in_channels)
    transform_val = get_transform(image_size, is_training=False, in_channels=in_channels)

    train_ds = ImportanceDataset(
        train_df,
        image_dir=image_dir,
        image_col=image_col,
        image_id_col=image_id_col,
        mask_dir=mask_dir_s,
        transform=transform_train,
        use_mask_channel=use_mask,
    )
    val_ds = ImportanceDataset(
        val_df,
        image_dir=image_dir,
        image_col=image_col,
        image_id_col=image_id_col,
        mask_dir=mask_dir_s,
        transform=transform_val,
        use_mask_channel=use_mask,
    )

    # WeightedRandomSampler: реальные примеры семплируются чаще
    sampler_weights = [1.0 if not p else pseudo_weight for p in train_df["_is_pseudo"].tolist()]
    sampler = WeightedRandomSampler(
        weights=sampler_weights,
        num_samples=len(train_ds),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=0, pin_memory=True, collate_fn=_collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, collate_fn=_collate_fn,
    )

    # ---- Модель ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImportanceModel(
        backbone_name=backbone,
        num_labels=NUM_LABELS,
        pretrained=True,
        in_channels=in_channels,
        dropout=0.3,
    ).to(device)

    # BCE без редукции — для взвешивания по примерам
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    # Два набора параметров: голова и бэкбон (разные lr)
    head_params = list(model.head.parameters())
    backbone_params = list(model.backbone.parameters())
    optimizer = torch.optim.AdamW([
        {"params": head_params, "lr": lr},
        {"params": backbone_params, "lr": lr * 0.1},
    ], weight_decay=1e-4)

    # Freeze backbone на первые freeze_backbone_epochs эпох
    def set_backbone_grad(requires_grad: bool):
        for p in backbone_params:
            p.requires_grad = requires_grad

    if freeze_backbone_epochs > 0:
        set_backbone_grad(False)
        print(f"Backbone frozen for first {freeze_backbone_epochs} epochs (warmup)")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_weights = train_weights.to(device)

    run_config = {
        "data_csv": str(data_csv.resolve()),
        "pseudo_csv": str(pseudo_csv) if pseudo_csv else None,
        "pseudo_weight": pseudo_weight,
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
    for epoch in range(epochs):
        # Размораживаем бэкбон после warmup
        if freeze_backbone_epochs > 0 and epoch == freeze_backbone_epochs:
            set_backbone_grad(True)
            print(f"Epoch {epoch+1}: backbone unfrozen")

        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            images = batch["image"].to(device)
            targets = batch["target"].to(device)
            idx = batch["idx"]
            w = train_weights[idx].unsqueeze(1)  # (B, 1) — broadcast по меткам

            optimizer.zero_grad()
            logits = model(images)

            smooth_targets = _smooth_targets(targets, label_smoothing)
            loss_mat = criterion(logits, smooth_targets)  # (B, num_labels)
            loss = (loss_mat * w).mean()

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()
        train_loss = running_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        all_logits = []
        all_targets = []
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                targets = batch["target"].to(device)
                logits = model(images)
                loss = criterion(logits, targets).mean()
                val_loss += loss.item()
                all_logits.append(logits)
                all_targets.append(targets)
        val_loss /= len(val_loader)
        logits_cat = torch.cat(all_logits, dim=0)
        targets_cat = torch.cat(all_targets, dim=0)
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
            ckpt = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "score": score,
                "prec10": prec10,
                "rec10": rec10,
                "args": run_config,
                "label_names": LABEL_NAMES,
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
