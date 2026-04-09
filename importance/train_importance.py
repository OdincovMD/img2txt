"""
Обучение модели отбора важных признаков.
Запуск: train_importance(data_csv=..., image_dir=...) из ноутбука или скрипта.

Поддерживает смешивание реальной разметки (500+) с псевдо-разметкой:
  - pseudo_csv: CSV с псевдо-метками (те же колонки, что и data_csv)
  - pseudo_weight: вес псевдо-примеров в функции потерь (по умолчанию 0.3)
  - freeze_backbone_epochs: сколько эпох обучать только голову (warmup)
  - label_smoothing: сглаживание меток (уменьшает overfit на шумные метки)
"""

import ctypes
import ctypes.util
import gc
import json
import random
from pathlib import Path

# Принудительный возврат свободных страниц glibc обратно ОС.
# Pillow/numpy аллоцируют через malloc(); glibc по умолчанию НЕ возвращает
# страницы после free() — RSS монотонно растёт. malloc_trim(0) лечит это.
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
from torch.utils.data import DataLoader, WeightedRandomSampler, default_collate
from tqdm import tqdm

try:
    from torchvision import transforms
except ImportError:
    transforms = None

from config.importance_config import LABEL_NAMES, NUM_LABELS
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


def get_transform(image_size: int = 224, is_training: bool = True, in_channels: int = 3, aug_strength: str = "mild"):
    if transforms is None:
        raise ImportError("torchvision required for transforms.")
    # Параметры аугментации по силе
    # mild: безопасно для дерматоскопии (не трогает цвет, не режет границы)
    # strong: исходный агрессивный вариант
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
    pseudo_subsample: Optional[int] = None,
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
        if pseudo_subsample is not None and pseudo_subsample < len(df_pseudo):
            df_pseudo = df_pseudo.sample(
                n=pseudo_subsample, random_state=seed
            ).reset_index(drop=True)
            print(f"Pseudo subsampled to {len(df_pseudo)} rows (random_state={seed})")
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
    transform_train = get_transform(image_size, is_training=True, in_channels=in_channels, aug_strength=aug_strength)
    transform_val = get_transform(image_size, is_training=False, in_channels=in_channels, aug_strength=aug_strength)
    print(f"Augmentation: {aug_strength}")

    train_ds = ImportanceDataset(
        train_df,
        image_dir=image_dir,
        image_col=image_col,
        image_id_col=image_id_col,
        mask_dir=mask_dir_s,
        transform=transform_train,
        use_mask_channel=use_mask,
        preload_to_memory=preload_to_memory,
        preload_size=preload_size,
    )
    val_ds = ImportanceDataset(
        val_df,
        image_dir=image_dir,
        image_col=image_col,
        image_id_col=image_id_col,
        mask_dir=mask_dir_s,
        transform=transform_val,
        use_mask_channel=use_mask,
        preload_to_memory=preload_to_memory,
        preload_size=preload_size,
    )

    # WeightedRandomSampler: подбираем веса так, чтобы реальные занимали
    # фиксированную долю батча (real_fraction_in_batch), независимо от объёма псевдо.
    # Это критично для маленьких real-выборок: иначе градиент тонет в псевдо.
    n_real_train = int((~train_df["_is_pseudo"]).sum())
    n_pseudo_train = int(train_df["_is_pseudo"].sum())
    real_fraction_in_batch = 0.5  # 50/50 в батче
    if n_pseudo_train > 0 and n_real_train > 0:
        # Per-sample вес: для real ставим 1/n_real, для pseudo — так, чтобы
        # суммарная вероятность псевдо-класса равнялась (1 - real_fraction)
        w_real = real_fraction_in_batch / n_real_train
        w_pseudo = (1.0 - real_fraction_in_batch) / n_pseudo_train
        sampler_weights = [w_real if not p else w_pseudo for p in train_df["_is_pseudo"].tolist()]
        # Размер эпохи — n_real * 4 (каждый реальный пример в среднем 2 раза за эпоху)
        epoch_size = n_real_train * 4
    else:
        sampler_weights = [1.0] * len(train_df)
        epoch_size = len(train_df)
    sampler = WeightedRandomSampler(
        weights=sampler_weights,
        num_samples=epoch_size,
        replacement=True,
    )
    print(f"Sampler: epoch_size={epoch_size}, real_fraction_in_batch={real_fraction_in_batch}")

    # При preload данные уже в RAM — воркеры не нужны и только дублируют кэш через fork-COW.
    if preload_to_memory and num_workers > 0:
        print(f"preload_to_memory=True → forcing num_workers=0 (avoid fork-copy of cache)")
        num_workers = 0

    # pin_memory с num_workers=0 в текущих версиях PyTorch может приводить
    # к постепенной утечке хост-буферов (особенно с non_blocking=True).
    # При воркерах=0 пиннинг почти не даёт прироста — отключаем.
    pin_memory = num_workers > 0
    loader_kwargs = dict(
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_collate_fn,
        persistent_workers=(num_workers > 0),
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 4

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler, **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs,
    )

    # ---- Модель ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImportanceModel(
        backbone_name=backbone,
        num_labels=NUM_LABELS,
        pretrained=True,
        in_channels=in_channels,
        dropout=dropout,
    ).to(device)

    # ---- pos_weight по частоте меток (только из реальных train-примеров) ----
    pos_weight_tensor = None
    if use_pos_weight:
        real_train_df = train_df[~train_df["_is_pseudo"]]
        n_real_train = len(real_train_df)
        if n_real_train > 0:
            pos_counts = torch.zeros(NUM_LABELS, dtype=torch.float32)
            for _, row in real_train_df.iterrows():
                strs = parse_expert_labels(row, None)
                t = build_target_vector(strs)
                pos_counts += t
            neg_counts = n_real_train - pos_counts
            # классический pos_weight = neg / pos, с защитой от деления на 0
            raw_pw = neg_counts / pos_counts.clamp(min=1.0)
            pos_weight_tensor = raw_pw.clamp(max=pos_weight_cap).to(device)
            print(f"pos_weight: min={pos_weight_tensor.min():.2f} "
                  f"max={pos_weight_tensor.max():.2f} "
                  f"mean={pos_weight_tensor.mean():.2f} (capped at {pos_weight_cap})")

    if loss_type == "bce":
        criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight_tensor)
    elif loss_type == "focal":
        # Multi-label focal loss с pos_weight
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

    # Два набора параметров: голова и бэкбон (разные lr)
    # NB: берём параметры ДО оборачивания в DataParallel
    head_params = list(model.head.parameters())
    backbone_params = list(model.backbone.parameters())
    optimizer = torch.optim.AdamW([
        {"params": head_params, "lr": lr},
        {"params": backbone_params, "lr": lr * 0.1},
    ], weight_decay=weight_decay)

    # Freeze backbone на первые freeze_backbone_epochs эпох
    def set_backbone_grad(requires_grad: bool):
        for p in backbone_params:
            p.requires_grad = requires_grad

    if freeze_backbone_epochs > 0:
        set_backbone_grad(False)
        print(f"Backbone frozen for first {freeze_backbone_epochs} epochs (warmup)")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_weights = train_weights.to(device)

    # Multi-GPU: оборачиваем в DataParallel ПОСЛЕ создания optimizer
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if multi_gpu and n_gpus > 1:
        model = nn.DataParallel(model)
        print(f"Using DataParallel on {n_gpus} GPUs")
    else:
        print(f"Using single device: {device} (n_gpus={n_gpus})")

    # AMP (mixed precision) — сильно ускоряет на T4/V100/A100
    amp_enabled = use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    if amp_enabled:
        print("Mixed precision (AMP) enabled")

    # cudnn autotuner — ускоряет свёртки при фиксированном input size
    torch.backends.cudnn.benchmark = True

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
        non_blocking = pin_memory
        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)):
            images = batch["image"].to(device, non_blocking=non_blocking)
            targets = batch["target"].to(device, non_blocking=non_blocking)
            idx = batch["idx"]
            if isinstance(idx, torch.Tensor):
                w = train_weights[idx.to(device)].unsqueeze(1)
            else:
                w = train_weights[torch.as_tensor(idx, device=device)].unsqueeze(1)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = model(images)
                smooth_targets = _smooth_targets(targets, label_smoothing)
                loss_mat = criterion(logits, smooth_targets)
                loss = (loss_mat * w).mean()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

            # Явно освобождаем ссылки на батч и периодически дёргаем GC,
            # иначе мелкие numpy/PIL аллокации копятся и RAM растёт ~MB/батч.
            del images, targets, logits, smooth_targets, loss_mat, loss, w, batch
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
                targets = batch["target"].to(device, non_blocking=non_blocking)
                with torch.amp.autocast("cuda", enabled=amp_enabled):
                    logits = model(images)
                    loss = criterion(logits, targets).mean()
                val_loss += loss.item()
                # Сразу переносим на CPU, чтобы не держать GPU-тензоры до конца валидации
                all_logits.append(logits.float().cpu())
                all_targets.append(targets.cpu())
                del images, targets, logits, loss, batch
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
            # Разворачиваем DataParallel перед сохранением
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
