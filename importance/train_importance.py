"""
Обучение модели отбора важных признаков.
Запуск: train_importance(data_csv=..., image_dir=...) из ноутбука или скрипта.
"""

import json
import random
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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
        base = [
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        if is_training:
            base.insert(2, transforms.RandomHorizontalFlip(p=0.5))
            base.insert(3, transforms.RandomVerticalFlip(p=0.5))
        return transforms.Compose(base)

    from PIL import Image as PILImage

    def transform_4ch(img):
        rgb = img[..., :3]
        mask = img[..., 3:4].squeeze(-1)
        if mask.max() <= 1:
            mask = (mask * 255).astype(np.uint8)
        rgb_pil = PILImage.fromarray(rgb).resize((image_size, image_size))
        mask_pil = PILImage.fromarray(mask).resize((image_size, image_size))
        if is_training:
            if random.random() > 0.5:
                rgb_pil = rgb_pil.transpose(PILImage.FLIP_LEFT_RIGHT)
                mask_pil = mask_pil.transpose(PILImage.FLIP_LEFT_RIGHT)
            if random.random() > 0.5:
                rgb_pil = rgb_pil.transpose(PILImage.FLIP_TOP_BOTTOM)
                mask_pil = mask_pil.transpose(PILImage.FLIP_TOP_BOTTOM)
        rgb_t = transforms.ToTensor()(rgb_pil)
        mask_t = torch.from_numpy(np.array(mask_pil)).float().unsqueeze(0) / 255.0
        x = torch.cat([rgb_t, mask_t], dim=0)
        x[0:3] = (x[0:3] - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) / torch.tensor(
            [0.229, 0.224, 0.225]
        ).view(3, 1, 1)
        return x

    return transform_4ch


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
    epochs: int = 30,
    batch_size: int = 16,
    lr: float = 1e-4,
    image_size: int = 224,
    use_mask: bool = False,
    out_dir: Union[str, Path] = "importance_checkpoints",
    seed: int = RANDOM_STATE,
) -> float:
    """
    Обучает модель; сохраняет лучший чекпоинт в ``out_dir / best.pt`` и при случайном
    сплите — ``out_dir / splits.json``.

    Требования к CSV: колонки ``image_id``, ``filename`` (или задайте *_col), ``important_labels``.

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

    df = pd.read_csv(data_csv)
    if image_id_col not in df.columns and "image_id" in df.columns:
        image_id_col = "image_id"
    if image_col not in df.columns and "filename" in df.columns:
        image_col = "filename"

    if splits_file and Path(splits_file).is_file():
        with open(splits_file) as f:
            splits = json.load(f)
        if "train_idx" in splits:
            train_idx = splits["train_idx"]
            val_idx = splits["val_idx"]
        else:
            train_ids = set(splits.get("train", []))
            val_ids = set(splits.get("val", []))
            train_idx = [i for i, x in df[image_id_col].items() if x in train_ids]
            val_idx = [i for i, x in df[image_id_col].items() if x in val_ids]
        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()
    else:
        n = len(df)
        n_val = max(1, int(n * val_ratio))
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n)
        val_pos = perm[:n_val]
        train_pos = perm[n_val:]
        train_df = df.iloc[train_pos].copy()
        val_df = df.iloc[val_pos].copy()
        splits = {"train_idx": train_df.index.tolist(), "val_idx": val_df.index.tolist()}
        with open(out_dir / "splits.json", "w") as f:
            json.dump(splits, f, indent=2)

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
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImportanceModel(
        backbone_name=backbone,
        num_labels=NUM_LABELS,
        pretrained=True,
        in_channels=in_channels,
        dropout=0.2,
    ).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    run_config = {
        "data_csv": str(data_csv.resolve()),
        "image_dir": image_dir,
        "mask_dir": mask_dir_s,
        "image_col": image_col,
        "image_id_col": image_id_col,
        "val_ratio": val_ratio,
        "splits_file": str(splits_file) if splits_file else None,
        "backbone": backbone,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "image_size": image_size,
        "use_mask": use_mask,
        "out_dir": str(out_dir),
        "seed": seed,
    }

    best_score = 0.0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            images = batch["image"].to(device)
            targets = batch["target"].to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
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
                loss = criterion(logits, targets)
                val_loss += loss.item()
                all_logits.append(logits)
                all_targets.append(targets)
        val_loss /= len(val_loader)
        logits_cat = torch.cat(all_logits, dim=0)
        targets_cat = torch.cat(all_targets, dim=0)
        prec10, rec10 = precision_recall_at_k_batch(logits_cat, targets_cat, k=10)
        score = (prec10 + rec10) / 2.0

        print(
            f"Epoch {epoch+1}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
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
