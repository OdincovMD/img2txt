"""
Обучение модели отбора важных признаков.
Конфиг через аргументы; данные — CSV с разметкой эксперта и путями к изображениям.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

try:
    from torchvision import transforms
except ImportError:
    transforms = None

from importance_config import LABEL_NAMES, NUM_LABELS
from importance_dataset import ImportanceDataset, parse_expert_labels
from importance_metrics import precision_recall_at_k_batch
from importance_model import ImportanceModel

RANDOM_STATE = 42


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

    # 4 channels: RGB + mask. Same geometric augments for both, then normalize RGB.
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
        x[0:3] = (x[0:3] - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return x
    return transform_4ch


def main():
    parser = argparse.ArgumentParser(description="Train importance selection model")
    parser.add_argument("--data_csv", type=str, required=True, help="CSV: image_id, filename/path, expert labels")
    parser.add_argument("--image_dir", type=str, required=True, help="Root directory for images")
    parser.add_argument("--mask_dir", type=str, default=None, help="Optional: directory with masks")
    parser.add_argument("--image_col", type=str, default="filename", help="Column with image filename")
    parser.add_argument("--image_id_col", type=str, default="image_id", help="Column with image id")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Validation fraction")
    parser.add_argument("--splits_file", type=str, default=None, help="Optional: JSON with train/val indices or image_ids")
    parser.add_argument("--backbone", type=str, default="efficientnet_b0", choices=["efficientnet_b0", "resnet18", "resnet34"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--use_mask", action="store_true", help="Use mask as 4th channel")
    parser.add_argument("--out_dir", type=str, default="importance_checkpoints", help="Where to save checkpoints")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data_csv)
    if args.image_id_col not in df.columns and "image_id" in df.columns:
        args.image_id_col = "image_id"
    if args.image_col not in df.columns and "filename" in df.columns:
        args.image_col = "filename"

    if args.splits_file and Path(args.splits_file).is_file():
        with open(args.splits_file) as f:
            splits = json.load(f)
        if "train_idx" in splits:
            train_idx = splits["train_idx"]
            val_idx = splits["val_idx"]
        else:
            train_ids = set(splits.get("train", []))
            val_ids = set(splits.get("val", []))
            train_idx = [i for i, x in df[args.image_id_col].items() if x in train_ids]
            val_idx = [i for i, x in df[args.image_id_col].items() if x in val_ids]
        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()
    else:
        n = len(df)
        n_val = max(1, int(n * args.val_ratio))
        n_train = n - n_val
        train_df, val_df = random_split(df, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))
        train_df = df.iloc[train_df.indices].copy()
        val_df = df.iloc[val_df.indices].copy()
        splits = {"train_idx": train_df.index.tolist(), "val_idx": val_df.index.tolist()}
        with open(out_dir / "splits.json", "w") as f:
            json.dump(splits, f, indent=2)

    in_channels = 4 if args.use_mask else 3
    transform_train = get_transform(args.image_size, is_training=True, in_channels=in_channels)
    transform_val = get_transform(args.image_size, is_training=False, in_channels=in_channels)

    train_ds = ImportanceDataset(
        train_df,
        image_dir=args.image_dir,
        image_col=args.image_col,
        image_id_col=args.image_id_col,
        mask_dir=args.mask_dir,
        transform=transform_train,
        use_mask_channel=args.use_mask,
    )
    val_ds = ImportanceDataset(
        val_df,
        image_dir=args.image_dir,
        image_col=args.image_col,
        image_id_col=args.image_id_col,
        mask_dir=args.mask_dir,
        transform=transform_val,
        use_mask_channel=args.use_mask,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImportanceModel(
        backbone_name=args.backbone,
        num_labels=NUM_LABELS,
        pretrained=True,
        in_channels=in_channels,
        dropout=0.2,
    ).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_score = 0.0
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False):
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

        print(f"Epoch {epoch+1}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  P@10={prec10:.4f}  R@10={rec10:.4f}  score={score:.4f}")

        if score > best_score:
            best_score = score
            ckpt = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "score": score,
                "prec10": prec10,
                "rec10": rec10,
                "args": vars(args),
                "label_names": LABEL_NAMES,
            }
            torch.save(ckpt, out_dir / "best.pt")
            print(f"  -> saved best.pt (score={score:.4f})")

    print(f"Done. Best score: {best_score:.4f}. Checkpoints in {out_dir}")


if __name__ == "__main__":
    main()
