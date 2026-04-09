"""Сегментация: UNet + YOLO для получения маски очага."""

import numpy as np
from PIL import Image
from resizeimage import resizeimage
from ultralytics import YOLO
import cv2

# PyTorch / UNet
import torch
import torch.nn as nn


class BlockBuilder:
    @staticmethod
    def create_enc_dec_block(in_dim: int, out_dim: int, is_last: bool = False) -> nn.Sequential:
        block = [
            nn.Sequential(
                nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features=out_dim),
                nn.ReLU(),
            )
        ]
        if is_last:
            block.append(nn.Conv2d(in_channels=out_dim, out_channels=1, kernel_size=3, padding=1))
        else:
            block.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=3, padding=1),
                    nn.BatchNorm2d(num_features=out_dim),
                    nn.ReLU(),
                )
            )
        return nn.Sequential(*block)

    @staticmethod
    def create_pool_block(is_unpool: bool = False, return_indices: bool = True):
        if is_unpool:
            return nn.MaxUnpool2d(2, stride=2)
        return nn.MaxPool2d(2, stride=2, return_indices=return_indices)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        builder = BlockBuilder()
        self.enc_conv0 = builder.create_enc_dec_block(3, 64)
        self.pool0 = builder.create_pool_block(return_indices=False)
        self.enc_conv1 = builder.create_enc_dec_block(64, 128)
        self.pool1 = builder.create_pool_block(return_indices=False)
        self.enc_conv2 = builder.create_enc_dec_block(128, 256)
        self.pool2 = builder.create_pool_block(return_indices=False)
        self.enc_conv3 = builder.create_enc_dec_block(256, 512)
        self.pool3 = builder.create_pool_block(return_indices=False)
        self.bottleneck_conv = builder.create_enc_dec_block(512, 1024)
        self.upsample0 = nn.Upsample(32)
        self.dec_conv0 = builder.create_enc_dec_block(1024 + 512, 512)
        self.upsample1 = nn.Upsample(64)
        self.dec_conv1 = builder.create_enc_dec_block(512 + 256, 256)
        self.upsample2 = nn.Upsample(128)
        self.dec_conv2 = builder.create_enc_dec_block(256 + 128, 128)
        self.upsample3 = nn.Upsample(256)
        self.dec_conv3 = builder.create_enc_dec_block(128 + 64, 32, True)

    def forward(self, x):
        e0 = self.enc_conv0(x)
        e1 = self.enc_conv1(self.pool0(e0))
        e2 = self.enc_conv2(self.pool1(e1))
        e3 = self.enc_conv3(self.pool2(e2))
        b = self.bottleneck_conv(self.pool3(e3))
        d0 = self.dec_conv0(torch.cat([self.upsample0(b), e3], dim=1))
        d1 = self.dec_conv1(torch.cat([self.upsample1(d0), e2], dim=1))
        d2 = self.dec_conv2(torch.cat([self.upsample2(d1), e1], dim=1))
        d3 = self.dec_conv3(torch.cat([self.upsample3(d2), e0], dim=1))
        return d3


def get_prediction(image_path: str, unet_weights: str) -> np.ndarray:
    """UNet-предсказание маски (fallback при отсутствии YOLO-масок)."""
    image = Image.open(image_path)
    image = np.array(resizeimage.resize_cover(image, [256, 256], validate=False))
    image = np.rollaxis(image, 2, 0)
    transformed_image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)

    model = UNet()
    model.load_state_dict(torch.load(unet_weights, weights_only=True, map_location="cpu"))
    model.eval()
    with torch.no_grad():
        output = model(transformed_image)
        pred = (torch.sigmoid(output) > 0.5).cpu().numpy()[0, 0]
    return pred


def segment_with_loaded_yolo(path_to_image: str, yolo_model: "YOLO") -> np.ndarray:
    """Сегментация уже загруженной YOLO-моделью (без ре-инициализации)."""
    results = yolo_model(path_to_image, retina_masks=True, save=False, verbose=False)
    orig_img = results[0].orig_img
    height, width = orig_img.shape[:2]
    combined_mask = np.zeros((height, width), dtype=np.uint8)
    if results[0].masks and results[0].masks.xy:
        for mask in results[0].masks.xy:
            mask_points = np.array(mask, dtype=np.int32)
            instance_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(instance_mask, [mask_points], 255)
            combined_mask = cv2.bitwise_or(combined_mask, instance_mask)
    return combined_mask


def main(
    path_to_image: str,
    yolo_weights: str = "/kaggle/input/weight-mask/weight/mask_builder_yolo.pt",
    unet_weights: str = "/kaggle/input/weight-mask/weight/mask_builder_unet.pth",
) -> np.ndarray:
    """
    Сегментация: YOLO (основной), UNet (fallback).
    Возвращает маску в размере исходного изображения.
    """
    model = YOLO(yolo_weights)
    results = model(path_to_image, retina_masks=True, save=False, verbose=False)
    orig_img = results[0].orig_img
    height, width = orig_img.shape[:2]
    combined_mask = np.zeros((height, width), dtype=np.uint8)

    if results[0].masks and results[0].masks.xy:
        for mask in results[0].masks.xy:
            mask_points = np.array(mask, dtype=np.int32)
            instance_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(instance_mask, [mask_points], 255)
            combined_mask = cv2.bitwise_or(combined_mask, instance_mask)
    else:
        mask = get_prediction(path_to_image, unet_weights)
        mask_img = Image.fromarray(mask)
        mask_resized = mask_img.resize((width, height), resample=Image.NEAREST)
        combined_mask = np.array(mask_resized, dtype=np.uint8)
    return combined_mask
