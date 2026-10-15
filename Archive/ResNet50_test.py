from contextlib import nullcontext
from typing import Sequence

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as transforms


from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt
from tsne_torch import TorchTSNE as TSNE

class ResNet50Encoder(nn.Module):
    def __init__(self, embedding_dim=None, freeze_backbone=False):
        super().__init__()

        weights = ResNet50_Weights.DEFAULT
        self.backbone = resnet50(weights=weights)

        # Remove the ImageNet classification layer.
        self.backbone.fc = nn.Identity()
        self.freeze_backbone = freeze_backbone

        # Optional projection into a shared multimodal embedding space.
        self.projection = (
            nn.Identity()
            if embedding_dim is None
            else nn.Linear(2048, embedding_dim)
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        # Keep frozen BatchNorm layers from updating their running statistics.
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, images):
        # images: [batch, 3, height, width]
        features = self.backbone(images)  # [batch, 2048]
        return self.projection(features)  # [batch, embedding_dim],


if __name__ == "__main__":
    data_folder = Path("./data/images")

    # Match preprocessing to the pretrained ResNet-50 weights.
    weights = ResNet50_Weights.DEFAULT
    preprocess = weights.transforms()

    allowed_filetypes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    image_paths = sorted(
        p for p in data_folder.rglob("*")
        if p.is_file() and p.suffix.lower() in allowed_filetypes
    )

    if not image_paths:
        raise ValueError(f"No images found in {data_folder.resolve()}")

    # Load and preprocess each image, then combine into [N, 3, H, W].
    image_tensors = []
    for path in image_paths:
        with Image.open(path) as image:
            image_tensors.append(preprocess(image.convert("RGB")))

    image_batch = torch.stack(image_tensors)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vision_encoder = ResNet50Encoder(
        embedding_dim=256,
        freeze_backbone=True,
    ).to(device).eval()

    with torch.inference_mode():
        embeddings = vision_encoder(image_batch.to(device))

    print(embeddings.shape)  # [number of images, 256]
    print(image_paths)