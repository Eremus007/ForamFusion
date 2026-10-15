from contextlib import nullcontext
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from torchvision.models import resnet50, ResNet50_Weights

import numpy as np
import matplotlib.pyplot as plt
from tsne_torch import TorchTSNE as TSNE


class BarcodeBERTEncoder(nn.Module):
    def __init__(
        self,
        model_name="bioscan-ml/BarcodeBERT",
        train_backbone=False,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        self.backbone = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        self.backbone.requires_grad_(train_backbone)

    def tokenize_batch(self, sequences):
        """Convert a list of raw DNA sequences into BarcodeBERT tokens"""
        token_ids = []
        for sequence in sequences:
            # Truncate
            sequence = sequence[:660]
            # Pad with N
            sequence = sequence + "N" * (660 - len(sequence))
            # Non-overlapping 4-mers
            kmers = [
                sequence[i:i + 4]
                for i in range(0, len(sequence), 4)
            ]

            ids = self.tokenizer.convert_tokens_to_ids(kmers)
            token_ids.append(ids)

        return torch.tensor(token_ids, dtype=torch.long)

    def forward(self, input_ids):
        """Token ID's through BarcodeBert backbone to generate embeddings"""
        input_ids = input_ids.to(self.backbone.device)
        outputs = self.backbone(
            input_ids=input_ids
        )

        hidden_states = outputs["hidden_states"][-1]
        # Global average pooling
        embeddings = hidden_states.mean(dim=1)
        return embeddings

    def predict(self, sequences):
        """Full pipeline, from raw DNA, tokenization, forward pass, embeddings."""
        self.eval()
        with torch.no_grad():
            input_ids = self.tokenize_batch(sequences)
            return self.forward(input_ids)

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

    dna_encoder = BarcodeBERTEncoder()

    sequences = [
        "ACGCGCTGACGCATCAGCATACGA",
        "ATGCTAGCTAGCTACGATCGATCG",
    ]

    embeddings = dna_encoder.predict(sequences)

    print(embeddings.shape)

    # To DO: tokenize in Dataset/DataLoader's collate function, rather than every time forward() is called