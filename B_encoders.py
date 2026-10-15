from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50
from transformers import AutoModel, AutoTokenizer

def normalize_dna_sequence(sequence: str) -> str:
    """Normalize a sequence body; do not pass FASTA headers."""
    sequence = "".join(sequence.split()).upper().replace("U", "T")
    # Keep the input alphabet compatible with BarcodeBERT.
    return "".join(base if base in "ACGT" else "N" for base in sequence)


class ResNet50Encoder(nn.Module):
    """ImageNet-pretrained ResNet-50 returning its 2048-D pooled features."""

    def __init__(
        self,
        weights=ResNet50_Weights.IMAGENET1K_V2,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.weights = weights
        self.preprocess = weights.transforms()
        self.backbone = resnet50(weights=weights)
        self.backbone.fc = nn.Identity()
        self.output_dim = 2048
        self.freeze_backbone = freeze_backbone

        if freeze_backbone:
            self.backbone.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images)


class BarcodeBERTEncoder(nn.Module):
    """BarcodeBERT tokenization and mask-aware mean pooling."""

    def __init__(
        self,
        model_name: str = "bioscan-ml/BarcodeBERT",
        train_backbone: bool = False,
    ):
        super().__init__()
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        self.backbone = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        self.output_dim = self.backbone.config.hidden_size
        self.train_backbone = train_backbone
        self.backbone.requires_grad_(train_backbone)

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.train_backbone:
            self.backbone.eval()
        return self

    def encode_sequence(self, sequence: str) -> tuple[torch.Tensor, torch.Tensor]:
        sequence = normalize_dna_sequence(sequence)
        if not sequence:
            raise ValueError("Encountered an empty DNA sequence.")

        # BarcodeBERT's custom tokenizer handles k-mer splitting and padding.
        encoded = self.tokenizer(
            sequence,
            padding=True,
            return_tensors="pt",
        )
        input_ids = torch.as_tensor(encoded["input_ids"]).reshape(-1).long()
        attention_mask = torch.as_tensor(encoded["attention_mask"]).reshape(-1).long()
        return input_ids, attention_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        device = next(self.backbone.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None:
            hidden_states = outputs["hidden_states"]
        hidden = hidden_states[-1]

        # Exclude padding and [UNK] positions from the pooled representation.
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        numerator = (hidden * mask).sum(dim=1)
        denominator = mask.sum(dim=1).clamp_min(1.0)
        return numerator / denominator