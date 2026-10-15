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
from Encoders import BarcodeBERTEncoder, ResNet50Encoder, ProjectionHead, PairedAligner


def get_sequences(file_path):
    with open(file_path, 'r') as inputfile:
        line = inputfile.readline()
        while len(line) != 0:
            # print(line[11], "\n")
            print(line, "\n")
            line = inputfile.readline().split(',')
        inputfile.close()

# Karaderi-style loss per anchor:
def anchor_loss(image_z, positive_anchor, negative_anchor, margin=0.5):
    positive_similarity = F.cosine_similarity(image_z, positive_anchor, dim=-1)
    negative_similarity = F.cosine_similarity(image_z, negative_anchor, dim=-1)

    return (1 - positive_similarity + F.relu(negative_similarity - margin)).mean()


class ImageToDNA(nn.Module):
    def __init__(self, dna_dim):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, dna_dim),
        )

    def forward(self, image_features):
        return F.normalize(self.projector(image_features), dim=-1)




def paired_contrastive_loss(image_z, dna_z, temperature=0.07):
    batch_size = image_z.shape[0]
    if batch_size < 2:
        raise ValueError("Contrastive training needs at least 2 paired specimens per batch.")

    # Diagonal entries are the matching image–DNA pairs.
    logits = image_z @ dna_z.T / temperature
    labels = torch.arange(batch_size, device=logits.device)

    image_to_dna = F.cross_entropy(logits, labels)
    dna_to_image = F.cross_entropy(logits.T, labels)
    return (image_to_dna + dna_to_image) / 2


    
if __name__ == "__main__":
    # file_path = "./csv/BIOSCAN_5M_Insect_Dataset_metadata.csv"
    # get_sequences(file_path)
    # dna_by_taxon: {taxon: tensor of shape [n_sequences, dna_dim]}
    # anchors = {
    #     taxon: F.normalize(v.median(dim=0).values, dim=0)
    #     for taxon, v in dna_by_taxon.items()
    # }
    pass