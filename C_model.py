#!/usr/bin/env python

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as FUNC_OP # PyTorch's optional neural network operations


class ProjectionHead(nn.Module):
    """
    This component takes an encoder's feature output and converts it to the shared feature space.
    Ensuring that both the visual and molecular modalities eventually share the same dimensionality. 
    """
    def __init__(
        self,
        input_dim: int,                                 # Dimension size of the encoder output
        shared_dim: int = 256,                          # Dimension size of the shared feature space
        hidden_dim: int = 512,                          # Dimension size of the intermediate hidden layer of the projection head.
    ):
        super().__init__()                              # Initialize parent neural network module that the projection head inherits from.  
        self.layers = nn.Sequential(                    # Create sequence of neural network layers. 
            nn.Linear(input_dim, hidden_dim),           # nn.Linear is a fully connected layer
            nn.GELU(),                                  # Activation function that uses Gaussian smoothing instead of a hard cut-off for negative weights.
            nn.Linear(hidden_dim, shared_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor: # takes embeddings/features as input in form of a Tensor, then outputs a Tensor with different dimensions. 
        return FUNC_OP.normalize(self.layers(features), p=2, dim=-1) # Runs features through the projection head turning them into the desired dimension, then normalizing them since we care about their direction, their magnitude is inconsequential. 


class PairedAlignmentModel(nn.Module):
    """
    This Component is a model that learns to put the representation of an image and its corresponding DNA Sequence
    closes together in the shared embedding space.
    """
    def __init__(                                       # Constructor that initializes the inner components and values when the object is created.  
        self,
        image_encoder: nn.Module,
        dna_encoder: nn.Module,
        shared_dim: int = 256,
        projection_hidden_dim: int = 512,
        freeze_encoders: bool = True,                   # Do we want to train the model?
    ):
        super().__init__()                              # Initialize parent neural network module that the projection head inherits from and sets the variables initialized during the constructor to the ones from the parent. 
        self.image_encoder = image_encoder
        self.dna_encoder = dna_encoder
        self.freeze_encoders = freeze_encoders
        self.shared_dim = shared_dim        

        self.image_projection = ProjectionHead(         # Converts the visual embeddings into the shared dimension using the projection above.
            image_encoder.output_dim,
            shared_dim,
            projection_hidden_dim,
        )
        self.dna_projection = ProjectionHead(           # Converts the visual embeddings into the shared dimension using the projection above.
            dna_encoder.output_dim,
            shared_dim,
            projection_hidden_dim,
        )

        if freeze_encoders:                             # Freezes the encoder weights if the boolean is set to True
            self.image_encoder.requires_grad_(False)
            self.dna_encoder.requires_grad_(False)

    def train(self, mode: bool = True):                 # Overwrite PyTorch's normal training method with our own.
        super().train(mode)                             # Performs PyTorch's normal training behavior
        if self.freeze_encoders:                        # Then force encoders into evaluation mode if the encoders are frozen. 
            self.image_encoder.eval()                   
            self.dna_encoder.eval()
        return self                                     # The training model returns the model itself allows for things like: model.train().to(device)

    def forward(
        self,
        images: torch.Tensor,                           # Image Tensor
        input_ids: torch.Tensor,                        # Tokenized DNA Sequences  
        attention_mask: torch.Tensor,                   # Tels the transformer which tokens are real and which are padding. 
    ) -> tuple[torch.Tensor, torch.Tensor]:             # The Method returns a tuple with two tensors, image_z and dna_z of shape [batch_size, shared_dim]
        if self.freeze_encoders:
            with torch.no_grad():
                image_features = self.image_encoder(images)                     
                dna_features = self.dna_encoder(input_ids, attention_mask)
        else:
            image_features = self.image_encoder(images)
            dna_features = self.dna_encoder(input_ids, attention_mask)

        image_z = self.image_projection(image_features) # transform the image representation into the shared space.
        dna_z = self.dna_projection(dna_features)       # transform the dna representation into the shared space.
        return image_z, dna_z

if __name__ == "__main__":
    pass
