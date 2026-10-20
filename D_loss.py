#!/usr/bin/env python

import torch
import torch.nn.functional as FUNC_OP


def symmetric_info_nce(
    image_z: torch.Tensor,                                                                              # image_z: Image features from the shared feature space
    dna_z: torch.Tensor,                                                                                # dna_z: DNA features from the shared feature space
    temperature: float = 0.07,                                                                          # Temperature: How strongly are similarity differences emphasised?
) -> torch.Tensor:
    """Bidirectional in-batch contrastive loss for paired specimens."""                                 # Sanity CHECKS
    if image_z.ndim != 2 or dna_z.ndim != 2:                                                            
        raise ValueError("Expected image_z and dna_z to have shape [batch, dim].")
    if image_z.shape != dna_z.shape:
        raise ValueError("Image and DNA embeddings must have the same shape.")
    if image_z.shape[0] < 2:
        raise ValueError("InfoNCE needs at least two paired specimens per batch.")                      # Contrastive learning needs negative examples
    if temperature <= 0:
        raise ValueError("temperature must be positive.")

    """Reads: Cosine similarities devided by temperature"""
    logits = (image_z @ dna_z.T) / temperature                                                          # (image_z @ dna_z.T) gives the dot product between every image embedding and every DNA embedding the '.T' transposes the dna features so they can be multiplied

    targets = torch.arange(image_z.shape[0], device=image_z.device)                                     # ensures the indexes for both dna and images correspond

    """The loss tells how good each prediction was."""
    image_to_dna = FUNC_OP.cross_entropy(logits, targets)                                               # Given an image embedding, which dna embedding is its correct match? 
    dna_to_image = FUNC_OP.cross_entropy(logits.T, targets)                                             # Given a DNA embedding, which image embedding is its correct match? 
    loss = 0.5 * (image_to_dna + dna_to_image) 
    return loss                                                                                         # We take the average of the loss of both directions which is our actual loss.  


def symmetric_accuracy(
    image_z: torch.Tensor,                                                                              
    dna_z: torch.Tensor,                                                                                
    temperature: float = 0.07,                                                                          
) -> torch.Tensor:
    """Calculate training accuracy for plotting."""                                 
    if image_z.ndim != 2 or dna_z.ndim != 2:                                                            
        raise ValueError("Expected image_z and dna_z to have shape [batch, dim].")
    if image_z.shape != dna_z.shape:
        raise ValueError("Image and DNA embeddings must have the same shape.")
    if image_z.shape[0] < 2:
        raise ValueError("InfoNCE needs at least two paired specimens per batch.")                      
    if temperature <= 0:
        raise ValueError("temperature must be positive.")

    """Reads: Cosine similarities devided by temperature"""
    logits = (image_z @ dna_z.T) / temperature                                                         

    targets = torch.arange(image_z.shape[0], device=image_z.device)                                    

    image_to_dna_acc = (logits.argmax(dim=1) == targets).float().mean()
    dna_to_image_acc = (logits.T.argmax(dim=1) == targets).float().mean()
    accuracy = 0.5 * (image_to_dna_acc + dna_to_image_acc)

    return accuracy                                                                                     # We take the average of the loss of both directions which is our actual accuracy.  

if __name__ == "__main__":
    print("This file holds the function for the infoNCE loss, please run either 'Run_alignment_train.py' for training or 'Run_evaluation.py' for evaluation of a previously trained model.")