#!/usr/bin/env python

import torch
import torch.nn.functional as FUNC_OP

from Bio import Entrez, SeqIO
from Bio.Align import PairwiseAligner
import math

"""
Distance Calculation functions, Author: shlomias1 on github - https://github.com/shlomias1/sequence_distance_calculator/blob/main/sequence_distance_calculator.py used under its MIT license.
"""
def align_sequences(seq1, seq2):
    """Align two sequences using PairwiseAligner with basic settings."""
    aligner = PairwiseAligner()
    aligner.mode = 'local'  # Use local alignment
    aligner.open_gap_score = -10  # Penalty for opening a gap
    aligner.extend_gap_score = -0.5  # Penalty for extending a gap
    alignments = list(aligner.align(seq1, seq2))
    best_alignment = alignments[0]  # Choose the best alignment
    aligned_seq1 = best_alignment.target  # Aligned sequence from target
    aligned_seq2 = best_alignment.query  # Aligned sequence from query
    return aligned_seq1, aligned_seq2

def preprocess_sequences(seq1, seq2):
    """Remove gaps ('-') from aligned sequences."""
    seq1_cleaned = "".join(base1 for base1, base2 in zip(seq1, seq2) if base1 != '-' and base2 != '-')
    seq2_cleaned = "".join(base2 for base1, base2 in zip(seq1, seq2) if base1 != '-' and base2 != '-')
    return seq1_cleaned, seq2_cleaned

def calculate_transitions_and_transversions(seq1, seq2):
    """Calculate the number of transitions (P) and transversions (Q)."""
    transitions = 0
    transversions = 0

    purines = {'A', 'G'}  # A and G are purines
    pyrimidines = {'C', 'T'}  # C and T are pyrimidines

    for base1, base2 in zip(seq1, seq2):
        if base1 != base2:  # Mutation occurred
            if {base1, base2} <= purines or {base1, base2} <= pyrimidines:
                transitions += 1
            else:
                transversions += 1

    return transitions, transversions

def hamming_distance(seq1, seq2):
    """Calculate Hamming distance between two sequences."""
    if len(seq1) != len(seq2):
        raise ValueError("Sequences must be of the same length.")
    return sum(el1 != el2 for el1, el2 in zip(seq1, seq2))

def jukes_cantor_distance(seq1, seq2):
    """Calculate Jukes-Cantor distance based on Hamming distance."""
    seq_length = len(seq1)
    dH = hamming_distance(seq1, seq2) / seq_length  # Proportion of differences
    if dH >= 0.75:
        raise ValueError("Jukes-Cantor model is not valid when dH >= 0.75.")
    dJC = -3/4 * math.log(1 - (4/3) * dH)
    return dJC

def kimura_distance(seq1, seq2):
    """Calculate Kimura distance between two sequences."""
    seq_length = len(seq1)
    transitions, transversions = calculate_transitions_and_transversions(seq1, seq2)

    P = transitions / seq_length  # Proportion of transitions
    Q = transversions / seq_length  # Proportion of transversions

    # Check for invalid values
    if 1 - 2 * P - Q <= 0 or 1 - 2 * Q <= 0:
        raise ValueError("Kimura model is not valid for these sequences.")
    dK = -0.5 * math.log(1 - 2 * P - Q) - 0.25 * math.log(1 - 2 * Q)
    return dK

#---------------------------------------------------------------------------------------------------

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
    loss = 0.5 * (image_to_dna + dna_to_image)                                                          # We take the average of the loss of both directions which is our actual loss.  
    return loss                                                                                         


def symmetric_accuracy(
    image_z: torch.Tensor,                                                                              
    dna_z: torch.Tensor,                                                                                
    temperature: float = 0.07,                                                                          
) -> torch.Tensor:
    """Calculate alignment training accuracy for plotting."""                                 
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
    return accuracy                                                                                     # We take the average of the loss of both directions which is our actual alignment accuracy.  

def distance_prediction_loss(predicted_distance_value: float, query_seq: str, target_seq: str) -> torch.Tensor:
    """Loss for phylogenetic distance, returns the difference between predicted value and """          

    aligned_query, aligned_target = align_sequences(query_seq, target_seq)
    cleaned_query, cleaned_target = preprocess_sequences(aligned_query, aligned_target)
    calculated_distance = kimura_distance(cleaned_query, cleaned_target) 
                                 
    loss = abs(predicted_distance_value - calculated_distance)                                                  
    return loss       

if __name__ == "__main__":
    print("This file holds the function for the infoNCE loss, please run either 'Run_alignment_train.py' for training or 'Run_evaluation.py' for evaluation of a previously trained model.")