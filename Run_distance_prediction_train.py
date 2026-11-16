#!/usr/bin/env python

from __future__ import annotations

import torch
import argparse
import csv
import random
from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as FUNC_OP

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from collections import OrderedDict

from B_encoders import (
    BarcodeBERTEncoder,
    ResNet50Encoder,
    normalize_dna_sequence,
)
from C_model import PairedAlignmentModel, FusionHead, DistanceHead
from D_loss import (
    symmetric_info_nce, 
    distance_prediction_loss, 
    align_sequences, 
    preprocess_sequences, 
    calculate_transitions_and_transversions, 
    hamming_distance,
    jukes_cantor_distance, 
    kimura_distance
    )
from E_visualisation import show_curve

class PairedSpecimenDataset(Dataset):
    """Takes Manifest CSV and turns it into paired training examples."""
    """One CSV row per specimen with a matched image and DNA sequence."""

    CHECK_IF_EXIST = {"processid", "sampleid", "species", "dna_barcode"}

    def __init__(self, manifest_path: str | Path, image_transform, dna_tokenizer, species_to_ids: dict[str, int]):
        self.manifest_path = Path(manifest_path).resolve()                                                  # Resolve CSV path into an absolute path.
        with self.manifest_path.open("r", newline="", encoding="utf-8-sig") as file:                           # Open/Read CSV
            reader = csv.DictReader(file)                                                                      # Turns each row of the CSV into a python dictionary.
            missing = self.CHECK_IF_EXIST - set(reader.fieldnames or [])                                   # Check missing columns.
            if missing:
                raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
            self.rows = list(reader)

        if not self.rows:
            raise ValueError(f"No samples found in {self.manifest_path}")

        sample_ids = [row["sampleid"] for row in self.rows]                                                # Creates a list of sample IDs
        if len(set(sample_ids)) != len(sample_ids):                                                         # Check for and remove duplicates
            raise ValueError(
                "Ensure one row per unique sample_id"
            )

        self.image_transform = image_transform                                                              # Stores the image preprocessing function in the dataset object.
        self.dna_tokenizer = dna_tokenizer                                                                  # Stores the BarcodeBERT tokenizer in the dataset object.
        self.species_to_ids = species_to_ids

    def __len__(self):
        return len(self.rows)                                                                               # Returns the number of CSV rows and thus the amount of sample pairs. 

    def __getitem__(self, index):                                                                           # Given an index, how do I retrieve one sample example?
        row = self.rows[index]  
        image_path = Path("./images/"+ row["processid"] + ".jpg")
        if not image_path.is_absolute():
            image_path = self.manifest_path.parent / image_path                                             # Turn absolute paths relative.

        with Image.open(image_path) as image:                                                               # Open the image using Pillow.
            image = self.image_transform(image.convert("RGB"))                                              # Ensure the image has three color channels then does the preprocessing for the image encoder.

        sequence = normalize_dna_sequence(row["dna_barcode"])                                               # Takes the barcode sequence from the csv and runs it through the normalization function. 
        if not sequence:
            raise ValueError(f"Empty DNA sequence for sample {row['sampleid']}")                            # Checks whether the sequence is empty after preprocessing. 
        encoded = self.dna_tokenizer(                                                                       # Passes the sequence into the BarcodeBERT tokenizer.
            sequence,
            padding=True,                                                                                   # Pads sequences when nessesary so they are all the same length. 
            return_tensors="pt",
        )
        input_ids = torch.as_tensor(encoded["input_ids"]).reshape(-1).long()                                # Returns the token IDs after flattening it to a single list in 64-bit integer tensor format. 
        attention_mask = torch.as_tensor(encoded["attention_mask"]).reshape(-1).long()                      # Does something similar for the attention mask as above.
        species = row["species"].strip()
        label = torch.tensor(self.species_to_ids[species], dtype=torch.long)

        return image, input_ids, attention_mask, label, sequence, row["sampleid"]                                     # Returns everything needed from the requested sample. 


class DistanceCache:
    def __init__(self, max_entries: int = 20_000):
        self.max_entries = max_entries
        self.values = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, id_a, id_b, seq_a, seq_b) -> float | None:
        id_a, id_b = str(id_a), str(id_b)

        if id_a <= id_b:
            key = (id_a, id_b)
        else:
            key = (id_b, id_a)
            seq_a, seq_b = seq_b, seq_a

        if key in self.values:
            self.hits += 1
            self.values.move_to_end(key)
            return self.values[key]

        self.misses += 1
        distance = calculate_distance_for_pair(seq_a, seq_b)
        self.values[key] = distance

        if len(self.values) > self.max_entries:
            self.values.popitem(last=False)

        return distance
    

def make_species_to_ids(manifest_path) -> dict[str, int]:
    """
    Creates a dictionary to easily convert from IDS to species labels for when we need them.
    """
    with Path(manifest_path).open(
        "r", newline="", encoding="utf-8-sig"
    ) as file:
        rows = list(csv.DictReader(file))

    species_names = {row["species"].strip() for row in rows}
    species_names.discard("")

    if not species_names:
        raise ValueError("No species labels found.")

    return {name: i for i, name in enumerate(sorted(species_names))}

def set_seed(seed: int):
    """
    Helper function to set random seed for reproducibility.
    """                                                                                    
    random.seed(seed)                                                                                       # Set Python's random number generator
    np.random.seed(seed)                                                                                    # Set Numpy's random number generator
    torch.manual_seed(seed)                                                                                 # Set PyTorch's random seed
    if torch.cuda.is_available():                                                                           # If we use CUDA...
        torch.cuda.manual_seed_all(seed)                                                                    # Set seed for CUDA random number generator as well. 


def build_model(dna_model_name: str, shared_dim: int, hidden_dim: int):
    """
    Creates the model component objects and then returns them as a single torch dictionary object.
    """
    image_encoder = ResNet50Encoder(freeze_backbone=True)
    dna_encoder = BarcodeBERTEncoder(model_name=dna_model_name, train_backbone=False)

    alignment = PairedAlignmentModel(
        image_encoder=image_encoder,
        dna_encoder=dna_encoder,
        shared_dim=shared_dim,
        freeze_encoders=True,
    )

    fusion = FusionHead(
        shared_dim=shared_dim,
        hidden_dim=hidden_dim,
    )

    distance = DistanceHead(
        fused_dim=fusion.output_dim,
        hidden_dim=hidden_dim,
    )

    return nn.ModuleDict({
        "alignment": alignment,
        "fusion": fusion,
        "distance": distance,
    })


def make_loader( 
        manifest_path: str | Path,                                                          
        model: nn.ModuleDict,
        batch_size: int,
        shuffle: bool,
        species_to_ids: dict,
        num_workers: int = 0,
    ) -> DataLoader:
    """
    Wraps the dataset object in a PyTorch dataloader object.
    """

    alignment = model["alignment"]                                                                          
    dataset = PairedSpecimenDataset(
        manifest_path,
        alignment.image_encoder.preprocess,                                                                 # Get the encoders from the alignment model component.
        alignment.dna_encoder.tokenizer,
        species_to_ids,
    )

    return DataLoader(                                                                                      # Return dataloader object.
        dataset,                                                                             
        batch_size=batch_size,                                                                              # How many image/dna pair samples per batch. 
        shuffle=shuffle,                                                                                    # Shuffle the dataset during each epoch, leads to more stable training. 
        num_workers=num_workers,                                                                            # How many subprocesses we have to load data.
        pin_memory=torch.cuda.is_available(),                                                               # If Cuda is available, enables pinned memory.
        drop_last=False,                                                                                    # Final batch is kept even if it is smaller than the requested batch size.
    )


def calculate_distance_for_pair(seq_a: str, seq_b: str, distance : str = "kimura") -> float | None:
    """
    Calculate one pair's distance, but return None if the pair is invalid.
    """
    aligned_a, aligned_b = align_sequences(seq_a, seq_b)
    cleaned_a, cleaned_b = preprocess_sequences(aligned_a, aligned_b)

    if distance == "kimura":
        try:
            return float(kimura_distance(cleaned_a, cleaned_b))
        except (ValueError, ZeroDivisionError): # IF the pair is saturated or has no comparable sites
            return None


def train_alignment_one_epoch(alignment, loader, optimizer, device, temperature):
    alignment.train()
    total_loss = 0.0
    steps = 0

    for images, input_ids, attention_mask, _labels, _sequences, _sample_ids in loader:
        if images.shape[0] < 2:
            continue

        images = images.to(device, non_blocking=True)
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)

        image_z, dna_z = alignment(images, input_ids, attention_mask)
        loss = symmetric_info_nce(image_z, dna_z, temperature=temperature)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        steps += 1

    if steps == 0:
        raise ValueError("No training batches had at least two specimens.")

    return total_loss / steps                                                                        # Return average loss over all batches. 


def train_distance_one_epoch(model, loader, optimizer, device, distance_cache, freeze_alignment=True):
    """one complete pass through the training dataset.."""
    model.train()

    if freeze_alignment:
        model["alignment"].eval()

    total_loss = 0.0
    total_abs_error = 0.0
    total_squared_error = 0.0
    total_pairs = 0

    for images, input_ids, attention_mask, _labels, sequences, sample_ids in loader:
        batch_size = len(sequences)
        if batch_size < 2:
            continue

        images = images.to(device, non_blocking=True)
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)

        query_ids = torch.arange(batch_size, device=device)                                     # Select one random target from this batch for each query .
        offset = torch.randint(batch_size - 1, (batch_size,), device=device)
        target_ids = offset + (offset >= query_ids).long()                                      # ensuring the target pair is not the same entry as the query pair

        image_z, dna_z = model["alignment"](images, input_ids, attention_mask)                  # Run model and encode each specimen once.
        fused = model["fusion"](image_z, dna_z)

        valid_queries = [] 
        valid_targets = [] 
        target_values = []

        for i, j in enumerate(target_ids.cpu().tolist()):                                       # Calculate/store or retrieve one sequence-distance target per sampled pair.
            calculated_distance = distance_cache.get(
                sample_ids[i], 
                sample_ids[j],
                sequences[i], 
                sequences[j],
                )
            
            if calculated_distance is not None:
                valid_queries.append(i)
                valid_targets.append(j)
                target_values.append(calculated_distance)
                                        
        if not target_values:
            continue

        query_ids = torch.tensor(valid_queries, device=device)
        target_ids = torch.tensor(valid_targets, device=device)

        predicted_distances = model["distance"](fused[query_ids], fused[target_ids]) 

        distance_targets = torch.tensor(
            target_values,
            dtype=predicted_distances.dtype,
            device=device,
        )

        loss = distance_prediction_loss(predicted_distances, distance_targets)
        errors = predicted_distances.detach() - distance_targets
        total_abs_error += errors.abs().sum().item()
        total_squared_error += errors.square().sum().item()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        number_of_pairs = len(target_values)
        total_loss += loss.item() * number_of_pairs
        total_pairs += number_of_pairs

        average_epoch_loss = total_loss / total_pairs
        mae = total_abs_error / total_pairs
        mse = total_squared_error / total_pairs
        rmse = mse ** 0.5

    if total_pairs == 0:
        raise ValueError("No valid distance pairs were found.")
    
    return average_epoch_loss, mae, mse, rmse   
    
def main():
    ### python run_distance_prediction_train.py --train-csv ./data/train_manifest.csv --output ./distance.pt --train-alignment
    parser = argparse.ArgumentParser()                                                                      # Arguments coming from the command line.
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--output", default="distance.pt")
    parser.add_argument("--dna-model", default="bioscan-ml/BarcodeBERT")
    parser.add_argument("--shared-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42069)
    parser.add_argument("--train-alignment", action=argparse.BooleanOptionalAction, default=True)           # specify "--no-train-alignment" to not train alignment
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--cache-size", type=int, default=20000)
    args = parser.parse_args()                                                                              # Creates an object containing all the command line arguments. 

    set_seed(args.seed)                                                                                     # Call the helper function to set the random seed. 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")                                   # Use Cuda if GPU is available, otherwise use the CPU.
    
    species_to_ids = make_species_to_ids(args.train_csv)
    model = build_model(                                                                                    # Creates the model components.
        dna_model_name=args.dna_model,
        shared_dim=args.shared_dim,
        hidden_dim=args.hidden_dim,
    ).to(device) 

    loader = make_loader(                                                                                   # Creates the dataset and dataloader from above. 
        args.train_csv,
        model,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        species_to_ids=species_to_ids,
    )

    alignment_training_losses = []  
    distance_training_losses = []                                                                                    # Create empty lists to store training values per epoch                                                                              

    if args.train_alignment:
        print("Training Alignment Model...")
        alignment = model["alignment"]
        alignment_optimizer = torch.optim.AdamW(
            (p for p in alignment.parameters() if p.requires_grad), lr=args.learning_rate)

        for epoch in range(1, args.epochs + 1):
            loss = train_alignment_one_epoch(
                alignment,
                loader,
                alignment_optimizer,
                device,
                temperature=args.temperature,
            )
            alignment_training_losses.append(loss)
            print(f"epoch={epoch:03d} train_loss={loss:.5f}") 

        model["alignment"].requires_grad_(False)                                                            # Freeze the pretrained alignment features for distance training.
        freeze_alignment = True

        distance_parameters = (
            list(model["fusion"].parameters())
            + list(model["distance"].parameters())
        )
    else:
        freeze_alignment = False
        distance_parameters = [p for p in model.parameters() if p.requires_grad]                          # Train alignment, fusion, and distance model together.

    
    print("Training Distance Model...")
    distance_optimizer = torch.optim.AdamW(distance_parameters, lr=args.learning_rate)
    target_distance_cache = DistanceCache(max_entries=args.cache_size)
    for epoch in range(1, args.epochs + 1):                                                                 # Distance prediction Training loop
        loss, mae, mse, rmse = train_distance_one_epoch(
            model,
            loader,
            distance_optimizer,
            device,
            target_distance_cache,
            freeze_alignment,
        )
        distance_training_losses.append(loss)
        print(f"epoch={epoch:03d} train_loss={loss:.5f} mean_absolute_error={mae:.5f} mean_squared_error={mse:.5f} Root_mean_squared_error={rmse:.5f}")

    show_curve(distance_training_losses)
        
    output_path = Path(args.output)                                                                         # Turns the output filename into a path
    output_path.parent.mkdir(parents=True, exist_ok=True)                                                   # Create parent directory if it does not already exist. 
    # UPDATE TO SAVE ONLY STUFF I NEED LATER!
    torch.save(                                                                                             # Create PyTorch checkpoint
        {
            "model_weights":  model.state_dict(),
            "dna_model_name": args.dna_model,
            "shared_dim": args.shared_dim,
            "hidden_dim": args.hidden_dim,
            "species_mapping": species_to_ids,
        },
        output_path,
    )
    print(f"Saved projection heads to {output_path.resolve()}")

if __name__ == "__main__":
    main()
