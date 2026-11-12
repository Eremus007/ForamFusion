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

from B_encoders import (
    BarcodeBERTEncoder,
    ResNet50Encoder,
    normalize_dna_sequence,
)
from C_model import PairedAlignmentModel, FusionHead, ClassificationHead
from D_loss import symmetric_info_nce, symmetric_accuracy
from E_visualisation import show_curve

class PairedSpecimenDataset(Dataset):
    """Takes Manifest CSV and turns it into paired training examples."""
    """One CSV row per specimen with a matched image and DNA sequence."""

    CHECK_IF_EXIST = {"processid", "sampleid", "species", "dna_barcode"}

    def __init__(self, manifest_path: str | Path, image_transform, dna_tokenizer, species_to_idx: dict[str, int]):
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
        self.species_to_idx = species_to_idx

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
        label = torch.tensor(self.species_to_idx[species], dtype=torch.long)

        return image, input_ids, attention_mask, label, row["sampleid"]                                     # Returns everything needed from the requested sample. 

def make_species_to_idx(manifest_path):
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
    """Helper function to set random seed for reproducibility."""                                                                                    
    random.seed(seed)                                                                                       # Set Python's random number generator
    np.random.seed(seed)                                                                                    # Set Numpy's random number generator
    torch.manual_seed(seed)                                                                                 # Set PyTorch's random seed
    if torch.cuda.is_available():                                                                           # If we use CUDA...
        torch.cuda.manual_seed_all(seed)                                                                    # Set seed for CUDA random number generator as well. 


def build_model(dna_model_name: str, shared_dim: int, hidden_dim: int, num_species: int):
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

    classifier = ClassificationHead(
        num_species=num_species,
        hidden_dim=hidden_dim,
    )

    return nn.ModuleDict({
        "alignment": alignment,
        "fusion": fusion,
        "classifier": classifier,
    })


def make_loader( 
        manifest_path: str | Path,                                                          
        model: nn.ModuleDict,
        batch_size: int,
        shuffle: bool,
        species_to_idx: dict,
        num_workers: int = 0,
    ) -> DataLoader:
    """Wraps the dataset object in a PyTorch dataloader object."""

    alignment = model["alignment"]                                                                          
    dataset = PairedSpecimenDataset(
        manifest_path,
        alignment.image_encoder.preprocess,                                                                 # Get the encoders from the alignment model component.
        alignment.dna_encoder.tokenizer,
        species_to_idx,
    )

    return DataLoader(                                                                                      # Return dataloader object.
        dataset,                                                                             
        batch_size=batch_size,                                                                              # How many image/dna pair samples per batch. 
        shuffle=shuffle,                                                                                    # Shuffle the dataset during each epoch, leads to more stable training. 
        num_workers=num_workers,                                                                            # How many subprocesses we have to load data.
        pin_memory=torch.cuda.is_available(),                                                               # If Cuda is available, enables pinned memory.
        drop_last=False,                                                                                    # Final batch is kept even if it is smaller than the requested batch size.
    )

def train_alignment_one_epoch(alignment, loader, optimizer, device, temperature):
    alignment.train()
    total_loss = 0.0
    steps = 0

    for images, input_ids, attention_mask, _labels, _sample_ids in loader:
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

def train_classifier_one_epoch(model, loader, optimizer, device, freeze_alignment=True):
    """one complete pass through the training dataset.."""
    model.train()

    if freeze_alignment:
        model["alignment"].eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, input_ids, attention_mask, labels, _sample_ids in loader:
        images = images.to(device, non_blocking=True)
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        image_z, dna_z = model["alignment"](images, input_ids, attention_mask)
        fused = model["fusion"](image_z, dna_z)
        logits = model["classifier"](fused)

        loss = FUNC_OP.cross_entropy(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += labels.size(0)

    return total_loss / total_samples, total_correct / total_samples                                         # Return average loss and accuracy for this epoch.


def main():
    ### python run_classification_train.py --train-csv ./data/train_manifest.csv --output ./classification.pt --train-alignment
    parser = argparse.ArgumentParser()                                                                      # Arguments coming from the command line.
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--output", default="classification.pt")
    parser.add_argument("--dna-model", default="bioscan-ml/BarcodeBERT")
    parser.add_argument("--shared-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42069)
    parser.add_argument("--train-alignment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--temperature", type=float, default=0.07)
    args = parser.parse_args()                                                                              # Creates an object containing all the command line arguments. 

    set_seed(args.seed)                                                                                     # Call the helper function to set the random seed. 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")                                   # Use Cuda if GPU is available, otherwise use the CPU.
    
    species_to_idx = make_species_to_idx(args.train_csv)
    model = build_model(                                                                                    # Creates the model components.
        dna_model_name=args.dna_model,
        shared_dim=args.shared_dim,
        hidden_dim=args.hidden_dim,
        num_species=len(species_to_idx),
    ).to(device) 

    loader = make_loader(                                                                                   # Creates the dataset and dataloader from above. 
        args.train_csv,
        model,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        species_to_idx=species_to_idx,
    )

    alignment_training_losses = []  
    classification_training_losses = []                                                                                    # Create empty lists to store training values per epoch
    classification_training_accuracies = []                                                                                  

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

        model["alignment"].requires_grad_(False)                                                            # Freeze the pretrained alignment features for classifier training.
        freeze_alignment = True

        classifier_parameters = (
            list(model["fusion"].parameters())
            + list(model["classifier"].parameters())
        )
    else:
        freeze_alignment = False
        classifier_parameters = [p for p in model.parameters() if p.requires_grad]                          # Train alignment, fusion, and classifier together.

    
    print("Training Classification Model...")
    classifier_optimizer = torch.optim.AdamW(classifier_parameters, lr=args.learning_rate)
    for epoch in range(1, args.epochs + 1):                                                                 # Classification Training loop
        loss, accuracy = train_classifier_one_epoch(
            model,
            loader,
            classifier_optimizer,
            device,
            freeze_alignment,
        )
        classification_training_losses.append(loss)
        if accuracy is not None:
            classification_training_accuracies.append(accuracy)

        if accuracy is not None:
            print(f"epoch={epoch:03d} train_loss={loss:.5f}, train_accuracy={accuracy:.5f}")                    # Print progress.
        else:
            print(f"epoch={epoch:03d} train_loss={loss:.5f}") 
    show_curve(classification_training_losses)
    if accuracy is not None:
        show_curve(classification_training_accuracies, title="Accuracy")
        
    output_path = Path(args.output)                                                                         # Turns the output filename into a path
    output_path.parent.mkdir(parents=True, exist_ok=True)                                                   # Create parent directory if it does not already exist. 
    # The pretrained encoders are frozen, so save only the learned heads.
    torch.save(                                                                                             # Create PyTorch checkpoint
        {
            "model_weights":  model.state_dict(),
            "dna_model_name": args.dna_model,
            "shared_dim": args.shared_dim,
            "hidden_dim": args.hidden_dim,
            "species_mapping": species_to_idx,
        },
        output_path,
    )
    print(f"Saved projection heads to {output_path.resolve()}")

if __name__ == "__main__":
    main()
