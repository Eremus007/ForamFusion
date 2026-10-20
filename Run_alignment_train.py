#!/usr/bin/env python

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch

from PIL import Image
from torch.utils.data import DataLoader, Dataset

from B_encoders import (
    BarcodeBERTEncoder,
    ResNet50Encoder,
    normalize_dna_sequence,
)
from C_model import PairedAlignmentModel
from D_loss import symmetric_info_nce, symmetric_accuracy


class PairedSpecimenDataset(Dataset):
    """Takes Manifest CSV and turns it into paired training examples."""
    """One CSV row per specimen with a matched image and DNA sequence."""

    REQUIRED_COLUMNS = {"processid", "sampleid", "species", "dna_barcode"}

    def __init__(self, manifest_path: str | Path, image_transform, dna_tokenizer):
        self.manifest_path = Path(manifest_path).resolve()                                                  # Resolve CSV path into an absolute path.
        with self.manifest_path.open("r", newline="", encoding="utf-8-sig") as f:                           # Open/Read CSV
            reader = csv.DictReader(f)                                                                      # Turns each row of the CSV into a python dictionary.
            missing = self.REQUIRED_COLUMNS - set(reader.fieldnames or [])                                  # Check missing columns.
            if missing:
                raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
            self.rows = list(reader)

        if not self.rows:
            raise ValueError(f"No samples found in {self.manifest_path}")

        sample_ids = [row["sampleid"] for row in self.rows]                                                # Creates a list of sample IDs
        if len(set(sample_ids)) != len(sample_ids):                                                         # Check for and remove duplicates
            raise ValueError(
                "This starter version expects one row per unique sample_id. "
                "Handle multiple views with a multi-positive loss or aggregate them first."
            )

        self.image_transform = image_transform                                                              # Stores the image preprocessing function in the dataset object.
        self.dna_tokenizer = dna_tokenizer                                                                  # Stores the BarcodeBERT tokenizer in the dataset object.

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
        return image, input_ids, attention_mask, row["sampleid"]                                            # Returns everything needed from the requested sample. 


def set_seed(seed: int):
    """Helper function to set random seed for reproducibility."""                                                                                    
    random.seed(seed)                                                                                       # Set Python's random number generator
    np.random.seed(seed)                                                                                    # Set Numpy's random number generator
    torch.manual_seed(seed)                                                                                 # Set PyTorch's random seed
    if torch.cuda.is_available():                                                                           # If we use CUDA...
        torch.cuda.manual_seed_all(seed)                                                                    # Set seed for CUDA random number generator as well. 


def build_model(dna_model_name: str, shared_dim: int, freeze_resnet: bool = True, train_BarcodeBERT: bool = False) -> PairedAlignmentModel:
    """This function builds the complete multimodal model."""
    image_encoder = ResNet50Encoder(freeze_backbone=freeze_resnet)                                          # Constructs RESNET-50 image encoder
    dna_encoder = BarcodeBERTEncoder(model_name=dna_model_name, train_backbone=train_BarcodeBERT)           # Constructs BarcodeBERT DNA encoder

    if freeze_resnet and not train_BarcodeBERT:                                                             # Sets the encoders to frozen on model level if both encoder settings are frozen. 
        freeze_encoders = True
    else:
        freeze_encoders = False

    return PairedAlignmentModel(                                                                            # Create alignmentmodel
        image_encoder=image_encoder,
        dna_encoder=dna_encoder,
        shared_dim=shared_dim,
        freeze_encoders=freeze_encoders,
    )


def make_loader( 
        manifest_path: str | Path,                                                          
        model: PairedAlignmentModel,
        batch_size: int,
        shuffle: bool,
        num_workers: int = 0,
    ) -> DataLoader:
    """Wraps the dataset object in a PyTorch dataloader object."""
    
    dataset = PairedSpecimenDataset(manifest_path, model.image_encoder.preprocess, model.dna_encoder.tokenizer)

    return DataLoader(dataset,                                                                              # Return dataloader object.
        batch_size=batch_size,                                                                              # How many image/dna pair samples per batch. 
        shuffle=shuffle,                                                                                    # Shuffle the dataset during each epoch, leads to more stable training. 
        num_workers=num_workers,                                                                            # How many subprocesses we have to load data.
        pin_memory=torch.cuda.is_available(),                                                               # If Cuda is available, enables pinned memory.
        drop_last=False,                                                                                    # Final batch is kept even if it is smaller than the requested batch size.
    )


def train_one_epoch(model, loader, optimizer, device, temperature, calc_accuracy):
    """This function performs one complete pass through the training dataset."""
    model.train()                                                                                           # Switch model to training mode.

    # Create counters starting at zero
    total_loss = 0.0                                                                                        # Accumulates the losses
    total_accuracy = 0.0
    steps = 0                                                                                               # counts how many batches contributed to training.

    for images, input_ids, attention_mask, _sample_ids in loader:                                           # Iterate through batches
        if images.shape[0] < 2:                                                                             # Checks whether batches have fewer than 2 samples and if it is, skip it.
            # A final singleton batch cannot form an in-batch negative.
            continue

        images = images.to(device, non_blocking=True)                                                       # Moves the image tensor to the selected device.
        input_ids = input_ids.to(device, non_blocking=True)                                                 # Moves the DNA token ids tensor to the selected device. 
        attention_mask = attention_mask.to(device, non_blocking=True)                                       # Moves the attention mask to the selected device. 

        image_z, dna_z = model(images, input_ids, attention_mask)                                           # Run the multimodal model and return shared embeddings space. 
        loss = symmetric_info_nce(image_z, dna_z, temperature=temperature)                                  # Calculate and store loss
        if calc_accuracy:                                                                                   
            accuracy = symmetric_accuracy(image_z, dna_z, temperature=temperature)                          # Calculate and store accuracy if specified in arguments

        optimizer.zero_grad(set_to_none=True)                                                               # Clear old gradients to prevent them from accumulating.
        loss.backward()                                                                                     # Backpropagation to calculate gradients
        optimizer.step()                                                                                    # Updates the parameters based on gradients. 

        total_loss += loss.item()                                                                           # loss is a PyTorch Tensor, .item() makes it a normal python number. 
        if calc_accuracy:                                
            total_accuracy += accuracy.item() 
        steps += 1                                                                                          # increment the batch counter.

    if steps == 0:                                                                                          # If every batch was skipped. 
        raise ValueError("No training batches had at least two specimens.")
    if calc_accuracy:
         return (total_loss / steps), (total_accuracy / steps)     
    else:
        return (total_loss / steps), None                                                                   # Return average loss over all batches. 

def show_curve(values, title="Loss"):
    plt.plot(values)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training {title} per epoch')
    plt.show()


def main():
    # python run_alignment_train.py --train-csv ./data/manifest.csv --output ./alignment.pt
    parser = argparse.ArgumentParser()                                                                      # Arguments coming from the command line.
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--output", default="alignment.pt")
    parser.add_argument("--dna-model", default="bioscan-ml/BarcodeBERT")
    parser.add_argument("--shared-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42069)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--calc_accuracy", dest="calc_accuracy", action="store_true")
    group.add_argument("--no_calc_accuracy", dest="calc_accuracy", action="store_false")
    parser.set_defaults(calc_accuracy=True)
    args = parser.parse_args()                                                                              # Creates an object containing all the command line arguments. 

    print(args.calc_accuracy)
    set_seed(args.seed)                                                                                     # Call the helper function to set the random seed. 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")                                   # Use Cuda if GPU is available, otherwise use the CPU.
    model = build_model(args.dna_model, args.shared_dim).to(device)                                         # Build the model and move to selected device. 
    loader = make_loader(                                                                                   # Creates the dataset and dataloader from above. 
        args.train_csv,
        model,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    training_losses = []                                                                                    # Create empty lists to store training values per epoch
    training_accuracy = []                                                                                  

    trainable_parameters = [p for p in model.parameters() if p.requires_grad]                               # Collects parameters that should actually be updated.
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.learning_rate)                              # Creates an AdamW optimizer using those trainable parameters.

    for epoch in range(1, args.epochs + 1):                                                                 # Training loop
        loss, accuracy = train_one_epoch(
            model,
            loader,
            optimizer,
            device,
            temperature=args.temperature,
            calc_accuracy=args.calc_accuracy,
        )
        training_losses.append(loss)
        if accuracy is not None:
            training_accuracy.append(accuracy)

        if accuracy is not None:
            print(f"epoch={epoch:03d} train_loss={loss:.5f}, train_accuracy={accuracy:.5f}")                    # Print progress.
        else:
            print(f"epoch={epoch:03d} train_loss={loss:.5f}") 
    show_curve(training_losses)
    if accuracy is not None:
        show_curve(training_accuracy, title="Accuracy")
        


    output_path = Path(args.output)                                                                         # Turns the output filename into a path
    output_path.parent.mkdir(parents=True, exist_ok=True)                                                   # Create parent directory if it does not already exist. 
    # The pretrained encoders are frozen, so save only the learned heads.
    torch.save(                                                                                             # Create PyTorch checkpoint
        {
            "image_projection": model.image_projection.state_dict(),
            "dna_projection": model.dna_projection.state_dict(),
            "dna_model_name": args.dna_model,
            "shared_dim": args.shared_dim,
            "temperature": args.temperature,
        },
        output_path,
    )
    print(f"Saved projection heads to {output_path.resolve()}")


if __name__ == "__main__":
    main()
