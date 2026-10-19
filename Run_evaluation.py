#!/usr/bin/env python

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from Run_alignment_train import build_model, make_loader


@torch.inference_mode()                                                                         # Disable gradient tracking (since we are evaluation and gradients are not updated.)
def encode_manifest(model, loader, device): 
    """reads in the input manifest."""                                                    
    model.eval()                                                                                # Switches model to evaluation mode
    image_vectors = []                                                                          # Image embeddings from each batch 
    dna_vectors = []                                                                            # DNA embeddings from each batch 
    sample_ids = []                                                                             # IDs identifying the specimens. 

    for images, input_ids, attention_mask, ids in loader:                                       # Iterates over the batches returned by the data loader
        images = images.to(device, non_blocking=True)                                           # Moves the three input tensors to the selected device
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)

        image_z, dna_z = model(images, input_ids, attention_mask)                               # get image and dna embeddings in a shared embedding space from the model. 
        image_vectors.append(image_z.cpu())                                                     # moves the image embeddings back to the CPU and appends it to the corresponding list.    
        dna_vectors.append(dna_z.cpu())                                                         # moves the dna embeddings back to the CPU and appends it to the corresponding list.    
        sample_ids.extend(list(ids))                                                            # Adds the specimen IDs from the current batch to the sample_ids

    return sample_ids, torch.cat(image_vectors), torch.cat(dna_vectors)                         # Returns All specimen IDs and All image and DNA embeddings concatenated corresponding tensors.


def retrieval_metrics(image_z, dna_z, ks=(1, 5, 10)):                                           # image_z: image embeddings, dna_z: dna embeddings, ks: the values of k used to calculate recall. 
    """This function measures if each image can retrieve 
    its matching DNA embedding from the dataset, and whether 
    each DNA embedding can retrieve its matching image."""
    if image_z.shape != dna_z.shape:                                                            # Checks whether embedding tensors have identical shapes.
        raise ValueError("Expected one matched image and DNA vector per sample.")

    similarity = image_z @ dna_z.T                                                              # Calculates the similarity between every image embedding and every DNA embedding. @ means matrix multiplication and .T transposes the dna tensor. 
    n = similarity.shape[0]                                                                     # Gets the number of rows in the similarity matrix, which is also the number of paired specimens. 
    targets = torch.arange(n)                                                                   # creates the tensor of correct-match indices. 
    results = {}                                                                                # Empty python dictionary to store the metrics.

    for direction, scores in (
        ("image_to_dna", similarity),                                                           # For loop evaluating Image-to-DNA retrieval.
        ("dna_to_image", similarity.T),                                                         # For loop evaluating DNA-to-Image retrieval transposing the image swaps the query and candidate modalities.
    ):
        ranks = (scores.argsort(dim=1, descending=True) == targets[:, None])                    # Produces a boolean matrix indicating where the correct candidate appears in each ranked list.
        rank_positions = ranks.float().argmax(dim=1) + 1                                        # Converts the boolean matrix into the numerical rank of each correct match. 
        for k in ks:                                                                            # Loop through the requested values of k.
            if n >= k:                                                                          # Only calculates recall if the datacet contains at least k candidates
                results[f"{direction}_recall@{k}"] = (                                          # The main retrieval metrics by checking whether the correct match appears in the top k candidates. 
                    rank_positions <= k
                ).float().mean().item()
        results[f"{direction}_median_rank"] = rank_positions.float().median().item()            # the median rank shows how far below rank 1 the correct matches tend to appear

    return results                                                                              # return the results dictionary. 


def main():
    parser = argparse.ArgumentParser()                                                          # arguments from the command line.
    parser.add_argument("--manifest", required=True, help="Held-out paired CSV")    
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-embeddings", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")                       # specify device 

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)            # load the saved checkpoint from the specified file. 
    model = build_model(                                                                        # constructs a model using the configuration stored in the checkpoint.
        checkpoint["dna_model_name"],
        checkpoint["shared_dim"],                                                               # You can load stored arguments and objects from the checkpoint.
    )
    model.image_projection.load_state_dict(checkpoint["image_projection"])                      # then you can load weights/states from certain objects from the checkpoint. 
    model.dna_projection.load_state_dict(checkpoint["dna_projection"])
    model = model.to(device).eval()                                                             # Moves the loaded parameters and buffers to the selected device (GPU or CPU) and then puts the model in evaluation mode.

    loader = make_loader(                                                                       # Creates the batch data loader for the CSV file.
        args.manifest,                                                                          # The CSV file to evaluate
        model,                                                                                  # include the model so the object can access the barcodebert tokenizer and image preprocess function.  
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    sample_ids, image_z, dna_z = encode_manifest(model, loader, device)                         # Get embeddings and sample ideas from the above function.
    metrics = retrieval_metrics(image_z, dna_z)                                                 # Calculate and retrieve metrics using the above function.

    print(f"Evaluated {len(sample_ids)} paired specimens")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    if args.save_embeddings:                                                                    # Save the embeddings if we specified to do so in the arguments.
        output_path = Path(args.save_embeddings)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "sample_ids": sample_ids,
                "image_embeddings": image_z,
                "dna_embeddings": dna_z,
            },
            output_path,
        )
        print(f"Saved embeddings to {output_path.resolve()}")


if __name__ == "__main__":
    main()
