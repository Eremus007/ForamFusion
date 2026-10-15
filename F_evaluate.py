#!/usr/bin/env python

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from E_alignment import build_model, make_loader


@torch.inference_mode()                                                                         # Disable gradient tracking (since we are evaluation and gradients are not updated.)
def encode_manifest(model, loader, device):                                                     
    model.eval()                                                                                # Switches model to evaluation mode
    image_vectors = []                                                                          # Image embeddings from each batch 
    dna_vectors = []                                                                            # DNA embeddings from each batch 
    sample_ids = []                                                                             # IDs identifying the specimens. 

    for images, input_ids, attention_mask, ids in loader:
        images = images.to(device, non_blocking=True)
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)

        image_z, dna_z = model(images, input_ids, attention_mask)
        image_vectors.append(image_z.cpu())
        dna_vectors.append(dna_z.cpu())
        sample_ids.extend(list(ids))

    return sample_ids, torch.cat(image_vectors), torch.cat(dna_vectors)


def retrieval_metrics(image_z, dna_z, ks=(1, 5, 10)):
    if image_z.shape != dna_z.shape:
        raise ValueError("Expected one matched image and DNA vector per sample.")

    similarity = image_z @ dna_z.T
    n = similarity.shape[0]
    targets = torch.arange(n)
    results = {}

    for direction, scores in (
        ("image_to_dna", similarity),
        ("dna_to_image", similarity.T),
    ):
        ranks = (scores.argsort(dim=1, descending=True) == targets[:, None])
        rank_positions = ranks.float().argmax(dim=1) + 1
        for k in ks:
            if n >= k:
                results[f"{direction}_recall@{k}"] = (
                    rank_positions <= k
                ).float().mean().item()
        results[f"{direction}_median_rank"] = rank_positions.float().median().item()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="Held-out paired CSV")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-embeddings", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model(
        checkpoint["dna_model_name"],
        checkpoint["shared_dim"],
    )
    model.image_projection.load_state_dict(checkpoint["image_projection"])
    model.dna_projection.load_state_dict(checkpoint["dna_projection"])
    model = model.to(device).eval()

    loader = make_loader(
        args.manifest,
        model,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    sample_ids, image_z, dna_z = encode_manifest(model, loader, device)
    metrics = retrieval_metrics(image_z, dna_z)

    print(f"Evaluated {len(sample_ids)} paired specimens")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    if args.save_embeddings:
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
