#!/usr/bin/env python

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as FUNC_OP

from Run_distance_prediction_train import build_model, make_loader, set_seed, DistanceCache
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
from E_visualisation import show_umap_features

@torch.inference_mode()
def evaluate_model(model, loader, device, distance_cache, pair_seed):
    model.eval()
    pair_generator = torch.Generator(device=device)
    pair_generator.manual_seed(pair_seed)

    total_loss = 0.0
    total_pairs = 0
    all_predictions = []
    all_targets = []
    pair_ids = []

    for images, input_ids, attention_mask, _labels, sequences, ids in loader:
        batch_size = len(sequences)
        if batch_size < 2:
            continue

        images = images.to(device, non_blocking=True)
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)

        query_ids = torch.arange(batch_size, device=device)
        offset = torch.randint(batch_size - 1, (batch_size,), device=device, generator=pair_generator)
        target_ids = offset + (offset >= query_ids).long()

        image_z, dna_z = model["alignment"](images, input_ids, attention_mask)
        fused = model["fusion"](image_z, dna_z)

        valid_queries = []
        valid_targets = []
        target_values = []
        valid_pair_ids = []

        for i, j in enumerate(target_ids.tolist()):
            distance = distance_cache.get(
                ids[i],
                ids[j],
                sequences[i],
                sequences[j],
            )
            if distance is not None:
                valid_queries.append(i)
                valid_targets.append(j)
                target_values.append(float(distance))
                valid_pair_ids.append((ids[i], ids[j]))
        if not target_values:
            continue

        query_ids = torch.tensor(
            valid_queries,
            dtype=torch.long,
            device=device,
        )

        target_ids = torch.tensor(
            valid_targets,
            dtype=torch.long,
            device=device,
        )

        predicted_distances = model["distance"](fused[query_ids], fused[target_ids]).reshape(-1)
        distance_targets = torch.tensor(target_values, dtype=predicted_distances.dtype, device=device).reshape(-1)

        if predicted_distances.shape != distance_targets.shape:
            raise ValueError(
                "Predicted distances and target distances have "
                f"different shapes: {predicted_distances.shape} "
                f"and {distance_targets.shape}"
            )

        loss = distance_prediction_loss(
            predicted_distances,
            distance_targets,
        )

        number_of_pairs = len(target_values)

        total_loss += loss.item() * number_of_pairs
        total_pairs += number_of_pairs

        all_predictions.append(predicted_distances.cpu())
        all_targets.append(distance_targets.cpu())

        pair_ids.extend(valid_pair_ids)

    if total_pairs == 0:
        raise ValueError(
            "No valid pairs were found. "
            "Check the validation dataset and distance cache"
        )

    predictions = torch.cat(all_predictions)
    targets = torch.cat(all_targets)
    errors = predictions - targets
    metrics = {
        "loss": total_loss / total_pairs,
        "mean_absolute_error": errors.abs().mean().item(),
        "mean_squared_error": errors.square().mean().item(),
        "root_mean_squared_error": errors.square().mean().sqrt().item(),
    }

    return pair_ids, targets, predictions, metrics


@torch.inference_mode()
def extract_embeddings(model, loader, device):
    model.eval()

    image_parts = []
    dna_parts = []
    fused_parts = []
    label_parts = []
    sample_ids = []

    for images, input_ids, attention_mask, labels, _sequences, ids in loader:
        images = images.to(device, non_blocking=True)
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)

        image_z, dna_z = model["alignment"](
            images, input_ids, attention_mask
        )
        fused_z = model["fusion"](image_z, dna_z)

        image_parts.append(image_z.cpu())
        dna_parts.append(dna_z.cpu())
        fused_parts.append(fused_z.cpu())
        label_parts.append(labels.cpu())
        sample_ids.extend(list(ids))

    return {
        "sample_ids": sample_ids,
        "labels": torch.cat(label_parts),
        "image_embeddings": torch.cat(image_parts),
        "dna_embeddings": torch.cat(dna_parts),
        "fused_embeddings": torch.cat(fused_parts),
    }

def main():
    # python Run_evaluation_distance.py --manifest ./data/val_manifest.csv --checkpoint ./distance.pt --visualize
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="Held-out labeled CSV")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-predictions", default=None)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--cache-size", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42069)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed) 

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    
    species_to_ids = checkpoint["species_mapping"]
    distance_cache = DistanceCache(max_entries=args.cache_size)

    model = build_model(
        dna_model_name=checkpoint["dna_model_name"],
        shared_dim=checkpoint["shared_dim"],
        hidden_dim=checkpoint["hidden_dim"],
    ).to(device)
    model.load_state_dict(checkpoint["model_weights"])
    model.eval()

    loader = make_loader(
        manifest_path=args.manifest,
        model=model,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        species_to_ids=species_to_ids,
    )

    if args.visualize:
        embeddings = extract_embeddings(model, loader, device)
        show_umap_features({"DNA": embeddings["dna_embeddings"], "Image": embeddings["image_embeddings"]}, embeddings["labels"])
        show_umap_features({"FUSED": embeddings["fused_embeddings"]}, embeddings["labels"])

    pair_ids, targets, predicted_distances, metrics = (
        evaluate_model(
            model,
            loader,
            device,
            distance_cache,
            args.seed,
        )
    )

    print(f"Evaluated {len(pair_ids)} pairs")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    if args.save_predictions:
        output_path = Path(args.save_predictions)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "pair_ids": pair_ids,
                "targets": targets,
                "predicted_distances": predicted_distances,
                "metrics": metrics,
                "species_mapping": species_to_ids,
            },
            output_path,
        )
        print(f"Saved predictions to {output_path.resolve()}")

if __name__ == "__main__":
    main()
