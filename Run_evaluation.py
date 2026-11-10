#!/usr/bin/env python

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as FUNC_OP

from Run_classification_train import build_model, make_loader


@torch.inference_mode()
def evaluate_model(model, loader, device, num_classes):
    model.eval()

    total_loss = 0.0                                                                            # Set the loss counter and the final 
    all_labels = []
    all_predictions = []
    all_logits = []
    sample_ids = []

    for images, input_ids, attention_mask, labels, ids in loader:
        images = images.to(device, non_blocking=True)
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        image_z, dna_z = model["alignment"](
            images, input_ids, attention_mask
        )
        fused = model["fusion"](image_z, dna_z)
        logits = model["classifier"](fused)

        # Sum batch losses so the final result can be averaged by specimen.
        total_loss += FUNC_OP.cross_entropy(
            logits, labels, reduction="sum"
        ).item()

        predictions = logits.argmax(dim=1)

        all_labels.append(labels.cpu())
        all_predictions.append(predictions.cpu())
        all_logits.append(logits.cpu())
        sample_ids.extend(list(ids))

    if not all_labels:
        raise ValueError("The evaluation manifest contains no samples.")

    labels = torch.cat(all_labels)
    predictions = torch.cat(all_predictions)
    logits = torch.cat(all_logits)

    metrics, confusion = classification_metrics(
        labels, predictions, num_classes
    )
    metrics["loss"] = total_loss / len(labels)

    return sample_ids, labels, predictions, logits, metrics, confusion


def classification_metrics(labels, predictions, num_classes):
    # Rows are true classes; columns are predicted classes.
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)

    for true_label, predicted_label in zip(labels, predictions):
        confusion[true_label, predicted_label] += 1

    support = confusion.sum(dim=1)
    predicted_count = confusion.sum(dim=0)
    true_positive = confusion.diag()

    precision = true_positive.float() / predicted_count.clamp_min(1)
    recall = true_positive.float() / support.clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)

    present_classes = support > 0

    metrics = {
        "accuracy": (true_positive.sum().float() / support.sum()).item(),
        "macro_f1": f1[present_classes].mean().item(),
    }
    return metrics, confusion


def main():
    # python Run_evaluation.py --manifest ./data/val_manifest.csv --checkpoint ./classification.pt
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="Held-out labeled CSV")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-predictions", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # This assumes the checkpoint contains the full ModuleDict state
    # and the same model configuration used during training.
    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=True,
    )

    species_to_idx = checkpoint["species_mapping"]

    model = build_model(
        dna_model_name=checkpoint["dna_model_name"],
        shared_dim=checkpoint["shared_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        num_species=len(species_to_idx),
    )
    model.load_state_dict(checkpoint["model_weights"])
    model = model.to(device).eval()

    loader = make_loader(
        manifest_path = args.manifest,
        model=model,
        species_to_idx=species_to_idx,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    sample_ids, labels, predictions, logits, metrics, confusion = (
        evaluate_model(
            model,
            loader,
            device,
            num_classes=len(species_to_idx),
        )
    )

    idx_to_species = {
        index: species
        for species, index in species_to_idx.items()
    }

    print(f"Evaluated {len(sample_ids)} specimens")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    # print("\nConfusion matrix: rows=true species, columns=predicted species")
    # print("Class order:", [idx_to_species[i] for i in range(len(species_to_idx))])
    print(confusion)

    if args.save_predictions:
        output_path = Path(args.save_predictions)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "sample_ids": sample_ids,
                "true_labels": labels,
                "predicted_labels": predictions,
                "logits": logits,
                "metrics": metrics,
                "confusion_matrix": confusion,
                "species_mapping": species_to_idx,
            },
            output_path,
        )
        print(f"Saved predictions to {output_path.resolve()}")

if __name__ == "__main__":
    main()
