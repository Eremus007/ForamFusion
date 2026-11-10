#!/usr/bin/env python
from pathlib import Path
import shutil
import csv
import os
import csv
import random
from collections import defaultdict

def get_sequences(file="./csv/BIOSCAN_5M_Insect_Dataset_metadata.csv"):
    with open(file, 'r') as inputfile:
        line = inputfile.readline()
        while len(line) != 0:
            print(line[11], "\n")
            line = inputfile.readline().split(',')
        inputfile.close()


def get_line(file="./csv/BIOSCAN_5M_Insect_Dataset_metadata.csv"):
    with open(file, 'r') as inputfile:
        line = inputfile.readline()
        print(line)
        while len(line) != 0:
            # print(line, "\n")
            line = inputfile.readline().split(',')
        inputfile.close()

def write_to_file(output_path, line, overwrite=True):
     if overwrite:
          open(output_path, 'w').close()
     with open(output_path, "a") as outputfile:
          outputfile.write(line)

def convert_BIOSCAN_metadata(file="./csv/BIOSCAN_5M_Insect_Dataset_metadata.csv", output_path="./data/manifest.csv", num_pairs=10):
    counter = 0
    with open(file, 'r') as inputfile:
        line = inputfile.readline().split(',')
        write_to_file(output_path, ','.join([line[0], line[1], line[9], line[11]]) + '\n')
        print(f"Saved headerfile to {output_path}")
        while len(line) != 0:
        # for i in range (0, 100):
            line = inputfile.readline().split(',')
            # print(','.join((line[1], (line[8], line[9]), line[11])), '\n')
            if len(line[9]) != 0:
                counter += 1
                write_to_file(output_path, ','.join([line[0], line[1], line[9], line[11]]) + '\n', False)
            if counter == num_pairs:
                break
        inputfile.close()

def find_image(directory, filename, filetype=".jpg"):
    image_paths = list(directory.rglob(filename+filetype))
    return image_paths

def copy_image_to_directory(original_directory, target_directory):
    source_path = Path(original_directory)
    target_directory = Path(target_directory)
    target_directory.mkdir(parents=True, exist_ok=True)

    filename = original_directory.name
    target_path = target_directory / filename

    shutil.copy2(original_directory, target_path)

def prepare_manifest(images_directory, manifest_file="./data/manifest.csv"):
    with open(manifest_file, 'r') as inputfile:
            line = inputfile.readline()
            while len(line) != 0:
                processid = line[0]
                paths = find_image(images_directory,  processid)
                for path in paths:
                    copy_image_to_directory(path, "./data/images")
                
                line = inputfile.readline().split(',')
            inputfile.close()

def split_manifest(
    input_path,
    train_path,
    val_path,
    val_fraction=0.30,
    seed=42,
):
    input_path = Path(input_path)

    with input_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        required = {"sampleid", "species", "dna_barcode"}

        missing = required - set(fieldnames)
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

        rows = list(reader)

    # Keep complete examples that this image-plus-DNA classifier can use.
    rows = [
        row for row in rows
        if row["species"].strip() and row["dna_barcode"].strip()
    ]

    sample_ids = [row["sampleid"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Manifest has duplicate sample IDs.")

    groups = defaultdict(list)
    for row in rows:
        groups[row["species"].strip()].append(row)

    rng = random.Random(seed)
    train_rows = []
    val_rows = []
    singleton_species = 0

    for species_rows in groups.values():
        rng.shuffle(species_rows)

        if len(species_rows) == 1:
            # Keep singleton classes in training. They cannot be evaluated
            # as known classes if placed in validation.
            train_rows.extend(species_rows)
            singleton_species += 1
            continue

        n_val = round(len(species_rows) * val_fraction)
        n_val = max(1, min(n_val, len(species_rows) - 1))

        val_rows.extend(species_rows[:n_val])
        train_rows.extend(species_rows[n_val:])

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)

    for output_path, output_rows in (
        (train_path, train_rows),
        (val_path, val_rows),
    ):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(output_rows)

    print(f"Train: {len(train_rows)} specimens")
    print(f"Validation: {len(val_rows)} specimens")
    print(f"Singleton species kept in train: {singleton_species}")
    print(f"Saved: {train_path}")
    print(f"Saved: {val_path}")
    


if __name__ == "__main__":

    # output = "./data/manifest.csv"
    # with open("./csv/BIOSCAN_5M_Insect_Dataset_metadata.csv", 'r') as inputfile:
    #         header = inputfile.readline()
    #         write_to_file(Path(output), header)
    #         print(header)
    #         print(inputfile.readline())
    # paths = find_image(images_directory, "HELAC44435-21")
    # for path in paths:
    #     copy_image_to_directory(path, "./data/images")

    images_directory = Path.cwd().parent / "BIOSCAN"
    # prepare_manifest(images_directory)
    split_manifest(input_path="./data/manifest.csv", train_path="./data/train_manifest.csv", val_path="./data/val_manifest.csv")

    # convert_BIOSCAN_metadata(num_pairs=1000)
    print("completed")

