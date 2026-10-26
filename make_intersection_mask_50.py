import argparse
import os
import pickle
import random

import numpy as np


TRAFFIC_NAME = "anon_4_4_hangzhou_real"
REFERENCE_MASK = os.path.join(
    "memory", "fourphase", "kmint", f"{TRAFFIC_NAME}_kmint.pkl"
)
NUM_INTERSECTIONS = 16
NUM_ROWS = 4
NUM_COLS = 4


def random_intersection_order(seed):
    # Names are column-major, so the list index equals DiffLight's intersection index.
    names = [
        f"intersection_{col}_{row}"
        for col in range(1, NUM_COLS + 1)
        for row in range(1, NUM_ROWS + 1)
    ]
    random.Random(seed).shuffle(names)
    return names


def intersection_indices(layout, seed=0, count=8):
    if layout == "center":
        return sorted(
            (col - 1) * NUM_ROWS + (row - 1)
            for col in (2, 3)
            for row in (2, 3)
        )
    if layout == "border":
        center = set(intersection_indices("center"))
        return [index for index in range(NUM_INTERSECTIONS) if index not in center]
    if layout == "random":
        # Masked set = the last `count` entries of the shuffled order, so lower counts are nested subsets.
        masked_names = random_intersection_order(seed)[::-1][:count]
        return [
            (int(name.split("_")[1]) - 1) * NUM_ROWS + (int(name.split("_")[2]) - 1)
            for name in masked_names
        ]
    raise ValueError(f"Unsupported layout: {layout}")


def main():
    parser = argparse.ArgumentParser(
        description="Create a DiffLight-format Hangzhou intersection mask."
    )
    parser.add_argument(
        "--layout",
        choices=("center", "border", "random"),
        default="random",
        help="mask the central 4, border 12, or a seeded random subset",
    )
    parser.add_argument("--seed", type=int, default=0, help="seed for --layout random")
    parser.add_argument(
        "--count", type=int, default=8, help="number of masked intersections for --layout random"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace the generated mask file if it already exists",
    )
    args = parser.parse_args()

    missing_indices = intersection_indices(args.layout, args.seed, args.count)
    missing_count = len(missing_indices)
    pattern = {"center": "kmcenter", "border": "kmborder", "random": f"kmrand{args.seed}"}[args.layout]
    mask_key = f"{pattern}_{missing_count}"
    mask_directory = os.path.join("memory", "fourphase", pattern)
    output_mask = os.path.join(mask_directory, f"{TRAFFIC_NAME}_{pattern}.pkl")

    existing = {}
    if os.path.exists(output_mask):
        with open(output_mask, "rb") as existing_file:
            existing = pickle.load(existing_file)
        already_there = any(mask_key in masks for masks in existing.values())
        if already_there and not args.overwrite:
            raise FileExistsError(f"{mask_key} already in {output_mask}; use --overwrite to replace it")

    with open(REFERENCE_MASK, "rb") as reference_file:
        reference = pickle.load(reference_file)

    output = {}
    for round_name, rate_masks in reference.items():
        example_mask = np.asarray(next(iter(rate_masks.values())))
        if example_mask.ndim != 3 or example_mask.shape[0] != NUM_INTERSECTIONS:
            raise ValueError(f"Unexpected mask shape in {round_name}: {example_mask.shape}")

        mask = np.ones(example_mask.shape, dtype=np.uint8)
        mask[missing_indices, :, :] = 0
        output[round_name] = {**existing.get(round_name, {}), mask_key: mask}

    os.makedirs(mask_directory, exist_ok=True)
    temporary_path = f"{output_mask}.tmp"
    with open(temporary_path, "wb") as output_file:
        pickle.dump(output, output_file, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary_path, output_mask)

    for round_name, masks in output.items():
        mask = masks[mask_key]
        entirely_missing = np.all(mask == 0, axis=(1, 2))
        if int(entirely_missing.sum()) != missing_count:
            raise RuntimeError(f"50% validation failed in {round_name}")
        if not np.all(mask[~entirely_missing] == 1):
            raise RuntimeError(f"Unexpected partial mask in {round_name}")

    print(f"Mask file: {output_mask}")
    print(f"Mask key for DiffLight: {mask_key}")
    print(f"Missing intersections (indices 0-based): {missing_indices}")
    print(f"Missing count: {missing_count}/{NUM_INTERSECTIONS}")
    if args.layout == "random":
        print(f"Random-mask seed: {args.seed}")
    print(f"Mask shape per round: {next(iter(output.values()))[mask_key].shape}")
    print("Original trajectories and existing masks were not modified.")
    print(
        f"Train using -missing_pattern {mask_key}; keep the dataset flag and "
        "the usual DiffLight training command."
    )


if __name__ == "__main__":
    main()
