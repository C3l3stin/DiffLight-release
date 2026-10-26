import os
import pickle
import sys

import numpy as np

import run_difflight


TRAFFIC_NAME = "anon_4_4_hangzhou_real"
MASK_TEMPLATE = "memory/fourphase/kmint/anon_4_4_hangzhou_real_kmint.pkl"
MASK_DIRECTORY = "memory/fourphase/kmew"
MASK_FILE = os.path.join(MASK_DIRECTORY, f"{TRAFFIC_NAME}_kmew.pkl")


def create_mask_file():
    with open(MASK_TEMPLATE, "rb") as source_file:
        template = pickle.load(source_file)

    mask_set = {}
    for sample_key, sample_masks in template.items():
        shape = np.asarray(next(iter(sample_masks.values()))).shape
        if len(shape) != 3 or shape[0] != 16:
            raise ValueError(f"Unexpected source-mask shape for {sample_key}: {shape}")
        lane_mask = np.ones((*shape, 12), dtype=np.uint8)
        lane_mask[..., :6] = 0
        mask_set[sample_key] = {"kmew_50": lane_mask}

    os.makedirs(MASK_DIRECTORY, exist_ok=True)
    with open(MASK_FILE, "wb") as mask_file:
        pickle.dump(mask_set, mask_file, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Generated W/E lane mask: {MASK_FILE}")
    print("Mask value 0 = missing; lane indices 0-5 (W/E) are missing, 6-11 (N/S) observed.")


def main():
    create_mask_file()
    sys.argv = [
        "run_difflight.py",
        "-hangzhou_1",
        "-missing_pattern",
        "kmew_50",
        "-memo",
        "DiffLight_EW",
    ]
    run_difflight.args = run_difflight.parse_args()
    run_difflight.main(run_difflight.args)


if __name__ == "__main__":
    main()