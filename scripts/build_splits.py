"""
Build deterministic train/val splits and save to a JSON file.
"""

import argparse
import os

from data_utils import list_split_samples, make_train_val_split, save_split_file, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="./data/chest_xray")
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_file",
        type=str,
        default="./splits/full_split.json",
        help="Output JSON file for train/val/test splits",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    train_samples = list_split_samples(args.dataset_dir, "train")
    test_samples = list_split_samples(args.dataset_dir, "test")

    train_split, val_split = make_train_val_split(
        train_samples, args.val_fraction, args.seed
    )

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    save_split_file(
        args.output_file,
        {
            "train": train_split,
            "val": val_split,
            "test": test_samples,
        },
    )

    print("Split file saved:", args.output_file)
    print("Train:", len(train_split), "Val:", len(val_split), "Test:", len(test_samples))


if __name__ == "__main__":
    main()
