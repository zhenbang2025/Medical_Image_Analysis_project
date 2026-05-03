"""
Medical CLIP baseline placeholder.

This script is intentionally pluggable because the exact MedCLIP/BiomedCLIP
implementation and weights vary. Fill in the TODOs once the target repo/model
is selected.
"""

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="./data/chest_xray")
    parser.add_argument("--split_file", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="./output")
    args = parser.parse_args()

    raise NotImplementedError(
        "Medical CLIP baseline is a pluggable stub. "
        "Choose a repo/model (MedCLIP/BiomedCLIP/OpenCLIP) and implement loading, "
        "then compute zero-shot or probe metrics with the same format as other scripts."
    )


if __name__ == "__main__":
    main()
