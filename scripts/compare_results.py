"""
Compare multiple result JSON files (MLP and/or VLM baseline outputs).
"""

import argparse
import os

from utils.env_utils import env_default
from utils.run_utils import load_json, prepare_run_dir, save_json


def read_metrics(path: str) -> dict:
    d = load_json(path)
    if "test_mean_iou" in d:
        return {
            "mean_iou": d.get("test_mean_iou", 0.0),
            "iou_at_0.25": d.get("test_iou_at_0.25", 0.0),
            "iou_at_0.5": d.get("test_iou_at_0.5", 0.0),
            "per_class": d.get("test_per_class", {}),
        }
    return {
        "mean_iou": d.get("mean_iou", 0.0),
        "iou_at_0.25": d.get("iou_at_0.25", 0.0),
        "iou_at_0.5": d.get("iou_at_0.5", 0.0),
        "per_class": d.get("per_class", {}),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", nargs="+", required=True, help="Display labels for each result file.")
    parser.add_argument("--files", nargs="+", required=True, help="Result JSON files, aligned with --labels.")
    parser.add_argument("--output_root", type=str, default=env_default("OUTPUT_COMPARE_ROOT", "./output/compare"))
    parser.add_argument("--run_name", type=str, default=env_default("RUN_NAME"))
    parser.add_argument("--allow_overwrite", action="store_true")
    args = parser.parse_args()

    if len(args.labels) != len(args.files):
        raise ValueError("--labels and --files must have the same length.")

    run_dir = prepare_run_dir(args.output_root, args.run_name, prefix="compare", allow_overwrite=args.allow_overwrite)
    methods = {}
    for label, f in zip(args.labels, args.files):
        methods[label] = read_metrics(f)

    print("\n=== Overall ===")
    print(f"{'Method':<25} {'MeanIoU':>10} {'@0.25':>10} {'@0.5':>10}")
    for k, v in methods.items():
        print(f"{k:<25} {v['mean_iou']:>10.4f} {v['iou_at_0.25']:>10.4f} {v['iou_at_0.5']:>10.4f}")

    classes = sorted({c for m in methods.values() for c in m["per_class"].keys()})
    for cls in classes:
        print(f"\n=== {cls} ===")
        print(f"{'Method':<25} {'MeanIoU':>10} {'@0.25':>10} {'@0.5':>10}")
        for k, v in methods.items():
            c = v["per_class"].get(cls, {})
            print(
                f"{k:<25} "
                f"{c.get('mean_iou', 0.0):>10.4f} "
                f"{c.get('iou_at_0.25', c.get('acc_0.25', 0.0)):>10.4f} "
                f"{c.get('iou_at_0.5', c.get('acc_0.5', 0.0)):>10.4f}"
            )

    save_json(os.path.join(run_dir, "comparison.json"), {"methods": methods, "classes": classes, "files": args.files})
    print(f"\nSaved comparison to: {run_dir}")


if __name__ == "__main__":
    main()
