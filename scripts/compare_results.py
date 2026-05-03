"""
Compare MLP vs Qwen baselines (zero-shot + few-shot) for bbox localization.

Usage:
    python compare_results.py
"""

import json
import os


def main():
    output_dir = "./output"

    def load(path):
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return None

    # Load 4 result files
    plain = load(os.path.join(output_dir, "plain", "bbox_eval.json"))
    residual = load(os.path.join(output_dir, "residual", "bbox_eval.json"))
    zs = load(os.path.join(output_dir, "zero_shot_eval.json"))
    fs = load(os.path.join(output_dir, "few_shot_3_eval.json"))

    methods = {}
    if plain:
        methods["MLP"] = plain
    if residual:
        methods["MLP+Residual"] = residual
    if zs:
        methods["Qwen Zero-shot"] = zs
    if fs:
        methods["Qwen Few-shot (n=3)"] = fs

    if not methods:
        print("No result files found.")
        return

    names = list(methods.keys())

    # Overall
    print(f"\n{'='*70}")
    print(f"BBOX LOCALIZATION: MLP vs Qwen Baselines")
    print(f"{'='*70}")
    print(f"\n{'Metric':<15}", end="")
    for n in names:
        print(f" {n:>18}", end="")
    print()
    print("-" * (15 + 18 * len(names)))

    for label, key in [("Mean IoU", "mean_iou"), ("IoU@0.25", "iou_at_0.25"), ("IoU@0.5", "iou_at_0.5")]:
        print(f"{label:<15}", end="")
        for n in names:
            print(f" {methods[n].get(key, 0):>18.4f}", end="")
        print()

    # Per-class detailed table
    classes = ["Atelectasis", "Effusion", "Cardiomegaly"]
    metrics = [("Mean IoU", "mean_iou"), ("IoU@0.25", "acc_0.25"), ("IoU@0.5", "acc_0.5")]

    for cls in classes:
        print(f"\n{'='*70}")
        print(f"CLASS: {cls}")
        print(f"{'='*70}")
        print(f"{'Metric':<15}", end="")
        for n in names:
            print(f" {n:>22}", end="")
        print()
        print("-" * (15 + 22 * len(names)))

        for label, key in metrics:
            print(f"{label:<15}", end="")
            for n in names:
                d = methods[n]
                cls_data = d.get(cls, {})
                if not isinstance(cls_data, dict) or key not in cls_data:
                    cls_data = d.get("per_class", {}).get(cls, {})
                val = cls_data.get(key, 0) if isinstance(cls_data, dict) else 0
                print(f" {val:>22.4f}", end="")
            print()

    # Improvements
    print(f"\n{'='*70}")
    print(f"MLP IMPROVEMENT vs BASELINES")
    print(f"{'='*70}")
    for mlp_name, mlp_d in [("MLP", plain), ("MLP+Residual", residual)]:
        if not mlp_d:
            continue
        for base_name, base_d in [("Qwen Zero-shot", zs), ("Qwen Few-shot (n=3)", fs)]:
            if not base_d:
                continue
            d = mlp_d["mean_iou"] - base_d["mean_iou"]
            pct = d / base_d["mean_iou"] * 100 if base_d["mean_iou"] > 0 else 0
            sign = "+" if d > 0 else ""
            print(f"  {mlp_name:<20s} vs {base_name:<25s} : {sign}{d:.4f} IoU ({sign}{pct:.1f}%)")

    print(f"\n{'='*70}")

    # Save
    comparison = {
        "methods": names,
        "overall": {
            label: {n: methods[n].get(key, 0) for n in names}
            for label, key in [("mean_iou", "mean_iou"), ("iou_at_0.25", "iou_at_0.25"), ("iou_at_0.5", "iou_at_0.5")]
        },
        "per_class": {
            cls: {
                n: (methods[n].get(cls, {}).get("mean_iou", 0)
                    if isinstance(methods[n].get(cls), dict) and "mean_iou" in methods[n].get(cls, {})
                    else methods[n].get("per_class", {}).get(cls, {}).get("mean_iou", 0))
                for n in names
            }
            for cls in classes
        },
    }
    with open(os.path.join(output_dir, "comparison.json"), "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"Saved to ./output/comparison.json")


if __name__ == "__main__":
    main()
