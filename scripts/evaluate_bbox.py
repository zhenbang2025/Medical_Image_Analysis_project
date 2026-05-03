"""
Evaluate unified bbox MLP and visualize results.

Loads the single bbox_mlp model, runs on test set,
computes per-class IoU, saves predictions and visualizations.

Usage:
    python evaluate_bbox.py [--n_show 10]
"""

import argparse
import json
import os
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image

matplotlib = None
plt = None
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
except ImportError:
    pass


class BBoxMLP(nn.Module):
    """MLP for bbox regression. Supports both plain and residual variants."""

    def __init__(self, input_dim, hidden_dim=256, dropout=0.2, use_residual=False):
        super().__init__()
        self.use_residual = use_residual

        if use_residual:
            self.input_proj = nn.Linear(input_dim, hidden_dim)
            self.res_block = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, hidden_dim),
            )
            self.skip_proj = nn.Linear(hidden_dim, hidden_dim)
            self.output = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 4),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 4),
            )

    def forward(self, x):
        if self.use_residual:
            h = self.input_proj(x)
            h = self.res_block(h) + self.skip_proj(h)
            return self.output(h)
        return self.net(x)


def compute_iou(pred, true):
    pred = pred.clamp(min=0)
    true = true.clamp(min=0)
    pred_x1, pred_y1 = pred[:, 0], pred[:, 1]
    pred_x2, pred_y2 = pred[:, 0] + pred[:, 2], pred[:, 1] + pred[:, 3]
    true_x1, true_y1 = true[:, 0], true[:, 1]
    true_x2, true_y2 = true[:, 0] + true[:, 2], true[:, 1] + true[:, 3]
    inter_x1 = torch.max(pred_x1, true_x1)
    inter_y1 = torch.max(pred_y1, true_y1)
    inter_x2 = torch.min(pred_x2, true_x2)
    inter_y2 = torch.min(pred_y2, true_y2)
    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h
    union_area = pred[:, 2] * pred[:, 3] + true[:, 2] * true[:, 3] - inter_area
    return inter_area / (union_area + 1e-6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to bbox_mlp.pt checkpoint. Auto-derives output_dir from its parent.")
    parser.add_argument("--embedding_dir", type=str, default="./embeddings")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--n_show", type=int, default=10)
    args = parser.parse_args()

    # Determine model path and output dir
    if args.model_path is None:
        args.model_path = "./output/plain/bbox_mlp.pt"

    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.model_path)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    ckpt = torch.load(args.model_path, weights_only=True)

    # Auto-detect architecture from checkpoint keys
    keys = ckpt["model_state_dict"].keys()
    use_residual = "input_proj.weight" in keys or ckpt.get("use_residual", False)
    model_tag = "residual" if use_residual else "plain"

    model = BBoxMLP(ckpt["input_dim"], ckpt["hidden_dim"], ckpt["dropout"], use_residual)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device)
    classes = ckpt["classes"]
    print(f"Model: {model_tag}")
    print(f"Loaded from {args.model_path}")
    print(f"Output dir: {args.output_dir}")
    print(f"  Classes: {classes}")

    # Viz dir inside output dir
    viz_dir = os.path.join(args.output_dir, "bbox_viz")
    os.makedirs(viz_dir, exist_ok=True)

    # Load class prompts
    prompts_path = os.path.join(args.embedding_dir, "class_prompts.json")
    class_prompts = {}
    if os.path.exists(prompts_path):
        with open(prompts_path) as f:
            class_prompts = json.load(f)

    # Load test data
    test_data = torch.load(
        os.path.join(args.embedding_dir, "bbox_test.pt"), weights_only=True
    )
    embeddings = test_data["embeddings"].float().to(device)
    bboxes = test_data["bboxes"].to(device)
    labels = test_data["labels"].to(device)

    # Load metadata
    with open(os.path.join(args.embedding_dir, "bbox_meta.json")) as f:
        meta = json.load(f)
    test_meta = meta["test"]

    # Predict
    with torch.no_grad():
        pred = model(embeddings)
    pred_clamped = pred.clamp(min=0)
    ious = compute_iou(pred_clamped, bboxes)

    # Summary
    print(f"\nTest samples: {len(bboxes)}")
    print(f"Mean IoU: {ious.mean():.4f}")
    print(f"IoU@0.25: {(ious >= 0.25).float().mean():.4f}")
    print(f"IoU@0.5:  {(ious >= 0.5).float().mean():.4f}")

    # Per-class
    print(f"\n{'Class':<20} {'Count':>6} {'Mean IoU':>10} {'@0.25':>8} {'@0.5':>8}")
    print("-" * 56)
    results = {}
    for i, cls in enumerate(classes):
        mask = labels == i
        if mask.sum() == 0:
            continue
        c_ious = ious[mask]
        a25 = (c_ious >= 0.25).float().mean().item()
        a50 = (c_ious >= 0.5).float().mean().item()
        print(f"{cls:<20} {mask.sum():>6} {c_ious.mean():>10.4f} {a25:>8.4f} {a50:>8.4f}")
        results[cls] = {
            "count": int(mask.sum()),
            "mean_iou": c_ious.mean().item(),
            "acc_0.25": a25,
            "acc_0.5": a50,
        }

    # Visualize: sort by IoU, show best and worst
    sorted_idx = ious.argsort(descending=True)
    n_half = args.n_show // 2
    viz_indices = list(sorted_idx[:n_half]) + list(sorted_idx[-n_half:])
    viz_kinds = ["best"] * n_half + ["worst"] * n_half

    for idx, kind in zip(viz_indices, viz_kinds):
        info = test_meta[idx]
        img_path = info["path"]
        cls_name = info["class"]
        true_bbox = bboxes[idx].cpu().tolist()
        pred_b = pred_clamped[idx].cpu().tolist()
        iou_val = ious[idx].cpu().item()

        img = Image.open(img_path).convert("RGB")
        fig, ax = plt.subplots(1, figsize=(6, 6))
        ax.imshow(img, cmap="gray")

        gt_rect = patches.Rectangle(
            (true_bbox[0], true_bbox[1]), true_bbox[2], true_bbox[3],
            linewidth=2, edgecolor="green", facecolor="none", label="GT"
        )
        ax.add_patch(gt_rect)

        pr_rect = patches.Rectangle(
            (pred_b[0], pred_b[1]), pred_b[2], pred_b[3],
            linewidth=2, edgecolor="red", facecolor="none",
            label=f"Pred IoU={iou_val:.2f}"
        )
        ax.add_patch(pr_rect)

        prompt = class_prompts.get(cls_name, "")
        ax.text(10, 15, f"GT: {cls_name}", color="green", fontsize=10, fontweight="bold")
        ax.text(10, 30, f"Pred IoU={iou_val:.2f}", color="red", fontsize=10, fontweight="bold")
        if prompt:
            ax.text(10, 45, f'Prompt: "{prompt}"', color="white", fontsize=8)
        ax.legend(loc="upper right")
        ax.axis("off")
        plt.tight_layout(pad=0)

        save_name = f"{kind}_{cls_name}_iou{iou_val:.2f}.png"
        save_path = os.path.join(viz_dir, save_name)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved {save_path}")

    # Save all results
    results["mean_iou"] = ious.mean().item()
    results["iou_at_0.25"] = (ious >= 0.25).float().mean().item()
    results["iou_at_0.5"] = (ious >= 0.5).float().mean().item()
    results["per_sample"] = [
        {
            "class": test_meta[i]["class"],
            "iou": ious[i].item(),
            "bbox_gt": bboxes[i].cpu().tolist(),
            "bbox_pred": pred_clamped[i].cpu().tolist(),
        }
        for i in range(len(bboxes))
    ]

    eval_file = os.path.join(args.output_dir, "bbox_eval.json")
    with open(eval_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {eval_file}")
    print(f"Visualizations saved to {viz_dir}/")


if __name__ == "__main__":
    main()
