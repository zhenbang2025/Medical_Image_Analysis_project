"""
Evaluate trained MLP checkpoint on test split and optionally export visualizations.
"""

import argparse
import os
from pathlib import Path

import torch

matplotlib = None
plt = None
patches = None
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
except ImportError:
    pass

from PIL import Image

from utils.bbox_utils import denormalize_xywh, iou_xywh
from utils.env_utils import env_default, env_int
from utils.models import BBoxMLP
from utils.run_utils import load_json, prepare_run_dir, resolve_device, save_json


def per_class_metrics(ious: torch.Tensor, labels: torch.Tensor, classes: list[str]) -> dict:
    out = {}
    for i, cls in enumerate(classes):
        m = labels == i
        if m.sum() == 0:
            continue
        c = ious[m]
        out[cls] = {
            "count": int(m.sum()),
            "mean_iou": float(c.mean()),
            "iou_at_0.25": float((c >= 0.25).float().mean()),
            "iou_at_0.5": float((c >= 0.5).float().mean()),
        }
    return out


def draw_examples(pred_norm, true_norm, sizes, labels, meta, classes, out_dir, n_show=10):
    if plt is None or n_show <= 0:
        return
    os.makedirs(out_dir, exist_ok=True)
    ious = iou_xywh(pred_norm, true_norm)
    sort_idx = torch.argsort(ious)
    half = max(1, n_show // 2)
    selected = list(sort_idx[:half]) + list(sort_idx[-half:])

    for idx in selected:
        idx = int(idx)
        sample = meta[idx]
        img = Image.open(sample["path"]).convert("RGB")
        w, h = img.size
        size = torch.tensor([[w, h]], dtype=torch.float32)
        gt_abs = denormalize_xywh(true_norm[idx:idx + 1], size).squeeze(0).tolist()
        pr_abs = denormalize_xywh(pred_norm[idx:idx + 1], size).squeeze(0).tolist()
        iou = float(ious[idx])
        cls = classes[int(labels[idx])]

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img)
        ax.add_patch(patches.Rectangle((gt_abs[0], gt_abs[1]), gt_abs[2], gt_abs[3],
                                       edgecolor="lime", facecolor="none", linewidth=2, label="GT"))
        ax.add_patch(patches.Rectangle((pr_abs[0], pr_abs[1]), pr_abs[2], pr_abs[3],
                                       edgecolor="red", facecolor="none", linewidth=2, label=f"Pred {iou:.2f}"))
        ax.set_title(f"{cls} | IoU={iou:.3f}")
        ax.legend()
        ax.axis("off")
        tag = "worst" if idx in set(sort_idx[:half].tolist()) else "best"
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{tag}_{cls}_iou{iou:.2f}.png"), dpi=150)
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    model_default = env_default("MODEL_CKPT")
    parser.add_argument("--model_path", type=str, default=model_default, required=model_default is None)
    parser.add_argument("--embedding_dir", type=str, default=env_default("EMBEDDING_DIR"),
                        help="Optional override. By default reads from checkpoint['embedding_dir'].")
    parser.add_argument("--output_root", type=str, default=env_default("OUTPUT_EVAL_ROOT", "./output/eval"))
    parser.add_argument("--run_name", type=str, default=env_default("RUN_NAME"))
    parser.add_argument("--allow_overwrite", action="store_true")
    parser.add_argument("--n_show", type=int, default=env_int("N_SHOW", 10))
    parser.add_argument("--device", type=str, default=env_default("DEVICE", "auto"), choices=["auto", "cuda", "mps", "cpu"])
    args = parser.parse_args()

    run_dir = prepare_run_dir(args.output_root, args.run_name, prefix="eval_mlp", allow_overwrite=args.allow_overwrite)
    device, _ = resolve_device(args.device)

    ckpt = torch.load(args.model_path, weights_only=True)
    emb_dir = args.embedding_dir or ckpt["embedding_dir"]
    test = torch.load(os.path.join(emb_dir, "test.pt"), weights_only=True)
    meta = load_json(os.path.join(emb_dir, "meta.json"))["test"]
    classes = ckpt["classes"]

    model = BBoxMLP(
        input_dim=ckpt["input_dim"],
        hidden_dim=ckpt["hidden_dim"],
        dropout=ckpt["dropout"],
        arch=ckpt.get("arch", "plain"),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with torch.no_grad():
        x = test["embeddings"].float().to(device)
        pred_norm = torch.sigmoid(model(x)).cpu()
    true_norm = test["bbox_norm_xywh"].float()
    labels = test["labels"]
    sizes = test["image_sizes_wh"].float()

    ious = iou_xywh(pred_norm, true_norm)
    out = {
        "model_path": str(Path(args.model_path).resolve()),
        "embedding_dir": emb_dir,
        "mean_iou": float(ious.mean()),
        "iou_at_0.25": float((ious >= 0.25).float().mean()),
        "iou_at_0.5": float((ious >= 0.5).float().mean()),
        "per_class": per_class_metrics(ious, labels, classes),
        "per_sample": [],
    }

    pred_abs = denormalize_xywh(pred_norm, sizes)
    true_abs = denormalize_xywh(true_norm, sizes)
    for i in range(len(ious)):
        out["per_sample"].append(
            {
                "name": meta[i]["name"],
                "class": meta[i]["class"],
                "iou": float(ious[i]),
                "bbox_pred_norm_xywh": pred_norm[i].tolist(),
                "bbox_gt_norm_xywh": true_norm[i].tolist(),
                "bbox_pred_abs_xywh": pred_abs[i].tolist(),
                "bbox_gt_abs_xywh": true_abs[i].tolist(),
            }
        )

    save_json(os.path.join(run_dir, "bbox_eval.json"), out)
    draw_examples(pred_norm, true_norm, sizes, labels, meta, classes, os.path.join(run_dir, "bbox_viz"), args.n_show)
    print(f"Saved evaluation to: {run_dir}")


if __name__ == "__main__":
    main()
