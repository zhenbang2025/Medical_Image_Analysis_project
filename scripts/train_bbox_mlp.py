"""
Train a single MLP for bbox regression on all classes.

The class-specific prompt is baked into the embedding during extraction,
so the MLP only needs to regress bbox coordinates (x, y, w, h).

Usage:
    python train_bbox_mlp.py [--epochs 100] [--lr 1e-3]

Data:
    - embeddings/bbox_train.pt: 90 samples (30 per class)
    - embeddings/bbox_test.pt:  30 samples (10 per class)
"""

import argparse
import json
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

matplotlib = None
plt = None
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    pass


class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, bboxes, labels):
        self.embeddings = embeddings
        self.bboxes = bboxes
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.bboxes[idx], self.labels[idx]


class BBoxMLP(nn.Module):
    """MLP with residual connections and LayerNorm for bbox regression."""

    def __init__(self, input_dim, hidden_dim=256, dropout=0.2, use_residual=False):
        super().__init__()
        self.use_residual = use_residual

        # Project input to hidden_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Residual block
        self.res_block = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )

        # Skip connection projection (hidden_dim -> hidden_dim)
        self.skip_proj = nn.Linear(hidden_dim, hidden_dim)

        # Output head
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 4),
        )

    def forward(self, x):
        h = self.input_proj(x)

        if self.use_residual:
            h = self.res_block(h) + self.skip_proj(h)
        else:
            h = self.res_block(h)

        return self.output(h)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    total = 0

    for embeddings, bboxes, _ in loader:
        embeddings = embeddings.to(device)
        bboxes = bboxes.to(device)

        optimizer.zero_grad()
        pred = model(embeddings)
        loss = criterion(pred, bboxes)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * embeddings.size(0)
        total += embeddings.size(0)

    return total_loss / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total = 0
    all_pred = []
    all_true = []
    all_labels = []

    for embeddings, bboxes, labels in loader:
        embeddings = embeddings.to(device)
        bboxes = bboxes.to(device)

        pred = model(embeddings)
        loss = criterion(pred, bboxes)

        total_loss += loss.item() * embeddings.size(0)
        total += embeddings.size(0)
        all_pred.append(pred.cpu())
        all_true.append(bboxes.cpu())
        all_labels.append(labels)

    all_pred = torch.cat(all_pred)
    all_true = torch.cat(all_true)
    all_labels = torch.cat(all_labels)
    iou = compute_iou(all_pred, all_true)

    return total_loss / total, iou, all_labels


def compute_iou(pred, true):
    pred = pred.clamp(min=0)
    true = true.clamp(min=0)
    pred_x1 = pred[:, 0]
    pred_y1 = pred[:, 1]
    pred_x2 = pred[:, 0] + pred[:, 2]
    pred_y2 = pred[:, 1] + pred[:, 3]
    true_x1 = true[:, 0]
    true_y1 = true[:, 1]
    true_x2 = true[:, 0] + true[:, 2]
    true_y2 = true[:, 1] + true[:, 3]
    inter_x1 = torch.max(pred_x1, true_x1)
    inter_y1 = torch.max(pred_y1, true_y1)
    inter_x2 = torch.min(pred_x2, true_x2)
    inter_y2 = torch.min(pred_y2, true_y2)
    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h
    pred_area = pred[:, 2] * pred[:, 3]
    true_area = true[:, 2] * true[:, 3]
    union_area = pred_area + true_area - inter_area
    return inter_area / (union_area + 1e-6)


def plot_history(history, output_dir):
    if plt is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = range(1, len(history["train_loss"]) + 1)
    axes[0].plot(epochs, history["train_loss"], "b-", label="Train")
    axes[0].plot(epochs, history["val_loss"], "r-", label="Val")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(epochs, history["val_iou"], "g-", label="Val IoU")
    axes[1].set_title("IoU")
    axes[1].legend()
    plt.tight_layout()
    save_path = os.path.join(output_dir, "bbox_training_curves.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Curves saved to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_dir", type=str, default="./embeddings")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_residual", action="store_true",
                        help="Use residual connections with LayerNorm")
    parser.add_argument("--output_dir", type=str, default="./output")
    args = parser.parse_args()

    model_tag = "residual" if args.use_residual else "plain"
    args.output_dir = os.path.join(args.output_dir, model_tag)

    torch.manual_seed(args.seed)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load class names
    with open(os.path.join(args.embedding_dir, "class_names.json")) as f:
        class_info = json.load(f)
    classes = class_info["classes"]

    # Load data
    train_data = torch.load(os.path.join(args.embedding_dir, "bbox_train.pt"), weights_only=True)
    test_data = torch.load(os.path.join(args.embedding_dir, "bbox_test.pt"), weights_only=True)

    # Cast to float32 for MPS compatibility
    train_data["embeddings"] = train_data["embeddings"].float()
    test_data["embeddings"] = test_data["embeddings"].float()

    train_ds = EmbeddingDataset(train_data["embeddings"], train_data["bboxes"], train_data["labels"])
    test_ds = EmbeddingDataset(test_data["embeddings"], test_data["bboxes"], test_data["labels"])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    input_dim = train_data["embeddings"].shape[1]

    model = BBoxMLP(input_dim, args.hidden_dim, args.dropout, args.use_residual).to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"Input dim: {input_dim}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Residual: {args.use_residual}")
    print(f"Device: {device}")
    print(f"Train: {len(train_ds)}, Test: {len(test_ds)}")
    print(f"Classes: {classes}")
    print()

    best_val_iou = -1
    best_model_state = None
    history = {"train_loss": [], "val_loss": [], "val_iou": [], "val_iou_per_class": []}

    for epoch in range(1, args.epochs + 1):
        t_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_ious, v_labels = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        v_mean_iou = v_ious.mean().item()

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["val_iou"].append(v_mean_iou)

        # Per-class IoU
        per_class = {}
        for i, cls in enumerate(classes):
            mask = v_labels == i
            if mask.sum() > 0:
                per_class[cls] = v_ious[mask].mean().item()
        history["val_iou_per_class"].append(per_class)

        if epoch <= 10 or epoch % 10 == 0 or epoch == args.epochs:
            detail = " ".join(f"{k}:{v:.3f}" for k, v in per_class.items())
            print(f"Epoch {epoch:03d} | Train: {t_loss:.4f} | Val Loss: {v_loss:.4f} IoU: {v_mean_iou:.4f} | {detail}")

        if v_mean_iou > best_val_iou:
            best_val_iou = v_mean_iou
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    _, final_ious, _ = evaluate(model, test_loader, criterion, device)
    final_mean = final_ious.mean().item()

    print(f"\nBest Val IoU: {best_val_iou:.4f}")
    print(f"Final Test IoU: {final_mean:.4f}")

    # Save model
    save_path = os.path.join(args.output_dir, "bbox_mlp.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": input_dim,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "use_residual": args.use_residual,
        "classes": classes,
        "best_val_iou": best_val_iou,
    }, save_path)

    # Save results
    results = {
        "best_val_iou": best_val_iou,
        "test_mean_iou": final_mean,
        "model_tag": model_tag,
        "use_residual": args.use_residual,
        "epochs": args.epochs,
        "lr": args.lr,
        "hidden_dim": args.hidden_dim,
    }
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    plot_history(history, args.output_dir)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
