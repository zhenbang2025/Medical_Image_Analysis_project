"""
Train MLP bbox regressor on pre-extracted embeddings.

Supports:
- architectures: plain / residual
- losses: smoothl1 / iou / ciou
- normalized bbox targets
- no-overwrite run directory
"""

import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from utils.bbox_utils import ciou_xywh, iou_xywh
from utils.env_utils import env_default, env_float, env_int
from utils.models import BBoxMLP
from utils.run_utils import load_json, prepare_run_dir, resolve_device, save_json


def make_loss(loss_name: str):
    if loss_name == "smoothl1":
        base = nn.SmoothL1Loss()

        def _loss(pred, target):
            return base(pred, target)

        return _loss

    if loss_name == "iou":
        def _loss(pred, target):
            return (1.0 - iou_xywh(pred, target)).mean()

        return _loss

    if loss_name == "ciou":
        def _loss(pred, target):
            return (1.0 - ciou_xywh(pred, target)).mean()

        return _loss

    raise ValueError(f"Unknown loss: {loss_name}")


def evaluate(model, loader, device):
    model.eval()
    all_pred, all_true, all_labels = [], [], []
    with torch.no_grad():
        for x, y, labels in loader:
            x = x.to(device)
            pred = torch.sigmoid(model(x)).cpu()
            all_pred.append(pred)
            all_true.append(y)
            all_labels.append(labels)
    pred = torch.cat(all_pred)
    true = torch.cat(all_true)
    labels = torch.cat(all_labels)
    ious = iou_xywh(pred, true)
    return ious, labels


def per_class_metrics(ious: torch.Tensor, labels: torch.Tensor, classes: list[str]) -> dict:
    out = {}
    for i, cls in enumerate(classes):
        mask = labels == i
        if mask.sum() == 0:
            continue
        c = ious[mask]
        out[cls] = {
            "count": int(mask.sum()),
            "mean_iou": float(c.mean()),
            "iou_at_0.25": float((c >= 0.25).float().mean()),
            "iou_at_0.5": float((c >= 0.5).float().mean()),
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    embedding_default = env_default("EMBEDDING_DIR")
    parser.add_argument(
        "--embedding_dir",
        type=str,
        default=embedding_default,
        required=embedding_default is None,
    )
    parser.add_argument("--output_root", type=str, default=env_default("OUTPUT_TRAIN_ROOT", "./output/train"))
    parser.add_argument("--run_name", type=str, default=env_default("RUN_NAME"))
    parser.add_argument("--allow_overwrite", action="store_true")
    parser.add_argument("--arch", type=str, default=env_default("ARCH", "residual"), choices=["plain", "residual"])
    parser.add_argument("--loss", type=str, default=env_default("LOSS_NAME", "smoothl1"), choices=["smoothl1", "iou", "ciou"])
    parser.add_argument("--epochs", type=int, default=env_int("EPOCHS", 100))
    parser.add_argument("--batch_size", type=int, default=env_int("BATCH_SIZE", 64))
    parser.add_argument("--hidden_dim", type=int, default=env_int("HIDDEN_DIM", 256))
    parser.add_argument("--dropout", type=float, default=env_float("DROPOUT", 0.2))
    parser.add_argument("--lr", type=float, default=env_float("LR", 1e-3))
    parser.add_argument("--weight_decay", type=float, default=env_float("WEIGHT_DECAY", 1e-4))
    parser.add_argument("--seed", type=int, default=env_int("SEED", 42))
    parser.add_argument("--device", type=str, default=env_default("DEVICE", "auto"), choices=["auto", "cuda", "mps", "cpu"])
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device, _ = resolve_device(args.device)

    run_dir = prepare_run_dir(
        args.output_root,
        args.run_name,
        prefix=f"train_{args.arch}_{args.loss}",
        allow_overwrite=args.allow_overwrite,
    )
    print(f"Run dir: {run_dir}")

    train = torch.load(os.path.join(args.embedding_dir, "train.pt"), weights_only=True)
    val = torch.load(os.path.join(args.embedding_dir, "val.pt"), weights_only=True)
    test = torch.load(os.path.join(args.embedding_dir, "test.pt"), weights_only=True)
    class_info = load_json(os.path.join(args.embedding_dir, "class_names.json"))
    classes = class_info["classes"]

    train_ds = TensorDataset(train["embeddings"].float(), train["bbox_norm_xywh"].float(), train["labels"])
    val_ds = TensorDataset(val["embeddings"].float(), val["bbox_norm_xywh"].float(), val["labels"])
    test_ds = TensorDataset(test["embeddings"].float(), test["bbox_norm_xywh"].float(), test["labels"])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    in_dim = train["embeddings"].shape[1]
    model = BBoxMLP(in_dim, hidden_dim=args.hidden_dim, dropout=args.dropout, arch=args.arch).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = make_loss(args.loss)

    best_iou = -1.0
    best_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n = 0
        for x, y, _ in tqdm(train_loader, desc=f"Epoch {epoch:03d}", leave=False):
            x = x.to(device)
            y = y.to(device)
            pred = torch.sigmoid(model(x))
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x.size(0)
            n += x.size(0)
        scheduler.step()

        val_ious, val_labels = evaluate(model, val_loader, device)
        val_mean_iou = float(val_ious.mean())
        train_loss = epoch_loss / max(1, n)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_mean_iou": val_mean_iou})

        if val_mean_iou > best_iou:
            best_iou = val_mean_iou
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch <= 10 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:03d} | train_loss={train_loss:.5f} | val_mean_iou={val_mean_iou:.5f}")

    model.load_state_dict(best_state)
    val_ious, val_labels = evaluate(model, val_loader, device)
    test_ious, test_labels = evaluate(model, test_loader, device)

    ckpt = {
        "model_state_dict": model.state_dict(),
        "input_dim": in_dim,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "arch": args.arch,
        "loss": args.loss,
        "classes": classes,
        "best_val_mean_iou": float(val_ious.mean()),
        "embedding_dir": args.embedding_dir,
    }
    torch.save(ckpt, os.path.join(run_dir, "bbox_mlp.pt"))

    result = {
        "arch": args.arch,
        "loss": args.loss,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "embedding_dir": args.embedding_dir,
        "best_val_mean_iou": float(val_ious.mean()),
        "best_val_iou_at_0.25": float((val_ious >= 0.25).float().mean()),
        "best_val_iou_at_0.5": float((val_ious >= 0.5).float().mean()),
        "test_mean_iou": float(test_ious.mean()),
        "test_iou_at_0.25": float((test_ious >= 0.25).float().mean()),
        "test_iou_at_0.5": float((test_ious >= 0.5).float().mean()),
        "val_per_class": per_class_metrics(val_ious, val_labels, classes),
        "test_per_class": per_class_metrics(test_ious, test_labels, classes),
        "history": history,
    }
    save_json(os.path.join(run_dir, "train_result.json"), result)
    print(f"Saved model/result to: {run_dir}")


if __name__ == "__main__":
    main()
