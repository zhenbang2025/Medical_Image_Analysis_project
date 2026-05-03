"""
Train a linear probe or MLP on pre-extracted embeddings.
"""

import argparse
import json
import os
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data_utils import set_seed, subsample_by_fraction
from metrics_utils import compute_metrics


class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


class LinearProbe(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


class MLPProbe(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for embeddings, labels in loader:
        embeddings = embeddings.to(device)
        labels = labels.float().to(device)

        optimizer.zero_grad()
        logits = model(embeddings)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * embeddings.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_model(model, loader, criterion, device) -> Dict:
    model.eval()
    total_loss = 0.0
    all_probs = []
    all_labels = []

    for embeddings, labels in loader:
        embeddings = embeddings.to(device)
        labels = labels.float().to(device)

        logits = model(embeddings)
        loss = criterion(logits, labels)
        probs = torch.sigmoid(logits).detach().cpu().tolist()

        total_loss += loss.item() * embeddings.size(0)
        all_probs.extend(probs)
        all_labels.extend(labels.long().cpu().tolist())

    metrics = compute_metrics(all_labels, all_probs)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_dir", type=str, default="./embeddings")
    parser.add_argument("--model_type", type=str, default="mlp", choices=["linear", "mlp"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--pos_weight", type=float, default=0.0)
    parser.add_argument("--train_fraction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument(
        "--run_name",
        type=str,
        default="",
        help="Optional tag to prefix output files (e.g., pooling_all_tokens)",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    train_data = torch.load(os.path.join(args.embedding_dir, "train.pt"), weights_only=True)
    val_data = torch.load(os.path.join(args.embedding_dir, "val.pt"), weights_only=True)
    test_data = torch.load(os.path.join(args.embedding_dir, "test.pt"), weights_only=True)

    train_embeddings = train_data["embeddings"]
    train_labels = train_data["labels"]

    if args.train_fraction < 1.0:
        samples = [
            (i, int(train_labels[i].item())) for i in range(len(train_labels))
        ]
        subsampled = subsample_by_fraction(samples, args.train_fraction, args.seed)
        if len(subsampled) == 0:
            raise ValueError(
                "train_fraction produced an empty subset; increase the fraction or check labels"
            )
        idx = [i for i, _ in subsampled]
        train_embeddings = train_embeddings[idx]
        train_labels = train_labels[idx]

    train_dataset = EmbeddingDataset(train_embeddings, train_labels)
    val_dataset = EmbeddingDataset(val_data["embeddings"], val_data["labels"])
    test_dataset = EmbeddingDataset(test_data["embeddings"], test_data["labels"])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    input_dim = train_embeddings.shape[1]

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.model_type == "linear":
        model = LinearProbe(input_dim)
    else:
        model = MLPProbe(input_dim, args.hidden_dim, args.dropout)

    model = model.to(device)

    if args.pos_weight > 0:
        pos_weight = torch.tensor([args.pos_weight], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = -1.0
    best_state = None

    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = eval_model(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])

        if val_metrics["accuracy"] > best_val:
            best_val = val_metrics["accuracy"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        print(
            f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} "
            f"Val Loss: {val_metrics['loss']:.4f} Val Acc: {val_metrics['accuracy']:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = eval_model(model, test_loader, criterion, device)

    os.makedirs(args.output_dir, exist_ok=True)
    result = {
        "model_type": args.model_type,
        "input_dim": input_dim,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "pos_weight": args.pos_weight,
        "train_fraction": args.train_fraction,
        "seed": args.seed,
        "test": test_metrics,
    }

    prefix = f"{args.run_name}_" if args.run_name else ""
    output_name = (
        f"{prefix}probe_{args.model_type}_frac{args.train_fraction}_seed{args.seed}.json"
    )
    with open(os.path.join(args.output_dir, output_name), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    model_path = os.path.join(
        args.output_dir,
        f"{prefix}probe_{args.model_type}_frac{args.train_fraction}_seed{args.seed}.pt",
    )
    torch.save({"model_state": model.state_dict(), "config": result}, model_path)

    print("Test metrics:", test_metrics)


if __name__ == "__main__":
    main()
