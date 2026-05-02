"""
Train an MLP classifier on pre-extracted Qwen2.5-VL image embeddings.

Usage:
    python train_classifier.py [--epochs 30] [--lr 1e-3] [--batch_size 64]

Notes:
    - Extremely lightweight — runs on any Mac (no GPU needed)
    - Only trains the MLP, Qwen model is not loaded
    - Embeddings must be pre-extracted by extract_embeddings.py
"""

import argparse
import json
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout=0.3):
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
    total_loss = 0
    correct = 0
    total = 0

    for embeddings, labels in loader:
        embeddings, labels = embeddings.to(device), labels.float().to(device)

        optimizer.zero_grad()
        logits = model(embeddings)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * embeddings.size(0)
        preds = (logits > 0).long()
        correct += (preds == labels.long()).sum().item()
        total += embeddings.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for embeddings, labels in loader:
        embeddings, labels = embeddings.to(device), labels.float().to(device)

        logits = model(embeddings)
        loss = criterion(logits, labels)

        total_loss += loss.item() * embeddings.size(0)
        preds = (logits > 0).long()
        correct += (preds == labels.long()).sum().item()
        total += embeddings.size(0)

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embedding_dir",
        type=str,
        default="./embeddings",
        help="Directory containing train.pt, val.pt, test.pt",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Directory to save the trained model",
    )
    args = parser.parse_args()

    # Load embeddings
    train_data = torch.load(os.path.join(args.embedding_dir, "train.pt"), weights_only=True)
    val_data = torch.load(os.path.join(args.embedding_dir, "val.pt"), weights_only=True)
    test_data = torch.load(os.path.join(args.embedding_dir, "test.pt"), weights_only=True)

    train_dataset = EmbeddingDataset(train_data["embeddings"], train_data["labels"])
    val_dataset = EmbeddingDataset(val_data["embeddings"], val_data["labels"])
    test_dataset = EmbeddingDataset(test_data["embeddings"], test_data["labels"])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    input_dim = train_data["embeddings"].shape[1]

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Model, criterion, optimizer
    model = MLPClassifier(input_dim, args.hidden_dim, args.dropout).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"Input dim: {input_dim}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {device}")
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    print()

    best_val_acc = 0
    best_model_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Evaluate on test set
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nBest Val Accuracy: {best_val_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    # Save model
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, "mlp_classifier.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "test_accuracy": test_acc,
        },
        save_path,
    )
    print(f"Model saved to {save_path}")

    # Save results as JSON
    results = {
        "mlp_test_accuracy": test_acc,
        "mlp_best_val_accuracy": best_val_acc,
        "epochs": args.epochs,
        "lr": args.lr,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
    }
    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Try to load zero-shot results for comparison
    zero_shot_path = os.path.join(args.output_dir, "zero_shot_results.json")
    if os.path.exists(zero_shot_path):
        with open(zero_shot_path, "r") as f:
            zero_shot = json.load(f)

        mlp_acc = results["mlp_test_accuracy"]
        zs_acc = zero_shot.get("test_accuracy", 0)

        print("\n" + "=" * 60)
        print("COMPARISON: Zero-shot vs MLP (Test Set)")
        print("=" * 60)
        print(f"  Zero-shot (Qwen2.5-VL only):  {zs_acc:.4f}")
        print(f"  Qwen embeddings + MLP:        {mlp_acc:.4f}")
        improvement = mlp_acc - zs_acc
        print(f"  Improvement:                  {improvement:+.4f}")
        if improvement > 0:
            print(f"  MLP correctly classifies {improvement * 100:.1f}% more test samples.")
        else:
            print(f"  MLP does not outperform zero-shot.")
        print("=" * 60)
    else:
        print(f"\nTip: Run evaluate_zero_shot.py and save to {zero_shot_path}")
        print("     to see a comparison table automatically.")


if __name__ == "__main__":
    main()
