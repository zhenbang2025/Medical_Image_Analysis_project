"""
Train a CNN/ViT baseline on chest X-ray images.
"""

import argparse
import json
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms

from data_utils import ImageDataset, load_split_file, list_split_samples
from metrics_utils import compute_metrics


def build_model(name: str) -> nn.Module:
    if name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, 1)
        return model
    if name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Linear(model.fc.in_features, 1)
        return model
    if name == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        model.classifier = nn.Linear(model.classifier.in_features, 1)
        return model

    raise ValueError(f"Unsupported model: {name}")


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.float().to(device)

        optimizer.zero_grad()
        logits = model(images).squeeze(-1)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_probs = []
    all_labels = []

    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.float().to(device)

        logits = model(images).squeeze(-1)
        loss = criterion(logits, labels)
        probs = torch.sigmoid(logits).detach().cpu().tolist()

        total_loss += loss.item() * images.size(0)
        all_probs.extend(probs)
        all_labels.extend(labels.long().cpu().tolist())

    metrics = compute_metrics(all_labels, all_probs)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="./data/chest_xray")
    parser.add_argument("--split_file", type=str, default="")
    parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "resnet50", "densenet121"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--pos_weight", type=float, default=0.0)
    parser.add_argument("--output_dir", type=str, default="./output")
    args = parser.parse_args()

    if args.split_file:
        splits = load_split_file(args.split_file)
        train_samples = splits["train"]
        val_samples = splits["val"]
        test_samples = splits["test"]
    else:
        train_samples = list_split_samples(args.dataset_dir, "train")
        val_samples = list_split_samples(args.dataset_dir, "val")
        test_samples = list_split_samples(args.dataset_dir, "test")

    train_tf = transforms.Compose(
        [
            transforms.Resize((448, 448)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(5),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    train_ds = ImageDataset(train_samples, transform=train_tf)
    val_ds = ImageDataset(val_samples, transform=eval_tf)
    test_ds = ImageDataset(test_samples, transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = build_model(args.model).to(device)

    if args.pos_weight > 0:
        pos_weight = torch.tensor([args.pos_weight], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        if val_metrics["accuracy"] > best_val:
            best_val = val_metrics["accuracy"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        print(
            f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} "
            f"Val Loss: {val_metrics['loss']:.4f} Val Acc: {val_metrics['accuracy']:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate(model, test_loader, criterion, device)

    os.makedirs(args.output_dir, exist_ok=True)
    result = {
        "model": args.model,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "pos_weight": args.pos_weight,
        "test": test_metrics,
    }

    output_name = f"cnn_{args.model}.json"
    with open(os.path.join(args.output_dir, output_name), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    model_path = os.path.join(args.output_dir, f"cnn_{args.model}.pt")
    torch.save({"model_state": model.state_dict(), "config": result}, model_path)

    print("Test metrics:", test_metrics)


if __name__ == "__main__":
    main()
