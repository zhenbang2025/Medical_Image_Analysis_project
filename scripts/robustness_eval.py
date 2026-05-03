"""
Evaluate robustness of CNN baseline under input perturbations.
"""

import argparse
import json
import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F

from data_utils import ImageDataset, load_split_file, list_split_samples
from metrics_utils import compute_metrics
from train_cnn import build_model


class AddGaussianNoise:
    def __init__(self, std: float):
        self.std = std

    def __call__(self, tensor):
        if self.std <= 0:
            return tensor
        noise = torch.randn_like(tensor) * self.std
        return torch.clamp(tensor + noise, 0.0, 1.0)


def evaluate(model, loader, device):
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            logits = model(images).squeeze(-1)
            probs = torch.sigmoid(logits).detach().cpu().tolist()
            all_probs.extend(probs)
            all_labels.extend(labels)

    return compute_metrics(all_labels, all_probs)


def build_transform(corruption: str, level: float):
    base = [transforms.Resize((448, 448))]

    if corruption == "brightness":
        base.append(transforms.Lambda(lambda img: F.adjust_brightness(img, 1.0 + level)))
    elif corruption == "contrast":
        base.append(transforms.Lambda(lambda img: F.adjust_contrast(img, 1.0 + level)))
    elif corruption == "rotation":
        base.append(transforms.Lambda(lambda img: F.rotate(img, level)))
    elif corruption == "blur":
        base.append(transforms.GaussianBlur(kernel_size=3, sigma=level))

    base.append(transforms.ToTensor())

    if corruption == "noise":
        base.append(AddGaussianNoise(level))

    base.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    return transforms.Compose(base)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="./data/chest_xray")
    parser.add_argument("--split_file", type=str, default="")
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--model_path", type=str, default="./output/cnn_resnet18.pt")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default="./output")
    args = parser.parse_args()

    if args.split_file:
        splits = load_split_file(args.split_file)
        test_samples = splits["test"]
    else:
        test_samples = list_split_samples(args.dataset_dir, "test")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = build_model(args.model)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state["model_state"])
    model = model.to(device)

    corruptions = {
        "brightness": [0.0, 0.1, 0.2],
        "contrast": [0.0, 0.1, 0.2],
        "rotation": [0.0, 3.0, 5.0],
        "blur": [0.0, 0.5, 1.0],
        "noise": [0.0, 0.02, 0.05],
    }

    results = {}
    for corruption, levels in corruptions.items():
        results[corruption] = {}
        for level in levels:
            tf = build_transform(corruption, level)
            dataset = ImageDataset(test_samples, transform=tf)
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

            metrics = evaluate(model, loader, device)
            results[corruption][str(level)] = metrics
            print(f"{corruption} level={level}: acc={metrics['accuracy']:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, "robustness_results.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("Saved robustness results:", save_path)


if __name__ == "__main__":
    main()
