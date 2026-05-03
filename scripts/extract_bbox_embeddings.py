"""
Extract embeddings for 3 target classes from NIH ChestX-ray14 dataset.

For each class, selects 40 images (30 train + 10 test) with bbox annotations.
Uses a class-specific short prompt so the embedding encodes "what to look for".
Saves all embeddings + bbox + class labels into a single file for unified training.

Usage:
    python extract_bbox_embeddings.py

Target classes:
    - Atelectasis (lung collapse)
    - Effusion (pleural effusion)
    - Cardiomegaly (heart enlargement)
"""

import argparse
import csv
import json
import os
import random
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

DEFAULT_CLASSES = ["Atelectasis", "Effusion", "Cardiomegaly"]
TRAIN_PER_CLASS = 30
TEST_PER_CLASS = 10

CLASS_PROMPTS = {
    "Atelectasis": "Describe lung collapse in this X-ray.",
    "Effusion": "Describe pleural effusion in this X-ray.",
    "Cardiomegaly": "Describe heart enlargement in this X-ray.",
}


def extract_batch_embeddings(model, processor, images, class_name, device):
    """Extract embeddings for a batch of images with class-specific prompt."""
    prompt = CLASS_PROMPTS[class_name]
    messages = []
    for _ in images:
        messages.append([{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }])

    text_inputs = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=text_inputs,
        images=images,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        embeddings = hidden_states.mean(dim=1)

    return embeddings.cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data/3")
    parser.add_argument("--bbox_csv", type=str, default="./data/3/BBox_List_2017.csv")
    parser.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES)
    parser.add_argument("--model", type=str, default="./models/Qwen2-VL-2B-Instruct")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "mps", "cuda", "cpu"])
    parser.add_argument("--output_dir", type=str, default="./embeddings")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Device
    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            dtype = torch.float16
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            dtype = torch.float16
        else:
            device = torch.device("cpu")
            dtype = torch.float32
    elif args.device == "mps":
        device = torch.device("mps")
        dtype = torch.float16
    elif args.device == "cuda":
        device = torch.device("cuda")
        dtype = torch.float16
    else:
        device = torch.device("cpu")
        dtype = torch.float32

    print(f"Device: {device}, dtype: {dtype}")
    print(f"Loading model: {args.model} ...")
    if device.type == "cpu":
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.model, torch_dtype=dtype, device_map="cpu",
        )
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.model, torch_dtype=dtype, device_map="auto",
        )
    processor = AutoProcessor.from_pretrained(args.model)
    print(f"Model loaded.")

    # Parse bbox CSV
    class_samples = {cls: [] for cls in args.classes}
    with open(args.bbox_csv) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            label = row[1]
            if label not in args.classes:
                continue
            x, y, w, h = float(row[2]), float(row[3]), float(row[4]), float(row[5])

            img_path = None
            for d in sorted(Path(args.data_dir).iterdir()):
                if d.is_dir() and d.name.startswith("images_"):
                    path = d / "images" / row[0]
                    if path.exists():
                        img_path = path
                        break
            if img_path is None:
                continue

            class_samples[label].append((str(img_path), row[0], x, y, w, h))

    # Select and split, then extract
    class2idx = {c: i for i, c in enumerate(args.classes)}
    total_per_class = TRAIN_PER_CLASS + TEST_PER_CLASS

    # Accumulators: list of dicts, then split at end
    train_embs, train_bbox, train_labels = [], [], []
    test_embs, test_bbox, test_labels = [], [], []
    meta = {"train": [], "test": []}

    for cls_name in args.classes:
        samples = class_samples[cls_name]
        if len(samples) < total_per_class:
            print(f"Warning: {cls_name} has only {len(samples)} samples, need {total_per_class}")
            selected = samples
        else:
            selected = random.sample(samples, total_per_class)

        random.shuffle(selected)
        train_samples = selected[:TRAIN_PER_CLASS]
        test_samples = selected[TRAIN_PER_CLASS:]

        for split, subsamples in [("train", train_samples), ("test", test_samples)]:
            label_idx = class2idx[cls_name]

            for i in tqdm(range(0, len(subsamples), args.batch_size),
                          desc=f"{cls_name}/{split}", leave=False):
                batch = subsamples[i:i + args.batch_size]
                batch_paths = [s[0] for s in batch]
                batch_images = [Image.open(p).convert("RGB") for p in batch_paths]

                embs = extract_batch_embeddings(model, processor, batch_images, cls_name, device)

                for emb, (path, name, x, y, w, h) in zip(embs, batch):
                    if split == "train":
                        train_embs.append(emb)
                        train_bbox.append(torch.tensor([x, y, w, h], dtype=torch.float32))
                        train_labels.append(label_idx)
                    else:
                        test_embs.append(emb)
                        test_bbox.append(torch.tensor([x, y, w, h], dtype=torch.float32))
                        test_labels.append(label_idx)

                    meta[split].append({
                        "class": cls_name,
                        "path": path,
                        "name": name,
                        "x": x, "y": y, "w": w, "h": h,
                    })

    os.makedirs(args.output_dir, exist_ok=True)

    torch.save({
        "embeddings": torch.stack(train_embs),
        "bboxes": torch.stack(train_bbox),
        "labels": torch.tensor(train_labels, dtype=torch.long),
    }, os.path.join(args.output_dir, "bbox_train.pt"))

    torch.save({
        "embeddings": torch.stack(test_embs),
        "bboxes": torch.stack(test_bbox),
        "labels": torch.tensor(test_labels, dtype=torch.long),
    }, os.path.join(args.output_dir, "bbox_test.pt"))

    with open(os.path.join(args.output_dir, "bbox_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(args.output_dir, "class_prompts.json"), "w") as f:
        json.dump(CLASS_PROMPTS, f, indent=2)
    with open(os.path.join(args.output_dir, "class_names.json"), "w") as f:
        json.dump({"classes": args.classes, "class2idx": class2idx}, f, indent=2)

    print(f"\nDone!")
    print(f"  Train: {len(train_embs)} samples")
    print(f"  Test:  {len(test_embs)} samples")
    print(f"  Classes: {args.classes}")
    print(f"  Embedding dim: {train_embs[0].shape[0]}")
    print(f"  Saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
