"""
Zero-shot classification using Qwen2-VL directly (no MLP).

Asks the model to classify each image as NORMAL or PNEUMONIA,
then computes accuracy for comparison with the trained MLP.

Usage:
    python evaluate_zero_shot.py [--model Qwen/Qwen2.5-VL-2B-Instruct] [--batch_size 4] [--split test]
"""

import argparse
import json
import re
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


PROMPT_ZERO_SHOT = (
    "Look at this chest X-ray image. "
    "Is it NORMAL or does it show PNEUMONIA? "
    "Answer with only one word: NORMAL or PNEUMONIA."
)


class ChestXRayDataset(Dataset):
    def __init__(self, root_dir, split="test"):
        self.split = split
        self.root_dir = Path(root_dir) / split
        self.samples = []
        self.labels = []

        for label_idx, class_name in enumerate(["NORMAL", "PNEUMONIA"]):
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                class_dir = self.root_dir / "chest_xray" / class_name
            for img_path in sorted(class_dir.glob("*.jpeg")):
                self.samples.append(str(img_path))
                self.labels.append(label_idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert("RGB")
        return img, self.samples[idx], self.labels[idx]


def collate_fn(batch):
    images = [item[0] for item in batch]
    paths = [item[1] for item in batch]
    labels = [item[2] for item in batch]
    return images, paths, labels


def classify_batch(model, processor, images, device):
    """Ask Qwen to classify images and parse responses."""
    texts = [PROMPT_ZERO_SHOT] * len(images)
    messages = []
    for text in texts:
        messages.append([
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": text},
                ],
            }
        ])

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
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=8,
            temperature=0.1,
        )
        # Extract only the newly generated tokens
        input_len = inputs.input_ids.shape[1]
        generated_tokens = generated_ids[:, input_len:]

    responses = processor.batch_decode(
        generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return responses


def parse_response(response):
    """Extract NORMAL or PNEUMONIA from the model's response."""
    text = response.strip().upper()
    if "PNEUMONIA" in text:
        return 1
    if "NORMAL" in text:
        return 0
    return None  # Could not parse


def evaluate_split(model, processor, loader, device, split_name):
    """Evaluate zero-shot accuracy on a dataset split."""
    all_preds = []
    all_labels = []
    all_responses = []
    all_paths = []
    parsed_count = 0

    print(f"\n[{split_name}] Running zero-shot classification...")
    for images, paths, labels in tqdm(loader, desc=split_name):
        responses = classify_batch(model, processor, images, device)

        for resp, path, label in zip(responses, paths, labels):
            pred = parse_response(resp)
            all_responses.append(resp.strip())
            all_paths.append(path)
            if pred is not None:
                all_preds.append(pred)
                all_labels.append(label)
                parsed_count += 1
            else:
                # Count unparsable as wrong
                all_preds.append(-1)
                all_labels.append(label)

    # Compute accuracy only on successfully parsed responses
    valid_pairs = [
        (p, l) for p, l in zip(all_preds, all_labels) if p != -1
    ]
    correct = sum(1 for p, l in valid_pairs if p == l)
    total_valid = len(valid_pairs)
    total_all = len(all_labels)
    unparsed = total_all - total_valid

    accuracy = correct / total_valid if total_valid > 0 else 0

    print(f"  Total images: {total_all}")
    print(f"  Parsed successfully: {total_valid}/{total_all}")
    if unparsed > 0:
        print(f"  Unparsed responses: {unparsed}")
    print(f"  Correct: {correct}/{total_valid}")
    print(f"  Accuracy: {accuracy:.4f}")

    # Show some example responses for debugging
    print(f"  Sample responses (first 10):")
    for path, resp, label, pred in zip(all_paths, all_responses, all_labels, all_preds):
        true_label = "PNEUMONIA" if label == 1 else "NORMAL"
        pred_label = "PNEUMONIA" if pred == 1 else ("NORMAL" if pred == 0 else "UNPARSED")
        print(f"    {Path(path).name:40s} True: {true_label:12s} Pred: {pred_label:12s} Response: {repr(resp[:50])}")
        if len([r for r in all_responses if r.strip()]) > 10:
            break

    return accuracy, correct, total_valid, unparsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=""
        "data/chest_xray_small",
        help="Path to the chest_xray subdirectory",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/Qwen2-VL-2B-Instruct",
        help="Path to local model directory or HuggingFace model name",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "mps", "cuda", "cpu"],
        help="Device to use: auto (prefer mps/cuda), or force cpu",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test", "all"],
        help="Which split to evaluate, or 'all' for every split",
    )
    args = parser.parse_args()

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
            args.model,
            torch_dtype=dtype,
            device_map="cpu",
        )
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.model,
            torch_dtype=dtype,
            device_map="auto",
        )
    processor = AutoProcessor.from_pretrained(args.model)

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    results = {}

    for split in splits:
        dataset = ChestXRayDataset(args.dataset_dir, split=split)
        if len(dataset) == 0:
            print(f"  No images found for split '{split}' in {args.dataset_dir}, skipping.")
            continue

        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
        )

        acc, correct, total_valid, unparsed = evaluate_split(
            model, processor, loader, device, split
        )
        results[split] = {
            "accuracy": acc,
            "correct": correct,
            "total_valid": total_valid,
            "unparsed": unparsed,
            "total": len(dataset),
        }

    print("\n" + "=" * 60)
    print("ZERO-SHOT QWEN2.5-VL CLASSIFICATION RESULTS")
    print("=" * 60)
    for split, r in results.items():
        print(
            f"  {split:6s} | Accuracy: {r['accuracy']:.4f} "
            f"| {r['correct']}/{r['total_valid']} correct "
            f"| {r['unparsed']} unparsed"
        )
    print("=" * 60)

    # Save results as JSON for comparison with MLP
    if "test" in results:
        save_results = {
            "test_accuracy": results["test"]["accuracy"],
            "test_correct": results["test"]["correct"],
            "test_total_valid": results["test"]["total_valid"],
            "model": args.model,
        }
        os.makedirs("./output", exist_ok=True)
        save_path = "./output/zero_shot_results.json"
        with open(save_path, "w") as f:
            json.dump(save_results, f, indent=2)
        print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()
