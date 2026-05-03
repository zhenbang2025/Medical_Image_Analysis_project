"""
Zero-shot classification using Qwen2-VL directly (no MLP).

Asks the model to classify each image as NORMAL or PNEUMONIA,
then computes accuracy for comparison with the trained MLP.

Usage:
    python evaluate_zero_shot.py [--model Qwen/Qwen2.5-VL-2B-Instruct] [--batch_size 4] [--split test]
"""

import argparse
import json
import os
from typing import List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

from data_utils import ImageDataset, load_split_file, list_split_samples
from metrics_utils import compute_metrics


PROMPT_ZERO_SHOT = (
    "Look at this chest X-ray image. "
    "Is it NORMAL or does it show PNEUMONIA? "
    "Answer with only one word: NORMAL or PNEUMONIA."
)


def collate_fn(batch):
    images = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    paths = [item[2] for item in batch]
    return images, paths, labels


def classify_batch_generate(model, processor, images, device):
    """Ask Qwen to classify images and parse responses (generation mode)."""
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


def parse_response(response: str):
    """Extract NORMAL or PNEUMONIA from the model's response."""
    text = response.strip().upper()
    if "PNEUMONIA" in text:
        return 1
    if "NORMAL" in text:
        return 0
    return None  # Could not parse


def score_candidates(
    model,
    processor,
    images,
    candidates: List[str],
    device,
    length_norm: str,
):
    """Score candidate answers by log-likelihood."""
    messages = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": PROMPT_ZERO_SHOT},
                ],
            }
        ]
        for _ in images
    ]

    prompt_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt_inputs = processor(
        text=prompt_text, images=images, padding=True, return_tensors="pt"
    ).to(device)

    prompt_len = prompt_inputs.input_ids.shape[1]

    scores = []
    for candidate in candidates:
        candidate_ids = processor.tokenizer(
            candidate, add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(device)
        candidate_len = candidate_ids.shape[1]

        expanded_candidate = candidate_ids.expand(prompt_inputs.input_ids.size(0), -1)
        input_ids = torch.cat([prompt_inputs.input_ids, expanded_candidate], dim=1)
        attention_mask = torch.cat(
            [
                prompt_inputs.attention_mask,
                torch.ones_like(expanded_candidate, device=device),
            ],
            dim=1,
        )

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        start = prompt_len - 1
        end = start + candidate_len
        candidate_logits = logits[:, start:end, :]
        log_probs = torch.log_softmax(candidate_logits, dim=-1)
        token_log_probs = log_probs.gather(
            -1, expanded_candidate.unsqueeze(-1)
        ).squeeze(-1)

        if length_norm == "avg":
            score = token_log_probs.mean(dim=1)
        else:
            score = token_log_probs.sum(dim=1)

        scores.append(score)

    stacked = torch.stack(scores, dim=1)
    probs = torch.softmax(stacked, dim=1)
    return probs


def evaluate_split(model, processor, loader, device, split_name, scoring, length_norm):
    """Evaluate zero-shot performance on a dataset split."""
    all_probs = []
    all_labels = []
    all_responses = []
    all_paths = []

    print(f"\n[{split_name}] Running zero-shot classification...")
    for images, paths, labels in tqdm(loader, desc=split_name):
        if scoring == "generate":
            responses = classify_batch_generate(model, processor, images, device)
            for resp, path, label in zip(responses, paths, labels):
                pred = parse_response(resp)
                all_responses.append(resp.strip())
                all_paths.append(path)
                all_labels.append(label)
                # Map to probabilities for metric computation
                if pred is None:
                    all_probs.append(0.5)
                else:
                    all_probs.append(1.0 if pred == 1 else 0.0)
        else:
            probs = score_candidates(
                model, processor, images, ["NORMAL", "PNEUMONIA"], device, length_norm
            )
            pneumonia_probs = probs[:, 1].detach().cpu().tolist()
            all_probs.extend(pneumonia_probs)
            all_labels.extend(labels)
            all_paths.extend(paths)

    metrics = compute_metrics(all_labels, all_probs)
    print(f"  Total images: {len(all_labels)}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="./data/chest_xray",
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
    parser.add_argument(
        "--split_file",
        type=str,
        default="",
        help="Optional split JSON file. If set, dataset_dir is ignored.",
    )
    parser.add_argument(
        "--scoring",
        type=str,
        default="loglik",
        choices=["loglik", "generate"],
        help="Scoring method for zero-shot",
    )
    parser.add_argument(
        "--length_norm",
        type=str,
        default="avg",
        choices=["avg", "sum"],
        help="Length normalization for log-likelihood scoring",
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

    if args.split_file:
        split_data = load_split_file(args.split_file)
    else:
        split_data = None

    for split in splits:
        if split_data is not None:
            samples = split_data.get(split, [])
        else:
            samples = list_split_samples(args.dataset_dir, split)

        if len(samples) == 0:
            print(f"  No images found for split '{split}', skipping.")
            continue

        dataset = ImageDataset(samples)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

        metrics = evaluate_split(
            model, processor, loader, device, split, args.scoring, args.length_norm
        )
        results[split] = metrics

    print("\n" + "=" * 60)
    print("ZERO-SHOT QWEN2-VL CLASSIFICATION RESULTS")
    print("=" * 60)
    for split, r in results.items():
        print(f"  {split:6s} | Accuracy: {r['accuracy']:.4f}")
    print("=" * 60)

    # Save results as JSON for comparison with MLP
    if "test" in results:
        save_results = {
            "model": args.model,
            "scoring": args.scoring,
            "length_norm": args.length_norm,
            "test": results["test"],
        }
        os.makedirs("./output", exist_ok=True)
        save_path = "./output/zero_shot_results.json"
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(save_results, f, indent=2)
        print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()
