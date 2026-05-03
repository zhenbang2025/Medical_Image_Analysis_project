"""
Extract image embeddings from Chest X-Ray dataset using Qwen2-VL.

Saves pooled embeddings + labels to .pt files for offline MLP training.

Usage:
    python extract_embeddings.py [--model Qwen/Qwen2.5-VL-2B-Instruct] [--batch_size 4]

Notes for Mac:
    - Uses float16 on MPS to reduce memory (~2-4GB for 2B/3B model)
    - If OOM, reduce batch_size to 1 or 2
    - Qwen2.5-VL-2B-Instruct is recommended for Mac with <=16GB RAM
    - Qwen2.5-VL-3B-Instruct for Mac with >=24GB RAM
"""

import argparse
import os
from typing import List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

from data_utils import ImageDataset, load_split_file, list_split_samples


# Fixed text prompt paired with each image
PROMPT = "Analyze this chest X-ray image and describe any abnormalities you observe."


def collate_fn(batch):
    images = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    paths = [item[2] for item in batch]
    return images, paths, labels


def masked_mean_pooling(hidden_states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.unsqueeze(-1).float()
    masked = hidden_states * mask
    denom = mask.sum(dim=1).clamp(min=1.0)
    return masked.sum(dim=1) / denom


def extract_embeddings(model, processor, images, device, pooling: str):
    """Run images through Qwen2-VL and return pooled embeddings."""
    texts = [PROMPT] * len(images)

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

    # Prepare inputs for the model
    text_inputs = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(text=text_inputs, images=images, padding=True, return_tensors="pt")
    inputs = inputs.to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]  # (B, seq_len, hidden_dim)
        attention_mask = inputs.attention_mask

        if pooling == "vision_only" and hasattr(processor, "image_token_id"):
            image_mask = inputs.input_ids.eq(processor.image_token_id)
            if image_mask.sum().item() == 0:
                # Fallback if vision tokens are not detected
                embeddings = masked_mean_pooling(hidden_states, attention_mask)
            else:
                embeddings = masked_mean_pooling(hidden_states, image_mask)
        else:
            embeddings = masked_mean_pooling(hidden_states, attention_mask)

    return embeddings.cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="./data/chest_xray",
        help="Path to the chest_xray subdirectory containing train/test/val",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="./models/Qwen2-VL-2B-Instruct",
        help="Path to local model directory or HuggingFace model name",
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Images per batch")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "mps", "cuda", "cpu"],
        help="Device to use: auto (prefer mps/cuda), or force cpu",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./embeddings",
        help="Directory to save embedding files",
    )
    parser.add_argument(
        "--split_file",
        type=str,
        default="",
        help="Optional split JSON file. If set, dataset_dir is ignored.",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        default="all_tokens",
        choices=["all_tokens", "vision_only"],
        help="Pooling mode for embeddings",
    )
    args = parser.parse_args()

    # Device setup — Mac MPS preferred, fallback to CPU
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

    # Load model and processor
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
    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Prepare splits
    if args.split_file:
        splits = load_split_file(args.split_file)
    else:
        splits = {
            split: list_split_samples(args.dataset_dir, split)
            for split in ["train", "val", "test"]
        }

    # Constrain max_pixels to avoid OOM (default 12845056 is too large for batch processing)
    # 3136*28*28 ≈ 2.4M pixels is sufficient for 224x224 images; use 608x608 as practical limit
    img_proc_dict = processor.image_processor.to_dict()
    img_proc_dict["max_pixels"] = 608 * 608  # ~370K pixels, well below the 2.4M+ default
    from transformers import Qwen2VLImageProcessor
    processor.image_processor = Qwen2VLImageProcessor(**img_proc_dict)
    print(f"  Image processor max_pixels limited to {processor.image_processor.max_pixels}")

    # Process each split
    for split, samples in splits.items():
        if len(samples) == 0:
            print(f"  No images found for split '{split}', skipping.")
            continue

        dataset = ImageDataset(samples)
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
        )

        all_embeddings = []
        all_labels = []

        print(f"\n[{split}] Extracting embeddings for {len(dataset)} images...")
        for images, paths, labels in tqdm(loader, desc=split):
            embeddings = extract_embeddings(model, processor, images, device, args.pooling)
            all_embeddings.append(embeddings)
            all_labels.extend(labels)

        # Concatenate and save
        all_embeddings = torch.cat(all_embeddings, dim=0)
        all_labels = torch.tensor(all_labels, dtype=torch.long)

        output_path = os.path.join(args.output_dir, f"{split}.pt")
        torch.save({"embeddings": all_embeddings, "labels": all_labels}, output_path)
        print(f"  Saved {output_path} — embeddings shape: {all_embeddings.shape}")

    print("\nAll embeddings extracted successfully!")
    print(f"Embedding dimension: {all_embeddings.shape[1]}")


if __name__ == "__main__":
    main()
