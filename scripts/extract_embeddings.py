"""
Extract image embeddings from Chest X-Ray dataset using Qwen2.5-VL.

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
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor


# Fixed text prompt paired with each image
PROMPT = "Analyze this chest X-ray image and describe any abnormalities you observe."


class ChestXRayDataset(Dataset):
    def __init__(self, root_dir, split="train"):
        self.root_dir = Path(root_dir) / split
        self.samples = []
        self.labels = []

        for label_idx, class_name in enumerate(["NORMAL", "PNEUMONIA"]):
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                # some dataset structures nest one extra level
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


def extract_embeddings(model, processor, images, device):
    """Run images through Qwen2.5-VL and return mean-pooled embeddings."""
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

    inputs = processor(
        text=text_inputs,
        images=images,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        # Get the last hidden state and mean-pool over all tokens
        hidden_states = outputs.last_hidden_state  # (B, seq_len, hidden_dim)
        embeddings = hidden_states.mean(dim=1)     # (B, hidden_dim)

    return embeddings.cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="./chest_xray/chest_xray",
        help="Path to the chest_xray subdirectory containing train/test/val",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-VL-2B-Instruct",
        help="HuggingFace model name or local path",
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Images per batch")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./embeddings",
        help="Directory to save embedding files",
    )
    args = parser.parse_args()

    # Device setup — Mac MPS preferred, fallback to CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        dtype = torch.float16
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.float16
    else:
        device = torch.device("cpu")
        dtype = torch.float32

    print(f"Device: {device}, dtype: {dtype}")

    # Load model and processor
    print(f"Loading model: {args.model} ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(args.model)
    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Process each split
    for split in ["train", "val", "test"]:
        dataset = ChestXRayDataset(args.dataset_dir, split=split)
        if len(dataset) == 0:
            print(f"  No images found for split '{split}', skipping.")
            continue

        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

        all_embeddings = []
        all_labels = []

        print(f"\n[{split}] Extracting embeddings for {len(dataset)} images...")
        for images, paths, labels in tqdm(loader, desc=split):
            embeddings = extract_embeddings(model, processor, images, device)
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
