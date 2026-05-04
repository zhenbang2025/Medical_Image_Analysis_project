"""
Extract normalized bbox datasets with selectable embedding pooling modes.

Key features:
1. Patient-level split (train/val/test) to reduce leakage.
2. Box normalization to [0, 1] coordinates.
3. Two embedding modes:
   - all_token_mean
   - image_token_mean
4. Non-overwriting run directory output.
"""

import argparse
import os
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from utils.bbox_utils import normalize_xywh
from utils.data_utils import cap_per_class, find_image_paths, load_bbox_rows, stratified_patient_split
from utils.env_utils import env_default, env_int
from utils.run_utils import prepare_run_dir, resolve_device, save_json


CLASS_PROMPTS = {
    "Atelectasis": "Locate atelectasis in this chest X-ray. Focus on collapsed lung regions.",
    "Effusion": "Locate pleural effusion in this chest X-ray. Focus on fluid accumulation zones.",
    "Cardiomegaly": "Locate cardiomegaly in this chest X-ray. Focus on enlarged heart silhouette.",
}


def build_inputs(processor, image: Image.Image, prompt: str):
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": prompt}],
        }
    ]
    text_inputs = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return processor(text=text_inputs, images=[image], padding=True, return_tensors="pt")


def pooled_embedding(outputs, inputs, model, mode: str) -> torch.Tensor:
    hs = outputs.hidden_states[-1]  # [B, T, D]
    if mode == "all_token_mean":
        return hs.mean(dim=1)

    image_token_id = getattr(model.config, "image_token_id", None)
    if image_token_id is None or "input_ids" not in inputs:
        return hs.mean(dim=1)

    mask = (inputs["input_ids"] == image_token_id).unsqueeze(-1)  # [B, T, 1]
    counts = mask.sum(dim=1).clamp(min=1)
    return (hs * mask).sum(dim=1) / counts


def extract_split_embeddings(split_rows, model, processor, device, mode, class2idx):
    embs, bbox_norm, bbox_abs, labels, sizes, meta = [], [], [], [], [], []
    for row in tqdm(split_rows, desc="Extracting", leave=False):
        img = Image.open(row["path"]).convert("RGB")
        w, h = img.size
        inputs = build_inputs(processor, img, CLASS_PROMPTS[row["class"]])
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            emb = pooled_embedding(outputs, inputs, model, mode).cpu().squeeze(0)

        abs_box = torch.tensor([row["x"], row["y"], row["w"], row["h"]], dtype=torch.float32)
        sz = torch.tensor([w, h], dtype=torch.float32)
        norm_box = normalize_xywh(abs_box.unsqueeze(0), sz.unsqueeze(0)).squeeze(0)

        embs.append(emb)
        bbox_abs.append(abs_box)
        bbox_norm.append(norm_box)
        labels.append(class2idx[row["class"]])
        sizes.append(sz)
        meta.append(
            {
                "name": row["name"],
                "path": row["path"],
                "class": row["class"],
                "patient_id": row["patient_id"],
                "width": float(w),
                "height": float(h),
                "bbox_abs_xywh": [row["x"], row["y"], row["w"], row["h"]],
                "bbox_norm_xywh": norm_box.tolist(),
            }
        )
    return {
        "embeddings": torch.stack(embs),
        "bbox_abs_xywh": torch.stack(bbox_abs),
        "bbox_norm_xywh": torch.stack(bbox_norm),
        "image_sizes_wh": torch.stack(sizes),
        "labels": torch.tensor(labels, dtype=torch.long),
        "meta": meta,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=env_default("DATA_ROOT", "./data/3"))
    parser.add_argument("--bbox_csv", type=str, default=env_default("BBOX_CSV", "./data/3/BBox_List_2017.csv"))
    parser.add_argument("--classes", nargs="+", default=["Atelectasis", "Effusion", "Cardiomegaly"])
    parser.add_argument("--model", type=str, default=env_default("MODEL_PATH", "./models/Qwen2-VL-2B-Instruct"))
    parser.add_argument("--device", type=str, default=env_default("DEVICE", "auto"), choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--embedding_mode", type=str, default="all_token_mean",
                        choices=["all_token_mean", "image_token_mean"])
    parser.add_argument("--max_per_class", type=int, default=env_int("MAX_PER_CLASS", 2200),
                        help="Cap each class to this many rows before split.")
    parser.add_argument("--seed", type=int, default=env_int("SEED", 42))
    parser.add_argument("--output_root", type=str, default=env_default("EMBEDDINGS_ROOT", "./embeddings"))
    parser.add_argument("--run_name", type=str, default=env_default("RUN_NAME"))
    parser.add_argument("--allow_overwrite", action="store_true")
    args = parser.parse_args()

    run_dir = prepare_run_dir(
        output_root=args.output_root,
        run_name=args.run_name,
        prefix=f"emb_{args.embedding_mode}",
        allow_overwrite=args.allow_overwrite,
    )
    os.makedirs(run_dir, exist_ok=True)

    image_map = find_image_paths(args.data_root)
    rows = load_bbox_rows(args.bbox_csv, args.classes)
    rows = [r for r in rows if r["name"] in image_map]
    for r in rows:
        r["path"] = image_map[r["name"]]
    rows = cap_per_class(rows, args.classes, args.max_per_class, seed=args.seed)

    split = stratified_patient_split(rows, args.classes, train_ratio=0.7, val_ratio=0.15, seed=args.seed)
    class2idx = {c: i for i, c in enumerate(args.classes)}

    device, dtype = resolve_device(args.device)
    print(f"Device={device}, dtype={dtype}, mode={args.embedding_mode}")
    print(f"Loading model: {args.model}")
    if device.type == "cpu":
        model = Qwen2VLForConditionalGeneration.from_pretrained(args.model, torch_dtype=dtype, device_map="cpu")
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(args.model, torch_dtype=dtype, device_map="auto")
    processor = AutoProcessor.from_pretrained(args.model)

    for s in ("train", "val", "test"):
        total = len(split[s])
        print(f"\n=== {s} ({total}) ===")
        for cls in args.classes:
            cnt = sum(1 for r in split[s] if r["class"] == cls)
            print(f"  {cls}: {cnt}")

    train_data = extract_split_embeddings(split["train"], model, processor, device, args.embedding_mode, class2idx)
    val_data = extract_split_embeddings(split["val"], model, processor, device, args.embedding_mode, class2idx)
    test_data = extract_split_embeddings(split["test"], model, processor, device, args.embedding_mode, class2idx)

    # Build split statistics report
    split_stats = {}
    for s in ("train", "val", "test"):
        split_stats[s] = {"total": len(split[s]), "per_class": {}}
        for cls in args.classes:
            split_stats[s]["per_class"][cls] = sum(1 for r in split[s] if r["class"] == cls)

    torch.save({k: v for k, v in train_data.items() if k != "meta"}, os.path.join(run_dir, "train.pt"))
    torch.save({k: v for k, v in val_data.items() if k != "meta"}, os.path.join(run_dir, "val.pt"))
    torch.save({k: v for k, v in test_data.items() if k != "meta"}, os.path.join(run_dir, "test.pt"))

    save_json(os.path.join(run_dir, "meta.json"), {
        "train": train_data["meta"],
        "val": val_data["meta"],
        "test": test_data["meta"],
    })
    save_json(os.path.join(run_dir, "split_stats.json"), split_stats)
    save_json(os.path.join(run_dir, "class_names.json"), {"classes": args.classes, "class2idx": class2idx})
    save_json(os.path.join(run_dir, "class_prompts.json"), CLASS_PROMPTS)
    save_json(
        os.path.join(run_dir, "config.json"),
        {
            "data_root": str(Path(args.data_root).resolve()),
            "bbox_csv": str(Path(args.bbox_csv).resolve()),
            "model": args.model,
            "device": str(device),
            "dtype": str(dtype),
            "embedding_mode": args.embedding_mode,
            "max_per_class": args.max_per_class,
            "seed": args.seed,
            "run_dir": str(Path(run_dir).resolve()),
        },
    )
    print(f"Saved embedding run to: {run_dir}")


if __name__ == "__main__":
    main()
