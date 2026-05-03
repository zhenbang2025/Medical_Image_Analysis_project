"""
Baseline: Qwen2-VL directly predicts bbox (no MLP).

Two modes:
  --n_shot 0  : Zero-shot (only class-specific instruction)
  --n_shot >0 : Few-shot (with training examples in prompt)

Compares Qwen text-generation bbox vs MLP regression bbox.

Usage:
    python evaluate_few_shot.py --n_shot 0    # Zero-shot baseline
    python evaluate_few_shot.py --n_shot 3    # Few-shot baseline
"""

import argparse
import json
import os
import re
import torch
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor


CLASS_PROMPTS = {
    "Atelectasis": "Locate the lung collapse. Output bbox: x y w h",
    "Effusion": "Locate the pleural effusion. Output bbox: x y w h",
    "Cardiomegaly": "Locate the heart enlargement. Output bbox: x y w h",
}


def compute_iou(pred, true):
    pred_x1, pred_y1 = pred[0], pred[1]
    pred_x2, pred_y2 = pred[0] + pred[2], pred[1] + pred[3]
    true_x1, true_y1 = true[0], true[1]
    true_x2, true_y2 = true[0] + true[2], true[1] + true[3]
    inter_x1 = max(pred_x1, true_x1)
    inter_y1 = max(pred_y1, true_y1)
    inter_x2 = min(pred_x2, true_x2)
    inter_y2 = min(pred_y2, true_y2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    pred_area = pred[2] * pred[3]
    true_area = true[2] * true[3]
    union_area = pred_area + true_area - inter_area
    return inter_area / (union_area + 1e-6)


def parse_bbox_response(text):
    """Parse bbox from response. Try multiple formats."""
    # Format: "x=100 y=200 w=50 h=60" or "x=100, y=200, w=50, h=60"
    m = re.search(
        r'x[=:\s]+([\d.]+)[,\s]*y[=:\s]+([\d.]+)[,\s]*w[=:\s]+([\d.]+)[,\s]*h[=:\s]+([\d.]+)',
        text, re.IGNORECASE
    )
    if m:
        return [float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))]

    # Format: "[100, 200, 50, 60]" or "(100, 200, 50, 60)"
    m = re.search(r'[\[\(]\s*([\d.]+)[,\s]+([\d.]+)[,\s]+([\d.]+)[,\s]+([\d.]+)\s*[\]\)]', text)
    if m:
        return [float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))]

    # Format: four consecutive numbers
    nums = re.findall(r'(\d+\.?\d*)', text)
    if len(nums) >= 4:
        return [float(nums[-4]), float(nums[-3]), float(nums[-2]), float(nums[-1])]

    return None


def build_prompt(cls_name, n_shot, train_meta):
    """Build prompt for a given class."""
    lines = [CLASS_PROMPTS[cls_name]]

    if n_shot > 0:
        # Add few-shot examples
        examples = [m for m in train_meta if m["class"] == cls_name][:n_shot]
        if examples:
            lines.append("Examples:")
            for ex in examples:
                lines.append(f"  GT: x={ex['x']:.0f} y={ex['y']:.0f} w={ex['w']:.0f} h={ex['h']:.0f}")

    lines.append("\nOutput ONLY four numbers: x y w h")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_dir", type=str, default="./embeddings")
    parser.add_argument("--model", type=str, default="./models/Qwen2-VL-2B-Instruct")
    parser.add_argument("--n_shot", type=int, default=0,
                        help="0=zero-shot, >0=few-shot examples per class")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "mps", "cuda", "cpu"])
    parser.add_argument("--output_dir", type=str, default="./output")
    args = parser.parse_args()

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

    mode = "zero_shot" if args.n_shot == 0 else f"few_shot_{args.n_shot}"
    print(f"Mode: {mode}")
    print(f"Device: {device}")
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

    # Load data
    with open(os.path.join(args.embedding_dir, "bbox_meta.json")) as f:
        meta = json.load(f)
    with open(os.path.join(args.embedding_dir, "class_names.json")) as f:
        class_info = json.load(f)
    classes = class_info["classes"]

    train_meta = meta["train"]
    test_meta = meta["test"]

    # Evaluate
    results = []
    desc = "Zero-shot" if args.n_shot == 0 else f"Few-shot (n={args.n_shot})"
    print(f"\nRunning {desc} bbox prediction on {len(test_meta)} test samples...")

    for i, info in enumerate(tqdm(test_meta, desc=desc)):
        img = Image.open(info["path"]).convert("RGB")
        true_bbox = [info["x"], info["y"], info["w"], info["h"]]
        cls_name = info["class"]

        prompt = build_prompt(cls_name, args.n_shot, train_meta)

        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }]

        text_inputs = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=text_inputs,
            images=[img],
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=32,
                temperature=0.1,
            )
            input_len = inputs.input_ids.shape[1]
            generated_tokens = generated_ids[:, input_len:]

        response = processor.batch_decode(
            generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        pred_bbox = parse_bbox_response(response)
        if pred_bbox is not None:
            iou = compute_iou(pred_bbox, true_bbox)
        else:
            pred_bbox = [0, 0, 0, 0]
            iou = 0.0

        results.append({
            "class": cls_name,
            "bbox_gt": true_bbox,
            "bbox_pred": pred_bbox,
            "iou": iou,
            "response": response.strip()[:100],
            "parsed": pred_bbox is not None,
        })

    # Metrics
    all_ious = [r["iou"] for r in results]
    parsed_count = sum(1 for r in results if r["parsed"])

    print(f"\n{'='*50}")
    print(f"QWEN2-VL {mode.upper()} BBOX RESULTS")
    print(f"{'='*50}")
    print(f"Total: {len(results)} | Parsed: {parsed_count}/{len(results)}")
    print(f"Mean IoU: {sum(all_ious)/len(all_ious):.4f}")
    print(f"IoU@0.25: {sum(1 for i in all_ious if i >= 0.25)/len(all_ious):.4f}")
    print(f"IoU@0.5:  {sum(1 for i in all_ious if i >= 0.5)/len(all_ious):.4f}")

    print(f"\n{'Class':<20} {'Mean IoU':>10} {'@0.25':>8} {'@0.5':>8}")
    print("-" * 48)
    per_class = {}
    for cls in classes:
        c_results = [r for r in results if r["class"] == cls]
        if not c_results:
            continue
        c_ious = [r["iou"] for r in c_results]
        a25 = sum(1 for i in c_ious if i >= 0.25) / len(c_ious)
        a50 = sum(1 for i in c_ious if i >= 0.5) / len(c_ious)
        mean_iou = sum(c_ious) / len(c_ious)
        print(f"{cls:<20} {mean_iou:>10.4f} {a25:>8.4f} {a50:>8.4f}")
        per_class[cls] = {
            "count": len(c_results),
            "mean_iou": mean_iou,
            "acc_0.25": a25,
            "acc_0.5": a50,
        }

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    output = {
        "method": mode,
        "n_shot": args.n_shot,
        "mean_iou": sum(all_ious) / len(all_ious),
        "iou_at_0.25": sum(1 for i in all_ious if i >= 0.25) / len(all_ious),
        "iou_at_0.5": sum(1 for i in all_ious if i >= 0.5) / len(all_ious),
        "parsed_count": parsed_count,
        "per_class": per_class,
        "per_sample": results,
    }
    out_file = os.path.join(args.output_dir, f"{mode}_eval.json")
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
