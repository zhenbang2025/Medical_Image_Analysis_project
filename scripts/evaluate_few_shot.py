"""
Evaluate zero-shot / few-shot Qwen2-VL baseline for bbox localization.
"""

import argparse
import os
import re

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from utils.bbox_utils import iou_xywh, normalize_xywh
from utils.env_utils import env_default, env_int
from utils.run_utils import load_json, prepare_run_dir, resolve_device, save_json


CLASS_PROMPTS = {
    "Atelectasis": "Locate atelectasis region. Output bbox as x y w h.",
    "Effusion": "Locate pleural effusion region. Output bbox as x y w h.",
    "Cardiomegaly": "Locate enlarged heart region (cardiomegaly). Output bbox as x y w h.",
}


def parse_bbox_response(text: str):
    m = re.search(
        r"x[=:\s]+([\d.]+)[,\s]*y[=:\s]+([\d.]+)[,\s]*w[=:\s]+([\d.]+)[,\s]*h[=:\s]+([\d.]+)",
        text, re.IGNORECASE,
    )
    if m:
        return [float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))]
    m = re.search(r"[\[\(]\s*([\d.]+)[,\s]+([\d.]+)[,\s]+([\d.]+)[,\s]+([\d.]+)\s*[\]\)]", text)
    if m:
        return [float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))]
    nums = re.findall(r"(\d+\.?\d*)", text)
    if len(nums) >= 4:
        return [float(nums[-4]), float(nums[-3]), float(nums[-2]), float(nums[-1])]
    return None


def prompt_for_sample(cls_name: str, train_meta: list[dict], n_shot: int):
    lines = [CLASS_PROMPTS[cls_name]]
    if n_shot > 0:
        examples = [m for m in train_meta if m["class"] == cls_name][:n_shot]
        lines.append("Examples:")
        for ex in examples:
            x, y, w, h = ex["bbox_abs_xywh"]
            lines.append(f"GT: x={x:.0f} y={y:.0f} w={w:.0f} h={h:.0f}")
    lines.append("Output ONLY four numbers: x y w h")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    embedding_default = env_default("EMBEDDING_DIR")
    parser.add_argument("--embedding_dir", type=str, default=embedding_default, required=embedding_default is None,
                        help="Embedding run dir to read meta split from.")
    parser.add_argument("--model", type=str, default=env_default("MODEL_PATH", "./models/Qwen2-VL-2B-Instruct"))
    parser.add_argument("--n_shot", type=int, default=env_int("N_SHOT", 0))
    parser.add_argument("--device", type=str, default=env_default("DEVICE", "auto"), choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--output_root", type=str, default=env_default("OUTPUT_VLM_ROOT", "./output/vlm"))
    parser.add_argument("--run_name", type=str, default=env_default("RUN_NAME"))
    parser.add_argument("--allow_overwrite", action="store_true")
    args = parser.parse_args()

    mode = f"few_shot_{args.n_shot}" if args.n_shot > 0 else "zero_shot"
    run_dir = prepare_run_dir(args.output_root, args.run_name, prefix=mode, allow_overwrite=args.allow_overwrite)
    device, dtype = resolve_device(args.device)

    print(f"Mode={mode} | device={device}")
    if device.type == "cpu":
        model = Qwen2VLForConditionalGeneration.from_pretrained(args.model, torch_dtype=dtype, device_map="cpu")
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(args.model, torch_dtype=dtype, device_map="auto")
    processor = AutoProcessor.from_pretrained(args.model)

    meta = load_json(os.path.join(args.embedding_dir, "meta.json"))
    classes = load_json(os.path.join(args.embedding_dir, "class_names.json"))["classes"]
    train_meta, test_meta = meta["train"], meta["test"]

    rows = []
    for info in tqdm(test_meta, desc=mode):
        cls = info["class"]
        img = Image.open(info["path"]).convert("RGB")
        prompt = prompt_for_sample(cls, train_meta, args.n_shot)
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text_inputs = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=text_inputs, images=[img], padding=True, return_tensors="pt")
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.no_grad():
            out_ids = model.generate(**inputs, max_new_tokens=32, temperature=0.1)
            gen_tokens = out_ids[:, inputs["input_ids"].shape[1]:]
        resp = processor.batch_decode(gen_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        pred_abs = parse_bbox_response(resp)
        gt_abs = info["bbox_abs_xywh"]
        w, h = float(info["width"]), float(info["height"])
        size = torch.tensor([[w, h]], dtype=torch.float32)
        gt_norm = normalize_xywh(torch.tensor([gt_abs], dtype=torch.float32), size).squeeze(0)
        if pred_abs is None:
            pred_norm = torch.tensor([0, 0, 0, 0], dtype=torch.float32)
            parsed = False
        else:
            pred_norm = normalize_xywh(torch.tensor([pred_abs], dtype=torch.float32), size).squeeze(0)
            parsed = True
        iou = float(iou_xywh(pred_norm.unsqueeze(0), gt_norm.unsqueeze(0))[0])
        rows.append(
            {
                "name": info["name"],
                "class": cls,
                "bbox_gt_abs_xywh": gt_abs,
                "bbox_pred_abs_xywh": pred_abs if pred_abs is not None else [0, 0, 0, 0],
                "bbox_gt_norm_xywh": gt_norm.tolist(),
                "bbox_pred_norm_xywh": pred_norm.tolist(),
                "iou": iou,
                "parsed": parsed,
                "response": resp.strip()[:200],
            }
        )

    all_ious = torch.tensor([r["iou"] for r in rows], dtype=torch.float32)
    result = {
        "method": mode,
        "n_shot": args.n_shot,
        "mean_iou": float(all_ious.mean()),
        "iou_at_0.25": float((all_ious >= 0.25).float().mean()),
        "iou_at_0.5": float((all_ious >= 0.5).float().mean()),
        "parsed_count": int(sum(1 for r in rows if r["parsed"])),
        "per_class": {},
        "per_sample": rows,
    }
    for cls in classes:
        sub = [r for r in rows if r["class"] == cls]
        if not sub:
            continue
        s = torch.tensor([r["iou"] for r in sub], dtype=torch.float32)
        result["per_class"][cls] = {
            "count": len(sub),
            "mean_iou": float(s.mean()),
            "iou_at_0.25": float((s >= 0.25).float().mean()),
            "iou_at_0.5": float((s >= 0.5).float().mean()),
        }

    save_json(os.path.join(run_dir, f"{mode}_eval.json"), result)
    print(f"Saved baseline result to: {run_dir}")


if __name__ == "__main__":
    main()
