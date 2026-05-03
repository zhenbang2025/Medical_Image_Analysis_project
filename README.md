# Chest X-Ray Bounding Box Localization

Adapt Qwen2-VL vision-language model for medical image bounding box localization by training lightweight MLP heads on top of frozen image embeddings.

## Overview

This project compares two paradigms for adapting large vision-language models (VLMs) to medical image bounding box localization:

1. **Zero-shot / Few-shot prompting** — Query the VLM directly to predict bbox coordinates
2. **MLP fine-tuning** — Extract image embeddings with class-specific prompts, train a small MLP on top

| Dataset | Classes | Train / Test per class |
|---|---|---|
| NIH ChestX-ray14 | Atelectasis, Effusion, Cardiomegaly | 30 / 10 |

## Results

### Overall Performance

| Method | Mean IoU | IoU@0.25 | IoU@0.5 |
|---|---|---|---|
| Plain MLP | 0.047 | 0.067 | 0.000 |
| **MLP + Residual** | **0.219** | **0.433** | **0.100** |
| Qwen Zero-shot | 0.070 | 0.167 | 0.000 |
| **Qwen Few-shot (n=3)** | **0.254** | **0.467** | **0.167** |

### Per-class Breakdown

| Class | Plain MLP | MLP+Residual | Qwen Zero-shot | Qwen Few-shot (n=3) |
|---|---|---|---|---|
| **Atelectasis** | 0.014 / 0.000 | 0.098 / 0.100 | 0.036 / 0.100 | 0.064 / 0.100 |
| **Effusion** | 0.124 / 0.200 | 0.119 / 0.200 | 0.063 / 0.100 | 0.186 / 0.400 |
| **Cardiomegaly** | 0.004 / 0.000 | 0.440 / 1.000 | 0.110 / 0.300 | 0.511 / 0.900 |

*Per-class values show Mean IoU / IoU@0.25*

### Analysis

- **Residual connections are essential**: Plain MLP collapses to near-zero predictions (all samples converge to a single point). Adding residual connections with LayerNorm improves mean IoU by **213.7%** over zero-shot.
- **Few-shot still leads**: Qwen with 3 exemplars achieves 0.254 IoU, 13.8% above MLP+Residual. The VLM's pre-trained spatial reasoning provides an edge that small supervised sets can't fully close.
- **Cardiomegaly is easiest**: Largest, most anatomically stable structure. MLP+Residual reaches 0.44 IoU (100% @0.25), few-shot reaches 0.51 (90% @0.25).
- **Atelectasis is hardest**: Small, variable regions. All methods struggle (< 0.1 IoU). Suggests need for more data or class-specific architectures.
- **Effusion is mixed**: High variance — some samples are well-localized (large effusions), others fail (small/subtle ones).

## Setup

```bash
pip install -r requirements.txt
```

### Dependencies

torch, transformers, pillow, tqdm, accelerate, qwen-vl-utils, matplotlib

## Quick Start

```bash
# 1. Extract bbox embeddings (class-specific prompts baked in)
python scripts/extract_bbox_embeddings.py --model ./models/Qwen2-VL-2B-Instruct

# 2. Train plain MLP
python scripts/train_bbox_mlp.py --epochs 100 --lr 1e-3

# 3. Train MLP with residual connections
python scripts/train_bbox_mlp.py --epochs 100 --lr 1e-3 --use_residual

# 4. Evaluate both variants
python scripts/evaluate_bbox.py --model_path ./output/plain/bbox_mlp.pt
python scripts/evaluate_bbox.py --model_path ./output/residual/bbox_mlp.pt

# 5. Zero-shot + few-shot Qwen baselines
python scripts/evaluate_few_shot.py --n_shot 0
python scripts/evaluate_few_shot.py --n_shot 3

# 6. Compare all 4 methods
python scripts/compare_results.py
```

### Mac-Specific Notes

- Uses `float16` on MPS (Apple Silicon) to reduce memory (~2-4GB for 2B/3B model)
- If OOM, reduce `--batch_size` to 1 or 2
- MLP training is extremely lightweight and runs on CPU/MPS

## Architecture

### Embedding Extraction

Image + class-specific prompt → Qwen2-VL → mean-pool `last_hidden_state` → `(B, hidden_dim)` embedding. The prompt (e.g., "Describe lung collapse in this X-ray.") encodes class semantics into the embedding, so the downstream MLP doesn't need class labels as input.

### BBox MLP - Plain

```
Linear(D→256) → ReLU → Dropout → Linear(256→128) → ReLU → Dropout → Linear(128→4)
```

### BBox MLP - Residual

```
h = Linear(D→256)
h = LayerNorm(h) → Linear(256→128) → ReLU → Dropout → Linear(128→256) + Linear(256→256)  # residual
out = LayerNorm(h) → Linear(256→128) → ReLU → Dropout → Linear(128→4)
```

### Few-shot / Zero-shot Qwen

Direct text generation from Qwen2-VL. Zero-shot: class description prompt only. Few-shot: prompt includes 3 ground-truth bbox examples before the query image.

## Project Structure

```
├── scripts/
│   ├── extract_bbox_embeddings.py     # Extract bbox embeddings (3 classes)
│   ├── train_bbox_mlp.py              # Train bbox MLP (plain + residual)
│   ├── evaluate_bbox.py               # Evaluate bbox MLP + visualize
│   ├── evaluate_few_shot.py           # Qwen zero/few-shot bbox baseline
│   └── compare_results.py             # Compare all 4 methods → table
├── output/
│   ├── plain/                         # Plain MLP results
│   │   ├── bbox_mlp.pt
│   │   └── bbox_eval.json
│   ├── residual/                      # Residual MLP results
│   │   ├── bbox_mlp.pt
│   │   └── bbox_eval.json
│   ├── zero_shot_eval.json            # Qwen zero-shot bbox
│   ├── few_shot_3_eval.json           # Qwen few-shot bbox (n=3)
│   └── comparison.json                # All methods compared
├── embeddings/                        # Pre-extracted Qwen embeddings
├── requirements.txt
└── README.md
```

## Key Script Arguments

| Argument | Description | Default |
|---|---|---|
| `--model` | Qwen model path | `./models/Qwen2-VL-2B-Instruct` |
| `--device` | `auto`, `mps`, `cuda`, `cpu` | `auto` |
| `--batch_size` | Images per batch | `4` |
| `--epochs` | Training epochs | `100` |
| `--lr` | Learning rate | `1e-3` |
| `--use_residual` | Use residual connections | Off |
| `--n_shot` | Number of few-shot examples | `0` |

## Why This Approach

Training a small MLP on frozen VLM embeddings is a **parameter-efficient** way to adapt large models to domain-specific tasks:

- **No VLM fine-tuning needed** — Qwen stays frozen, only the MLP (a few hundred KB) is trained
- **Fast iteration** — Embedding extraction is one-time; MLP training takes seconds
- **Fair comparison** — Same embeddings, same test set, only the head differs (MLP vs prompt)
- **Deployable** — The MLP checkpoint is tiny and can run independently once embeddings are computed

## License

This project is for research and educational purposes.
