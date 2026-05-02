# Medical Image Analysis: Chest X-Ray Classification

Classify chest X-rays as **NORMAL** or **PNEUMONIA** using Qwen2-VL vision-language model. Compares two approaches:

1. **Zero-shot** — Direct classification via prompting
2. **MLP fine-tune** — Extract image embeddings, train a lightweight MLP on top

## Setup

```bash
pip install -r requirements.txt
```

### Dependencies

torch, transformers, pillow, tqdm, accelerate, qwen-vl-utils, matplotlib, kagglehub

## Dataset

Download the Chest X-Ray Pneumonia dataset from Kaggle:

```bash
python data/download.py
```

The dataset is saved to `data/chest_xray/`:

```
data/chest_xray/
├── train/NORMAL/       (1,341 images)
├── train/PNEUMONIA/    (3,875 images)
├── val/NORMAL/         (8 images)
├── val/PNEUMONIA/      (8 images)
├── test/NORMAL/        (234 images)
└── test/PNEUMONIA/     (390 images)
```

### Small Dataset (quick test)

A small subset at `data/chest_xray_small/` is provided for fast prototyping:

```
├── train/  (30 images: 10 NORMAL + 20 PNEUMONIA)
├── val/    (16 images: 8 NORMAL + 8 PNEUMONIA)
└── test/   (20 images: 8 NORMAL + 12 PNEUMONIA)
```

## Model

Download Qwen2-VL-2B-Instruct:

```bash
# Option 1: HuggingFace
huggingface-cli download Qwen/Qwen2-VL-2B-Instruct --local-dir models/Qwen2-VL-2B-Instruct

# Option 2: ModelScope
modelscope download --model Qwen/Qwen2-VL-2B-Instruct --local_dir models/Qwen2-VL-2B-Instruct
```

## Usage

All scripts default to the small dataset (`data/chest_xray_small`). For the full dataset, add `--dataset_dir ./data/chest_xray`.

### Step 1: Zero-shot Evaluation

```bash
python scripts/evaluate_zero_shot.py --device cpu --split test
```

Results → `output/zero_shot_results.json`

### Step 2: Extract Embeddings

```bash
python scripts/extract_embeddings.py --device cpu
```

Embeddings → `embeddings/{train,val,test}.pt`

### Step 3: Train MLP

```bash
python scripts/train_classifier.py
```

Outputs: `output/mlp_classifier.pt`, `output/results.json`, `output/training_curves.png`

### Full Dataset Example

```bash
python scripts/evaluate_zero_shot.py --dataset_dir ./data/chest_xray --split test --device cpu
python scripts/extract_embeddings.py --dataset_dir ./data/chest_xray --device cpu
python scripts/train_classifier.py
```

## Common Arguments

| Argument | Description | Default |
|---|---|---|
| `--model` | Model path | `./models/Qwen2-VL-2B-Instruct` |
| `--dataset_dir` | Dataset path | `./data/chest_xray_small` |
| `--device` | `auto`, `mps`, `cuda`, or `cpu` | `auto` |
| `--batch_size` | Images per batch | `1` (zero-shot), `4` (extract) |

## Results

### Small Dataset (Qwen2-VL-2B-Instruct)

| Method | Test Accuracy |
|---|---|
| Zero-shot (Qwen2-VL prompting) | 0.4000 |
| Qwen embeddings + MLP | 1.0000 |

> **Note**: The small dataset (66 images) is for rapid prototyping. MLP accuracy is inflated due to limited data. Use the full dataset for realistic evaluation.

### Why MLP Works Better

The MLP classifier learns to map embeddings to labels from supervised examples, while zero-shot relies solely on the model's pre-trained knowledge and prompt formatting. Even with a small labeled set, supervised fine-tuning significantly outperforms zero-shot prompting on this task.

## Output Files

```
output/
├── zero_shot_results.json   # Zero-shot metrics
├── results.json             # MLP training metrics
├── mlp_classifier.pt        # Trained MLP weights
└── training_curves.png      # Loss & accuracy curves

embeddings/
├── train.pt
├── val.pt
└── test.pt
```
