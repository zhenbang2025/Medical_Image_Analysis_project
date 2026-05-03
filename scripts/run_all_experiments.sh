#!/bin/bash
set -e

# ---------------------------
# User-configurable variables
# ---------------------------
DATASET_DIR="./data/chest_xray"
SPLIT_FILE="./splits/full_split.json"
MODEL_DIR="./models/Qwen2-VL-2B-Instruct"
EMBEDDINGS_ROOT="./embeddings"
OUTPUT_ROOT="./output"
POOLINGS=("all_tokens" "vision_only")
RUN_MEDCLIP=0

mkdir -p "$OUTPUT_ROOT" "$(dirname "$SPLIT_FILE")"

# ---------------------------
# 1) Build split file
# ---------------------------
python scripts/build_splits.py \
  --dataset_dir "$DATASET_DIR" \
  --val_fraction 0.1 \
  --seed 42 \
  --output_file "$SPLIT_FILE"

# ---------------------------
# 2) Extract embeddings + probe training (pooling ablation)
# ---------------------------
for pooling in "${POOLINGS[@]}"; do
  EMBEDDING_DIR="$EMBEDDINGS_ROOT/$pooling"
  OUTPUT_DIR="$OUTPUT_ROOT/probe_$pooling"
  mkdir -p "$EMBEDDING_DIR" "$OUTPUT_DIR"

  python scripts/extract_embeddings.py \
    --split_file "$SPLIT_FILE" \
    --model "$MODEL_DIR" \
    --device cuda \
    --batch_size 8 \
    --output_dir "$EMBEDDING_DIR" \
    --pooling "$pooling"

  python scripts/train_probe.py \
    --embedding_dir "$EMBEDDING_DIR" \
    --model_type linear \
    --epochs 30 \
    --lr 1e-3 \
    --batch_size 64 \
    --run_name "pooling_$pooling" \
    --output_dir "$OUTPUT_DIR"

  python scripts/train_probe.py \
    --embedding_dir "$EMBEDDING_DIR" \
    --model_type mlp \
    --epochs 50 \
    --lr 1e-4 \
    --batch_size 64 \
    --run_name "pooling_$pooling" \
    --output_dir "$OUTPUT_DIR"
done

# ---------------------------
# 3) Zero-shot baseline
# ---------------------------
python scripts/evaluate_zero_shot.py \
  --split_file "$SPLIT_FILE" \
  --model "$MODEL_DIR" \
  --device cuda \
  --batch_size 1 \
  --scoring loglik \
  --length_norm avg \
  --split test

# ---------------------------
# ---------------------------
# 4) CNN baseline
# ---------------------------
python scripts/train_cnn.py \
  --split_file "$SPLIT_FILE" \
  --model resnet18 \
  --epochs 10 \
  --lr 1e-4 \
  --batch_size 32 \
  --output_dir "$OUTPUT_ROOT"

# ---------------------------
# 5) Data efficiency sweep (probe, default pooling: vision_only)
# ---------------------------
EMBEDDING_DIR="$EMBEDDINGS_ROOT/vision_only"
OUTPUT_DIR="$OUTPUT_ROOT/probe_vision_only"
mkdir -p "$EMBEDDING_DIR" "$OUTPUT_DIR"
for frac in 0.01 0.05 0.1 0.2 1.0; do
  for seed in 1 2 3; do
    python scripts/train_probe.py \
      --embedding_dir "$EMBEDDING_DIR" \
      --model_type mlp \
      --epochs 30 \
      --lr 1e-4 \
      --batch_size 64 \
      --train_fraction "$frac" \
      --seed "$seed" \
      --run_name "pooling_vision_only" \
      --output_dir "$OUTPUT_DIR"
  done
 done

# ---------------------------
# 6) Robustness evaluation (CNN)
# ---------------------------
python scripts/robustness_eval.py \
  --split_file "$SPLIT_FILE" \
  --model resnet18 \
  --model_path "$OUTPUT_ROOT/cnn_resnet18.pt" \
  --batch_size 32 \
  --output_dir "$OUTPUT_ROOT"

# ---------------------------
# 7) Medical CLIP baseline (optional)
# ---------------------------
if [ "$RUN_MEDCLIP" -eq 1 ]; then
  python scripts/medclip_baseline.py \
    --split_file "$SPLIT_FILE" \
    --output_dir "$OUTPUT_ROOT"
fi
