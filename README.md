# Chest X-Ray BBox Localization (Qwen2-VL + MLP)

本项目用于 NIH ChestX-ray14 三类病灶框定位（Atelectasis / Effusion / Cardiomegaly），目标是让 **VLM+MLP** 超过 **Zero-shot / Few-shot VLM baseline**。

---

## 1. 环境配置（`mia`）

### 1.1 创建环境

```bash
conda create -n mia python=3.10 -y
conda activate mia
pip install -r requirements.txt
```

### 1.2 依赖版本说明

本仓库已与参考项目对齐共同依赖版本（例如 `torch==2.2.0`, `transformers==4.45.2`）。

---

## 2. 数据与模型准备

## 2.1 下载 NIH bbox 数据到 `data/` 下

已实现下载脚本：`data/download.py`。  
默认会把可直接使用的数据复制到 `./data/3`（而不是只留在 kagglehub 缓存路径）。

```bash
python data/download.py --target_dir ./data/3
```

执行后应至少包含：

```text
data/3/
├── BBox_List_2017.csv
├── images_001/
│   └── images/*.png
├── images_002/
│   └── images/*.png
└── ...
```

## 2.2 下载 Qwen2-VL-2B-Instruct

```bash
huggingface-cli download Qwen/Qwen2-VL-2B-Instruct --local-dir models/Qwen2-VL-2B-Instruct
```

---

## 3. 前置要求

sbatch 脚本假设项目结构如下（数据下载和模型下载见 Section 2）：

```text
PROJECT_DIR/
├── data/3/                     ← NIH 数据（images_* + BBox_List_2017.csv）
├── models/Qwen2-VL-2B-Instruct ← Qwen2-VL 模型
├── scripts/
├── sbatch/
└── ...
```

环境：脚本硬编码使用 `$HOME/.conda/envs/mia`，通过 `module load Anaconda3/2023.09-0` + `module load cuda12.2/toolkit/12.2.2` 加载系统 CUDA。

---

## 4. 数据划分逻辑（你问到的 split）

本项目**不需要单独的 split 脚本文件**，因为在 embedding 提取阶段已经内置划分：

- 代码位置：`scripts/extract_bbox_embeddings.py`
- 划分函数：`scripts/utils/data_utils.py::stratified_patient_split`
- 划分策略：按 `patient_id` 分层，默认 `train/val/test = 70/15/15`
- 固定随机种子：`--seed`（默认 42）

提取 embedding 时会同时写出：

- `train.pt`
- `val.pt`
- `test.pt`
- `meta.json`（含三个 split 的样本元信息）

---

## 5. 快速启动（命令行）

## 5.1 提取 embeddings（两种 pooling）

```bash
python scripts/extract_bbox_embeddings.py \
  --data_root "$DATA_ROOT" \
  --bbox_csv "$BBOX_CSV" \
  --model "$MODEL_PATH" \
  --embedding_mode all_token_mean \
  --max_per_class 2200 \
  --device cuda \
  --run_name emb_alltoken_manual

python scripts/extract_bbox_embeddings.py \
  --data_root "$DATA_ROOT" \
  --bbox_csv "$BBOX_CSV" \
  --model "$MODEL_PATH" \
  --embedding_mode image_token_mean \
  --max_per_class 2200 \
  --device cuda \
  --run_name emb_imagetoken_manual
```

## 5.2 训练 MLP

```bash
python scripts/train_bbox_mlp.py \
  --embedding_dir ./embeddings/emb_alltoken_manual \
  --arch residual \
  --loss ciou \
  --epochs 100 \
  --batch_size 128 \
  --device cuda \
  --run_name train_residual_ciou_manual
```

## 5.3 评估 MLP

```bash
python scripts/evaluate_bbox.py \
  --model_path ./output/train/train_residual_ciou_manual/bbox_mlp.pt \
  --embedding_dir ./embeddings/emb_alltoken_manual \
  --device cuda \
  --run_name eval_residual_ciou_manual
```

## 5.4 评估 zero/few-shot baseline

```bash
python scripts/evaluate_few_shot.py \
  --embedding_dir ./embeddings/emb_alltoken_manual \
  --model "$MODEL_PATH" \
  --n_shot 0 \
  --device cuda \
  --run_name vlm_zero_shot_manual

python scripts/evaluate_few_shot.py \
  --embedding_dir ./embeddings/emb_alltoken_manual \
  --model "$MODEL_PATH" \
  --n_shot 3 \
  --device cuda \
  --run_name vlm_few_shot3_manual
```

---

## 6. Slurm/H800 实验流程

### 6.1 提取 Embedding

```bash
# 一次性提取两种 pooling（all_token + image_token）
sbatch sbatch/run_extract_embeddings.sbatch

# 或单独提取
sbatch sbatch/extract_embeddings_alltoken.sbatch
sbatch sbatch/extract_embeddings_imagetoken.sbatch
```

输出路径：`embeddings/emb_alltoken/` 和 `embeddings/emb_imagetoken/`

### 6.2 训练 MLP（一键跑完所有配置）

一次性跑完 12 种配置（2 embedding × 2 arch × 3 loss）：

```bash
sbatch sbatch/train_mlp_all.sbatch
```

配置列表：

| Embedding | Arch | Loss |
|-----------|------|------|
| alltoken / imagetoken | residual / plain | ciou / smoothl1 / iou |

输出：`output/train/train_<arch>_<loss>_<embedding>/`

### 6.3 评估 MLP（一键评估所有配置）

训练完成后一键评估所有 12 个模型：

```bash
sbatch sbatch/eval_mlp_all.sbatch
```

### 6.4 评估 VLM Baseline（Zero-shot + Few-shot）

```bash
# 同时运行 zero-shot 和 few-shot (n=3)
sbatch sbatch/run_eval_vlm_baselines.sbatch

# 或单独运行
sbatch sbatch/eval_vlm_zeroshot.sbatch
sbatch sbatch/eval_vlm_fewshot3.sbatch
```

> 默认使用 `embeddings/emb_alltoken`（已提取的 embedding）。如需更换，运行时覆盖 `EMBEDDING_DIR` 环境变量。

---

## 7. 输出目录

```text
embeddings/emb_alltoken/
  train.pt / val.pt / test.pt / meta.json / config.json / split_stats.json

embeddings/emb_imagetoken/
  train.pt / val.pt / test.pt / meta.json / config.json / split_stats.json

output/train/train_<arch>_<loss>_<embedding>/
  bbox_mlp.pt / train_result.json

output/eval/eval_<arch>_<loss>_<embedding>/
  bbox_eval.json

output/vlm/
  vlm_zeroshot / vlm_fewshot3
```

---

## 8. 详细实现文档

- `project_implementation_details.md`
- `sbatch/README.md`
