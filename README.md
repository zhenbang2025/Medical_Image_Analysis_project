# Medical Image Analysis: Chest X-Ray 分类

使用 Qwen2-VL 与标准视觉基线完成胸片二分类（NORMAL vs PNEUMONIA）。本项目实现：

1. **Zero-shot (Qwen2-VL)** — 候选答案 log-likelihood 评分
2. **VLM embeddings + probe** — 线性探针与 MLP
3. **CNN baseline** — ResNet/DenseNet 全量微调
4. **数据效率 + 鲁棒性** — 可选评估维度

## 环境配置

```bash
pip install -r requirements.txt
```

### 远程服务器（H800）推荐流程

1. 创建并激活 conda 环境（Python 3.10）

```bash
conda create -n mia python=3.10 -y
conda activate mia
```

2. 安装依赖

```bash
pip install -r requirements.txt
```

3. 验证环境

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

4. sbatch 使用说明

- 先修改每个 sbatch 文件里的 `PROJECT_ROOT` 和 `ENV_PATH`
- 提交方式：`sbatch sbatch/<task_name>.sbatch`
- 任务日志输出在 `logs/`

### 依赖

完整依赖见 [requirements.txt](requirements.txt)。

## 数据集（全量默认）

从 Kaggle 下载 Chest X-Ray Pneumonia 数据集：

```bash
python data/download.py
```

数据集保存在 `data/chest_xray/`：

```
data/chest_xray/
├── train/NORMAL/       (1,341 images)
├── train/PNEUMONIA/    (3,875 images)
├── val/NORMAL/         (8 images)
├── val/PNEUMONIA/      (8 images)
├── test/NORMAL/        (234 images)
└── test/PNEUMONIA/     (390 images)
```

小数据集验证已移除，所有脚本默认以全量数据为准。

## 模型

### Qwen2-VL-2B-Instruct

下载 Qwen2-VL-2B-Instruct（本项目主模型）：

```bash
# Option 1: HuggingFace
huggingface-cli download Qwen/Qwen2-VL-2B-Instruct --local-dir models/Qwen2-VL-2B-Instruct

# Option 2: ModelScope
modelscope download --model Qwen/Qwen2-VL-2B-Instruct --local_dir models/Qwen2-VL-2B-Instruct
```

- 模型规模：2B
- 磁盘占用（参考）：约 4-6 GB（fp16 权重，具体以仓库文件大小为准）
- 安装路径建议：`models/Qwen2-VL-2B-Instruct`

### CNN baseline（ResNet/DenseNet）

- 由 `torchvision` 自动下载预训练权重
- 权重缓存路径：`~/.cache/torch/hub`
- 磁盘占用：通常在数百 MB 量级

### Medical CLIP baseline（可选）

- 当前脚本是可插拔 stub，选择仓库后再实现
- 若启用，请在 README 和脚本内写清权重路径与大小

## 目录结构

```
Medical_Image_Analysis_project/
├── data/
├── docs/
│   └── PROJECT_DETAILS.md
├── embeddings/
├── output/
├── sbatch/
│   └── run_all_experiments.sbatch
├── scripts/
│   ├── build_splits.py
│   ├── evaluate_zero_shot.py
│   ├── extract_embeddings.py
│   ├── train_probe.py
│   ├── train_cnn.py
│   ├── robustness_eval.py
│   ├── run_all_experiments.sh
│   └── medclip_baseline.py
└── README.md
```

## 使用说明（GPU 优先）

凡涉及训练或推理的任务一律用 sbatch（GPU），仅数据 split 构建使用 CPU。

### Step 1: 构建确定性数据分割

```bash
python scripts/build_splits.py \
	--dataset_dir ./data/chest_xray \
	--val_fraction 0.1 \
	--seed 42 \
	--output_file ./splits/full_split.json
```

### Step 2: 提取 embeddings（Pooling 消融）

```bash
sbatch sbatch/run_extract_embeddings_pooling.sbatch
```

### Step 3: Zero-shot baseline

```bash
sbatch sbatch/run_zero_shot.sbatch
```

### Step 4: 线性探针 + MLP（Pooling 消融）

```bash
sbatch sbatch/run_probe_pooling.sbatch
```

### Step 5: CNN baseline

```bash
sbatch sbatch/run_cnn_resnet18.sbatch
```

### Step 6: 数据效率扫描（可选）

```bash
sbatch sbatch/run_data_efficiency_vision_only.sbatch
```

### Step 7: 鲁棒性评估（可选）

```bash
sbatch sbatch/run_robustness_resnet18.sbatch
```

### 可选：一键跑完整实验

```bash
sbatch sbatch/run_all_experiments.sbatch
```

该脚本包含 pooling 消融（`vision_only` vs `all_tokens`），会自动将结果写入不同输出目录。

## 常用参数

| Argument | Description | Default |
|---|---|---|
| `--model` | Model path | `./models/Qwen2-VL-2B-Instruct` |
| `--dataset_dir` | Dataset path | `./data/chest_xray` |
| `--split_file` | Split JSON file | empty (disabled) |
| `--device` | `auto`, `mps`, `cuda`, or `cpu` | `auto` |
| `--batch_size` | Images per batch | script-specific |

## 结果说明

所有最终结论请使用全量数据。

## 输出文件

```
output/
├── zero_shot_results.json
├── cnn_resnet18.json
├── cnn_resnet18.pt
├── robustness_results.json
├── probe_all_tokens/
│   ├── pooling_all_tokens_probe_linear_*.json
│   ├── pooling_all_tokens_probe_mlp_*.json
│   └── pooling_all_tokens_probe_*.pt
└── probe_vision_only/
	├── pooling_vision_only_probe_linear_*.json
	├── pooling_vision_only_probe_mlp_*.json
	└── pooling_vision_only_probe_*.pt

embeddings/
├── all_tokens/
│   ├── train.pt
│   ├── val.pt
│   └── test.pt
└── vision_only/
	├── train.pt
	├── val.pt
	└── test.pt
```

## 详细说明文档

详见 [docs/PROJECT_DETAILS.md](docs/PROJECT_DETAILS.md)。

---

## 实验清单与启动命令（sbatch）

- 构建 split 文件: `python scripts/build_splits.py ...`
- 提取 embeddings (pooling 消融): `sbatch sbatch/run_extract_embeddings_pooling.sbatch`
- Zero-shot baseline: `sbatch sbatch/run_zero_shot.sbatch`
- Linear+MLP probe (pooling 消融): `sbatch sbatch/run_probe_pooling.sbatch`
- 数据效率扫描 (vision_only): `sbatch sbatch/run_data_efficiency_vision_only.sbatch`
- CNN baseline (ResNet18): `sbatch sbatch/run_cnn_resnet18.sbatch`
- 鲁棒性评估 (ResNet18): `sbatch sbatch/run_robustness_resnet18.sbatch`
- 一键完整实验: `sbatch sbatch/run_all_experiments.sbatch`

---

## sbatch 任务列表（建议）

- [sbatch/run_build_splits.sbatch](sbatch/run_build_splits.sbatch)
- [sbatch/run_extract_embeddings_all_tokens.sbatch](sbatch/run_extract_embeddings_all_tokens.sbatch)
- [sbatch/run_extract_embeddings_vision_only.sbatch](sbatch/run_extract_embeddings_vision_only.sbatch)
- [sbatch/run_zero_shot.sbatch](sbatch/run_zero_shot.sbatch)
- [sbatch/run_probe_linear_all_tokens.sbatch](sbatch/run_probe_linear_all_tokens.sbatch)
- [sbatch/run_probe_mlp_all_tokens.sbatch](sbatch/run_probe_mlp_all_tokens.sbatch)
- [sbatch/run_probe_linear_vision_only.sbatch](sbatch/run_probe_linear_vision_only.sbatch)
- [sbatch/run_probe_mlp_vision_only.sbatch](sbatch/run_probe_mlp_vision_only.sbatch)
- [sbatch/run_data_efficiency_vision_only.sbatch](sbatch/run_data_efficiency_vision_only.sbatch)
- [sbatch/run_cnn_resnet18.sbatch](sbatch/run_cnn_resnet18.sbatch)
- [sbatch/run_robustness_resnet18.sbatch](sbatch/run_robustness_resnet18.sbatch)
- [sbatch/run_all_experiments.sbatch](sbatch/run_all_experiments.sbatch)
