# Medical Image Analysis Final Project 仓库分析与后续实验计划

## 1. 这个代码仓库在做什么？技术栈是什么？

这个仓库在做 **胸部X光病灶框定位（bounding box localization）**：  
用 **Qwen2-VL** 提取图像嵌入，再训练一个轻量 **MLP** 回归框坐标 `(x, y, w, h)`，并和两种 VLM baseline 对比：

1. **Zero-shot VLM**（直接提示词输出框）
2. **Few-shot VLM**（提示词中加入少量标注示例，当前是 n=3）
3. （扩展）VLM embedding + MLP（plain/residual）

**主要技术栈**

- 语言/框架：Python, PyTorch
- 大模型：Qwen2-VL (`transformers`)
- 数据处理：PIL, csv, json
- 训练评估：torch DataLoader, SmoothL1Loss, IoU指标
- 可视化：matplotlib
- 数据下载：kagglehub（`nih-chest-xrays/data`）

---

## 2. 目录结构、职责与核心文件

```text
.
├─ README.md                        # 项目说明、结果汇总、运行流程
├─ requirements.txt                 # 依赖
├─ data/
│  └─ download.py                   # 从 Kaggle 下载 NIH ChestX-ray 数据
├─ embeddings/
│  ├─ bbox_train.pt                 # 训练集 embeddings + bbox + labels
│  ├─ bbox_test.pt                  # 测试集 embeddings + bbox + labels
│  ├─ bbox_meta.json                # 样本元信息（路径、类别、坐标）
│  ├─ class_names.json              # 类别与id映射
│  └─ class_prompts.json            # 类别提示词
├─ scripts/
│  ├─ extract_bbox_embeddings.py    # 采样并提取Qwen embedding
│  ├─ train_bbox_mlp.py             # 训练MLP回归头
│  ├─ evaluate_bbox.py              # 评估MLP并导出可视化
│  ├─ evaluate_few_shot.py          # zero/few-shot VLM baseline
│  └─ compare_results.py            # 汇总对比各方法
└─ output/
   ├─ zero_shot_eval.json
   ├─ few_shot_3_eval.json
   ├─ plain/..., residual/...       # MLP两种变体结果与模型
   └─ comparison.json               # 四方法汇总结果
```

---

## 3. 当前代码实现细节（核心流程）

### 3.1 Embedding提取（`extract_bbox_embeddings.py`）

- 目标类：`Atelectasis`, `Effusion`, `Cardiomegaly`
- 每类采样：`30 train + 10 test`（固定常量）
- 用类别专属提示词与图像一起送入 Qwen2-VL
- 取最后层 hidden state 做 mean pooling 得到 embedding
- 输出：
  - `bbox_train.pt` / `bbox_test.pt`
  - `bbox_meta.json`（样本路径、类别、bbox）
  - 类别和提示词json

### 3.2 MLP训练（`train_bbox_mlp.py`）

- 输入：冻结后的 embedding
- 输出：4维bbox回归
- 损失：`SmoothL1Loss`
- 优化器：Adam (`lr=1e-3`, `weight_decay=1e-4`)
- 学习率调度：CosineAnnealingLR
- 默认超参：`epochs=100, batch_size=16, hidden_dim=256, dropout=0.2`
- 可选 `--use_residual` 开关

### 3.3 MLP评估（`evaluate_bbox.py`）

- 读取 `bbox_test.pt` 和模型权重
- 计算总体与分类别 IoU / IoU@0.25 / IoU@0.5
- 导出 `bbox_eval.json` 与可视化图（best/worst样本）

### 3.4 Zero/Few-shot baseline（`evaluate_few_shot.py`）

- `--n_shot 0` = zero-shot
- `--n_shot 3` = few-shot（提示词中追加每类训练样本框坐标示例）
- 生成文本后用正则解析 bbox，计算 IoU
- 导出 `zero_shot_eval.json` 或 `few_shot_3_eval.json`

---

## 4. 数据集分析

### 4.1 数据来源

- `data/download.py` 指向 Kaggle 数据：`nih-chest-xrays/data`
- 该项目使用了其 **bounding box 标注表**（代码默认 `./data/3/BBox_List_2017.csv`）和对应图像

### 4.2 字段与使用方式

从代码看，读取了CSV前6列：

1. 图像文件名（`row[0]`）
2. 疾病标签（`row[1]`）
3. `x`（`row[2]`）
4. `y`（`row[3]`）
5. `w`（`row[4]`）
6. `h`（`row[5]`）

仅筛选三类病灶；每个样本由「图像 + 类别prompt + bbox标签」组成。

### 4.3 当前数据量

- 从 `bbox_meta.json` 统计到三类各40条，共 **120** 条（Atelectasis/Effusion/Cardiomegaly 各40）
- 按脚本固定切分为：每类 **30 train + 10 test**
- 总计：**90训练 + 30测试**

---

## 5. 已有实验、配置、结果与问题分析

### 5.1 已做实验组

1. Qwen Zero-shot (`n_shot=0`)
2. Qwen Few-shot (`n_shot=3`)
3. MLP（plain）
4. MLP + Residual

### 5.2 结果（以 `output/comparison.json` 为准）

| Method | Mean IoU | IoU@0.25 | IoU@0.5 |
|---|---:|---:|---:|
| MLP | 0.047 | 0.067 | 0.000 |
| MLP+Residual | 0.219 | 0.433 | 0.100 |
| Qwen Zero-shot | 0.070 | 0.167 | 0.000 |
| Qwen Few-shot (n=3) | **0.254** | **0.467** | **0.167** |

结论：当前最好的是 **few-shot VLM**，VLM+MLP 还没有超过 few-shot。

### 5.3 结果说服力评估

当前结论说服力有限，主要因为：

- 测试集仅30张，统计波动很大
- 仅一次随机种子（seed=42），没有重复试验均值/方差
- 没有交叉验证或多次重采样
- 类别难度差异明显（Cardiomegaly显著更容易）

### 5.4 当前实验设置中的关键问题（需要优先修正）

1. **潜在数据泄漏风险**：训练脚本每轮在 test set 上评估并按 test IoU 选 best checkpoint，这会导致测试集被用于模型选择。  
2. **实现不一致风险**：`train_bbox_mlp.py` 与 `evaluate_bbox.py` 对 “plain” 结构定义不完全一致，可能造成训练-评估结构不对齐。  
3. **结果文件不一致**：`output/plain/results.json` 与 `output/plain/bbox_eval.json` 指标明显冲突，说明结果管理/复现实验流程存在问题。  
4. **小数据下回归塌缩**：从 per-sample 可见模型常输出接近固定框，说明当前 head 与训练策略对小样本不够稳健。  

---

## 6. 下一步指导：围绕“VLM+MLP显著超过 zero/few-shot”如何做

你的目标是：**baseline只保留 zero-shot 和 few-shot，重点把 VLM+MLP 做到明显更好**。  
建议把主指标目标定为：`Mean IoU > 0.35` 且稳定超过 few-shot（当前0.254）。

### 6.1 先固定对照组（必须统一评估）

- **B0**: Zero-shot (`n_shot=0`)
- **B1**: Few-shot (`n_shot=3`)
- **M0**: 当前 VLM+MLP 基线（修正实现一致性后重跑）

### 6.2 VLM+MLP实验组设计（建议顺序）

1. **G1 数据扩容组（优先级最高）**  
   - 每类从 30/10 增到 100/30、300/100 两档  
   - 保持模型不变，验证性能是否受数据瓶颈限制

2. **G2 目标归一化 + 损失改进组**  
   - 把 bbox 坐标按图像尺寸归一化到 [0,1]  
   - 损失改为 `L = SmoothL1 + λ*(1-IoU)`（如 λ=1.0）

3. **G3 Prompt-Ensemble Embedding 组**  
   - 每类使用3~5条语义不同 prompt 提 embedding 并平均/拼接  
   - 目标是增强 VLM embedding 的鲁棒性和可分性

4. **G4 更强回归头组**  
   - 尝试 MLP + residual + LayerNorm + GELU  
   - 或 class-aware head（输入拼接类别embedding/one-hot）

5. **G5 稳健训练组**  
   - K-fold 或 3次不同seed重复实验  
   - 报告均值±标准差，提高说服力

### 6.3 推荐最小可执行实验矩阵（你当前阶段）

| 组别 | 目的 | 关键变量 |
|---|---|---|
| B0 | 基线 | n_shot=0 |
| B1 | 基线 | n_shot=3 |
| M0 | 当前MLP基线 | 30/10每类，现有超参 |
| M1 | 验证数据量贡献 | 100/30每类 |
| M2 | 验证损失设计 | M1 + IoU联合损失 |
| M3 | 验证表示增强 | M2 + prompt ensemble |

如果你只做一轮冲刺，优先顺序：**M1 > M2 > M3**。  
原因：当前最大瓶颈首先是样本量和训练目标设计，不是网络深度本身。

---

## 7. 交付建议（课程汇报可直接使用）

- 报告中明确 baseline 仅有两项：zero-shot/few-shot
- 强调你的贡献点：**VLM embedding + MLP 的系统优化**（数据、损失、表示、稳健性）
- 结果展示必须包含：
  - 总体指标（Mean IoU, IoU@0.25, IoU@0.5）
  - 分类别指标
  - 可视化成功/失败案例
  - 多seed均值±方差（至少3次）

