# 下一步实验计划（按你的要求更新）

## 0. 目标

在固定大规模训练数据下，完成最关键对比，验证 **VLM+MLP** 明显超过 **Zero-shot / Few-shot VLM**。

---

## 1. 全局固定设置（所有实验组一致）

| 项 | 固定值 |
|---|---|
| 坐标表示 | **必须做 box 归一化**：`(x/W, y/H, w/W, h/H)` |
| 数据规模 | **直接使用大规模训练**（不再做数据规模对比实验） |
| 数据划分 | 按 `patient_id` 划分，`train/val/test = 70/15/15` |
| 评估策略 | test 仅用于最终评估，不用于选模型 |
| 随机性 | 每组 3 seeds，汇报 `mean ± std` |
| 主指标 | Mean IoU, IoU@0.25, IoU@0.5 |

---

## 2. 大规模数据配置（固定）

参考公开统计（图像级标签）：
- Atelectasis: 11,559
- Effusion: 13,317
- Cardiomegaly: 2,776

考虑 Cardiomegaly 上限，固定训练规模为：
- **Large**：每类 2,200（总计 6,600）

验证集与测试集使用固定 bbox 金标准集。

---

## 3. 实验组总览（每个对比维度完整列出）

## 3.1 Baseline 维度（2组）

| 组ID | 组名 | 设置 |
|---|---|---|
| B0 | Zero-shot | Qwen2-VL，`n_shot=0` |
| B1 | Few-shot | Qwen2-VL，`n_shot=3` |

## 3.2 Embedding 对比维度（2组）

> 固定：MLP 使用 `A1`，Loss 使用 `L1`

| 组ID | 组名 | Embedding 设置 |
|---|---|---|
| E1 | AllTokenMean | All token mean pooling |
| E2 | ImageTokenMean | Image token only mean pooling |

## 3.3 MLP 架构对比维度（2组）

> 固定：Embedding 使用 `E*`（E1/E2中更优者），Loss 使用 `L1`

| 组ID | 组名 | MLP 设置 |
|---|---|---|
| A1 | MLP-Plain | 原始 MLP |
| A2 | MLP-Residual | MLP + Residual |

## 3.4 Loss 对比维度（3组）

> 固定：Embedding 使用 `E*`，MLP 使用 `A*`（A1/A2中更优者）

| 组ID | 组名 | Loss 设置 |
|---|---|---|
| L1 | SmoothL1 | SmoothL1（在归一化坐标上计算） |
| L2 | IoU | IoU Loss |
| L3 | CIoU | CIoU Loss |

## 3.5 最终模型（1组）

| 组ID | 组名 | 设置 |
|---|---|---|
| F0 | Final-Best | `Large + E* + A* + L*`，与 B0/B1 直接对比 |

---

## 4. 总实验数量

- Baseline: 2 组
- Embedding: 2 组
- MLP 架构: 2 组
- Loss: 3 组
- Final: 1 组

**合计：10 组**

---

## 5. 执行顺序

1. 跑 baseline：`B0, B1`
2. 跑 embedding：`E1, E2`，确定 `E*`
3. 跑 MLP 架构：`A1, A2`，确定 `A*`
4. 跑 loss：`L1, L2, L3`，确定 `L*`
5. 训练并评估最终模型：`F0`，与 `B0/B1` 同表对比

---

## 6. 成功判据

1. `F0 Mean IoU` 相对 `B1` 提升 ≥ 20%
2. `F0 IoU@0.5` 相对 `B1` 提升 ≥ 10 个百分点
3. 3 seeds 下提升方向一致

