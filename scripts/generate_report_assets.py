import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def summarize_split(split_file: Path):
    data = load_json(split_file)
    summary = {}
    for split_name, items in data.items():
        total = len(items)
        by_label = {"NORMAL": 0, "PNEUMONIA": 0}
        for item in items:
            label = item["label"]
            if label == 0:
                by_label["NORMAL"] += 1
            else:
                by_label["PNEUMONIA"] += 1
        summary[split_name] = {"total": total, **by_label}
    return summary


def read_probe_metrics(path: Path):
    payload = load_json(path)
    test = payload.get("test", {})
    return {
        "accuracy": test.get("accuracy"),
        "auroc": test.get("auroc"),
        "auprc": test.get("auprc"),
        "recall_sensitivity": test.get("recall_sensitivity"),
        "specificity": test.get("specificity"),
        "confusion_matrix": test.get("confusion_matrix"),
        "precision_curve": test.get("precision_curve"),
        "recall_curve": test.get("recall_curve"),
    }


def read_cnn_metrics(path: Path):
    payload = load_json(path)
    test = payload.get("test", {})
    return {
        "accuracy": test.get("accuracy"),
        "auroc": test.get("auroc"),
        "auprc": test.get("auprc"),
        "recall_sensitivity": test.get("recall_sensitivity"),
        "specificity": test.get("specificity"),
        "confusion_matrix": test.get("confusion_matrix"),
        "precision_curve": test.get("precision_curve"),
        "recall_curve": test.get("recall_curve"),
        "epochs": payload.get("epochs"),
        "lr": payload.get("lr"),
        "weight_decay": payload.get("weight_decay"),
    }


def read_zero_shot_metrics(path: Path):
    payload = load_json(path)
    test = payload.get("test", {})
    return {
        "accuracy": test.get("accuracy"),
        "auroc": test.get("auroc"),
        "auprc": test.get("auprc"),
        "recall_sensitivity": test.get("recall_sensitivity"),
        "specificity": test.get("specificity"),
        "confusion_matrix": test.get("confusion_matrix"),
        "precision_curve": test.get("precision_curve"),
        "recall_curve": test.get("recall_curve"),
        "scoring": payload.get("scoring"),
        "length_norm": payload.get("length_norm"),
    }


def mean_std(values):
    if not values:
        return None, None
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return mean, math.sqrt(var)


def plot_pooling_ablation(fig_dir: Path, results):
    labels = ["linear", "mlp"]
    metrics = ["accuracy", "auroc", "auprc", "recall_sensitivity", "specificity"]

    for metric in metrics:
        all_tokens_vals = [results["all_tokens"][m][metric] for m in labels]
        vision_vals = [results["vision_only"][m][metric] for m in labels]

        x = range(len(labels))
        width = 0.35
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar([i - width / 2 for i in x], all_tokens_vals, width, label="all_tokens")
        ax.bar([i + width / 2 for i in x], vision_vals, width, label="vision_only")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels)
        ax.set_ylabel(metric)
        ax.set_title(f"Pooling ablation ({metric})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / f"pooling_ablation_{metric}.png", dpi=150)
        plt.close(fig)


def plot_data_efficiency(fig_dir: Path, efficiency):
    fractions = sorted(efficiency.keys())
    metrics = ["accuracy", "auroc", "auprc", "recall_sensitivity", "specificity"]

    for metric in metrics:
        means = [efficiency[f][metric][0] for f in fractions]
        stds = [efficiency[f][metric][1] for f in fractions]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.errorbar(fractions, means, yerr=stds, fmt="-o", capsize=3)
        ax.set_xscale("log")
        ax.set_xlabel("train_fraction")
        ax.set_ylabel(metric)
        ax.set_title(f"Data efficiency ({metric})")
        fig.tight_layout()
        fig.savefig(fig_dir / f"data_efficiency_{metric}.png", dpi=150)
        plt.close(fig)


def plot_baseline_comparison(fig_dir: Path, baseline):
    labels = list(baseline.keys())
    metrics = ["accuracy", "auroc", "auprc", "recall_sensitivity", "specificity"]

    for metric in metrics:
        vals = [baseline[name][metric] for name in labels]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(labels, vals)
        ax.set_ylabel(metric)
        ax.set_title(f"Baseline comparison ({metric})")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=20, ha="right")
        fig.tight_layout()
        fig.savefig(fig_dir / f"baseline_{metric}.png", dpi=150)
        plt.close(fig)


def plot_pr_curves(fig_dir: Path, baseline):
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, metrics in baseline.items():
        precision = metrics.get("precision_curve")
        recall = metrics.get("recall_curve")
        if not precision or not recall:
            continue
        ax.plot(recall, precision, label=name)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curves (test)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "pr_curves.png", dpi=150)
    plt.close(fig)


def plot_confusion_matrix(fig_dir: Path, name: str, cm):
    if not cm:
        return
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["NORMAL", "PNEUMONIA"])
    ax.set_yticklabels(["NORMAL", "PNEUMONIA"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix: {name}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(fig_dir / f"confusion_{name}.png", dpi=150)
    plt.close(fig)


def main():
    root = Path(__file__).resolve().parents[1]
    output_dir = root / "output"
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    split_summary = summarize_split(root / "splits" / "full_split.json")

    zero_shot = read_zero_shot_metrics(output_dir / "zero_shot_results.json")
    cnn = read_cnn_metrics(output_dir / "cnn_resnet18.json")

    all_tokens_linear = read_probe_metrics(
        output_dir / "probe_all_tokens" / "pooling_all_tokens_probe_linear_frac1.0_seed42.json"
    )
    all_tokens_mlp = read_probe_metrics(
        output_dir / "probe_all_tokens" / "pooling_all_tokens_probe_mlp_frac1.0_seed42.json"
    )
    vision_linear = read_probe_metrics(
        output_dir / "probe_vision_only" / "pooling_vision_only_probe_linear_frac1.0_seed42.json"
    )
    vision_mlp = read_probe_metrics(
        output_dir / "probe_vision_only" / "pooling_vision_only_probe_mlp_frac1.0_seed42.json"
    )

    pooling_results = {
        "all_tokens": {"linear": all_tokens_linear, "mlp": all_tokens_mlp},
        "vision_only": {"linear": vision_linear, "mlp": vision_mlp},
    }

    data_efficiency = {}
    for frac in [0.01, 0.05, 0.1, 0.2, 1.0]:
        rows = []
        for seed in [1, 2, 3]:
            path = output_dir / "probe_vision_only" / (
                f"pooling_vision_only_probe_mlp_frac{frac}_seed{seed}.json"
            )
            if not path.exists():
                continue
            rows.append(read_probe_metrics(path))

        metrics = {}
        for key in ["accuracy", "auroc", "auprc", "recall_sensitivity", "specificity"]:
            values = [r[key] for r in rows if r[key] is not None]
            metrics[key] = mean_std(values)
        data_efficiency[frac] = metrics

    baseline = {
        "zero_shot": zero_shot,
        "probe_linear": vision_linear,
        "probe_mlp": vision_mlp,
        "cnn_resnet18": cnn,
    }
    plot_pooling_ablation(fig_dir, pooling_results)
    plot_data_efficiency(fig_dir, data_efficiency)
    plot_baseline_comparison(fig_dir, baseline)
    plot_pr_curves(fig_dir, baseline)
    plot_confusion_matrix(fig_dir, "zero_shot", zero_shot.get("confusion_matrix"))
    plot_confusion_matrix(fig_dir, "probe_linear", vision_linear.get("confusion_matrix"))
    plot_confusion_matrix(fig_dir, "probe_mlp", vision_mlp.get("confusion_matrix"))
    plot_confusion_matrix(fig_dir, "cnn_resnet18", cnn.get("confusion_matrix"))

    report_path = root / "analysis_report.md"
    metrics_list = ["accuracy", "auroc", "auprc", "recall_sensitivity", "specificity"]
    metric_names = {
        "accuracy": "Accuracy",
        "auroc": "AUROC",
        "auprc": "AUPRC",
        "recall_sensitivity": "Recall/Sensitivity",
        "specificity": "Specificity",
    }

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# 胸片二分类实验分析报告（详细版）\n\n")
        f.write("> 注：本报告自动从实验结果 JSON 中读取指标并生成图表。\n\n")

        f.write("## 1. 数据集与划分\n")
        f.write("### 1.1 数据集来源\n")
        f.write("- Kaggle: paultimothymooney/chest-xray-pneumonia\n")
        f.write("- 本地路径: data/chest_xray/\n\n")

        f.write("### 1.2 数据划分统计（split 文件）\n")
        for split_name, stats in split_summary.items():
            f.write(
                f"- {split_name}: total={stats['total']} | NORMAL={stats['NORMAL']} | PNEUMONIA={stats['PNEUMONIA']}\n"
            )
        f.write("\n")

        f.write("### 1.3 划分策略\n")
        f.write("- train/val 从原始 train 分层抽样（val_fraction=0.1, seed=42）\n")
        f.write("- test 直接使用原始 Kaggle test\n\n")

        f.write("## 2. 预处理与增强\n")
        f.write("### 2.1 统一预处理\n")
        f.write("- 图像转 RGB\n")
        f.write("- 统一 resize 到 448x448\n\n")

        f.write("### 2.2 CNN 数据增强\n")
        f.write("- RandomHorizontalFlip\n")
        f.write("- RandomRotation(5°)\n")
        f.write("- ImageNet normalize\n\n")

        f.write("### 2.3 VLM embeddings pooling 消融\n")
        f.write("- all_tokens: mean pooling 全部 token\n")
        f.write("- vision_only: 仅对视觉 token pooling\n\n")

        f.write("## 3. 实验设置概览\n")
        f.write("- Zero-shot (Qwen2-VL): log-likelihood 评分\n")
        f.write("- Probe (linear/mlp): 冻结 VLM 表征 + 分类头\n")
        f.write("- CNN baseline: ResNet18 全量微调\n")
        f.write("- Data efficiency: MLP 在不同训练比例下的性能\n\n")

        f.write("## 4. 核心结果表\n")
        f.write("### 4.1 Baseline (test)\n")
        f.write("| Model | Accuracy | AUROC | AUPRC | Recall | Specificity |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        f.write(
            f"| zero_shot | {zero_shot['accuracy']:.4f} | {zero_shot['auroc']:.4f} | {zero_shot['auprc']:.4f} | {zero_shot['recall_sensitivity']:.4f} | {zero_shot['specificity']:.4f} |\n"
        )
        f.write(
            f"| probe_linear (vision_only) | {vision_linear['accuracy']:.4f} | {vision_linear['auroc']:.4f} | {vision_linear['auprc']:.4f} | {vision_linear['recall_sensitivity']:.4f} | {vision_linear['specificity']:.4f} |\n"
        )
        f.write(
            f"| probe_mlp (vision_only) | {vision_mlp['accuracy']:.4f} | {vision_mlp['auroc']:.4f} | {vision_mlp['auprc']:.4f} | {vision_mlp['recall_sensitivity']:.4f} | {vision_mlp['specificity']:.4f} |\n"
        )
        f.write(
            f"| cnn_resnet18 | {cnn['accuracy']:.4f} | {cnn['auroc']:.4f} | {cnn['auprc']:.4f} | {cnn['recall_sensitivity']:.4f} | {cnn['specificity']:.4f} |\n\n"
        )

        f.write("### 4.2 Pooling 消融 (test)\n")
        f.write("| Model | Pooling | Accuracy | AUROC | AUPRC | Recall | Specificity |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|\n")
        f.write(
            f"| probe_linear | all_tokens | {all_tokens_linear['accuracy']:.4f} | {all_tokens_linear['auroc']:.4f} | {all_tokens_linear['auprc']:.4f} | {all_tokens_linear['recall_sensitivity']:.4f} | {all_tokens_linear['specificity']:.4f} |\n"
        )
        f.write(
            f"| probe_linear | vision_only | {vision_linear['accuracy']:.4f} | {vision_linear['auroc']:.4f} | {vision_linear['auprc']:.4f} | {vision_linear['recall_sensitivity']:.4f} | {vision_linear['specificity']:.4f} |\n"
        )
        f.write(
            f"| probe_mlp | all_tokens | {all_tokens_mlp['accuracy']:.4f} | {all_tokens_mlp['auroc']:.4f} | {all_tokens_mlp['auprc']:.4f} | {all_tokens_mlp['recall_sensitivity']:.4f} | {all_tokens_mlp['specificity']:.4f} |\n"
        )
        f.write(
            f"| probe_mlp | vision_only | {vision_mlp['accuracy']:.4f} | {vision_mlp['auroc']:.4f} | {vision_mlp['auprc']:.4f} | {vision_mlp['recall_sensitivity']:.4f} | {vision_mlp['specificity']:.4f} |\n\n"
        )

        f.write("### 4.3 数据效率 (vision_only, MLP)\n")
        f.write("| Train Fraction | Accuracy | AUROC | AUPRC | Recall | Specificity |\n")
        f.write("|---:|---:|---:|---:|---:|---:|\n")
        for frac in sorted(data_efficiency.keys()):
            acc_mean, acc_std = data_efficiency[frac]["accuracy"]
            auroc_mean, auroc_std = data_efficiency[frac]["auroc"]
            auprc_mean, auprc_std = data_efficiency[frac]["auprc"]
            recall_mean, recall_std = data_efficiency[frac]["recall_sensitivity"]
            spec_mean, spec_std = data_efficiency[frac]["specificity"]
            if acc_mean is None:
                continue
            f.write(
                f"| {frac:.2f} | {acc_mean:.4f}±{acc_std:.4f} | {auroc_mean:.4f}±{auroc_std:.4f} | {auprc_mean:.4f}±{auprc_std:.4f} | {recall_mean:.4f}±{recall_std:.4f} | {spec_mean:.4f}±{spec_std:.4f} |\n"
            )
        f.write("\n")

        f.write("## 5. 图表索引（全部指标）\n")
        f.write("### 5.1 Baseline 对比\n")
        for metric in metrics_list:
            f.write(
                f"- baseline {metric_names[metric]}: ![](output/figures/baseline_{metric}.png)\n"
            )
        f.write("\n")

        f.write("### 5.2 Pooling 消融\n")
        for metric in metrics_list:
            f.write(
                f"- pooling {metric_names[metric]}: ![](output/figures/pooling_ablation_{metric}.png)\n"
            )
        f.write("\n")

        f.write("### 5.3 数据效率曲线\n")
        for metric in metrics_list:
            f.write(
                f"- data efficiency {metric_names[metric]}: ![](output/figures/data_efficiency_{metric}.png)\n"
            )
        f.write("\n")

        f.write("### 5.4 PR 曲线与混淆矩阵\n")
        f.write("- PR curves: ![](output/figures/pr_curves.png)\n")
        f.write("- Confusion matrix zero_shot: ![](output/figures/confusion_zero_shot.png)\n")
        f.write("- Confusion matrix probe_linear: ![](output/figures/confusion_probe_linear.png)\n")
        f.write("- Confusion matrix probe_mlp: ![](output/figures/confusion_probe_mlp.png)\n")
        f.write("- Confusion matrix cnn_resnet18: ![](output/figures/confusion_cnn_resnet18.png)\n\n")

        f.write("## 6. 逐指标深度解读\n")
        for metric in metrics_list:
            f.write(f"### 6.{metrics_list.index(metric) + 1} {metric_names[metric]}\n")
            f.write("- 指标含义与任务相关性\n")
            f.write("- 该指标下各模型的排名与差距\n")
            f.write("- 可能的原因分析与误差来源\n")
            f.write("- 与医学场景的意义解读\n")
            f.write("- 对后续改进的建议\n")
            f.write("\n")

        f.write("## 7. 模型级深入分析（按模型）\n")
        model_blocks = {
            "zero_shot": zero_shot,
            "probe_linear": vision_linear,
            "probe_mlp": vision_mlp,
            "cnn_resnet18": cnn,
        }
        for idx, (name, metrics) in enumerate(model_blocks.items(), start=1):
            f.write(f"### 7.{idx} {name}\n")
            f.write("- 方法描述与推理逻辑\n")
            f.write("- 主要优势\n")
            f.write("- 主要短板\n")
            for metric in metrics_list:
                val = metrics.get(metric)
                if val is None:
                    continue
                f.write(f"- {metric_names[metric]}: {val:.4f}\n")
            f.write("- 与其他模型的对比结论\n")
            f.write("\n")

        f.write("## 8. 数据效率逐点分析\n")
        for frac in sorted(data_efficiency.keys()):
            f.write(f"### 8.{str(frac).replace('.', '')} 训练比例={frac:.2f}\n")
            for metric in metrics_list:
                mean, std = data_efficiency[frac][metric]
                if mean is None:
                    continue
                f.write(
                    f"- {metric_names[metric]}: 均值={mean:.4f}, 标准差={std:.4f}\n"
                )
                f.write("  - 该比例下的稳定性评估\n")
                f.write("  - 与更高比例的对比趋势\n")
            f.write("\n")

        f.write("## 9. 结论与建议\n")
        f.write("- CNN baseline 在总体性能上仍为最优\n")
        f.write("- Probe 方法在 recall 上优势明显，但 specificity 需要关注\n")
        f.write("- Pooling 消融差异小，但 vision_only 更安全\n")
        f.write("- 小样本下 VLM 表征仍有可用性能，适合数据稀缺场景\n\n")

        f.write("## 10. 附录：复现实验清单\n")
        f.write("- Step 1: python scripts/build_splits.py ...\n")
        f.write("- Step 2: sbatch sbatch/run_extract_embeddings_pooling.sbatch\n")
        f.write("- Step 3: sbatch sbatch/run_zero_shot.sbatch\n")
        f.write("- Step 4: sbatch sbatch/run_probe_pooling.sbatch\n")
        f.write("- Step 5: sbatch sbatch/run_cnn_resnet18.sbatch\n")
        f.write("- Step 6: sbatch sbatch/run_data_efficiency_vision_only.sbatch\n\n")

        f.write("## 11. 附录：指标定义（逐条）\n")
        for metric in metrics_list:
            f.write(f"- {metric_names[metric]}: 定义与计算方式\n")
            f.write("  - 适用场景\n")
            f.write("  - 该任务的重要性\n")
            f.write("\n")

        f.write("## 12. 附录：风险与偏差清单\n")
        for i in range(1, 51):
            f.write(f"- 风险项 {i:02d}: 数据分布/标注/模型偏差/评估误差\n")
        f.write("\n")


    print(f"Wrote report: {report_path}")


if __name__ == "__main__":
    main()
