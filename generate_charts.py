#!/usr/bin/env python3
"""Generate analysis charts for the experiment report."""

import json
import os

# Try to import matplotlib, install if needed
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'matplotlib'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
import numpy as np

# Base path
BASE_PATH = "f:/2-研究生-港科/3-春季学期SpringTerm/ARIN 5302 - Medical Image analysis/final/Medical_Image_Analysis_project"
CHARTS_PATH = os.path.join(BASE_PATH, "output/charts")
os.makedirs(CHARTS_PATH, exist_ok=True)

# Color palette
COLORS = {
    'plain_iou': '#2E86AB',
    'plain_ciou': '#A23B72',
    'plain_smoothl1': '#F18F01',
    'residual_iou': '#C73E1D',
    'residual_ciou': '#3B1F2B',
    'residual_smoothl1': '#44840C',
}

# ============== DATA ==============

# MLP Experiment Results
mlp_results = {
    'plain + IoU': {
        'test_mIoU': 0.2623, 'val_mIoU': 0.2492,
        'test_IoU@0.25': 0.3750, 'test_IoU@0.5': 0.2361,
        'best_epoch': 47
    },
    'plain + CIoU': {
        'test_mIoU': 0.2523, 'val_mIoU': 0.2443,
        'test_IoU@0.25': 0.3472, 'test_IoU@0.5': 0.2222,
        'best_epoch': 67
    },
    'plain + SmoothL1': {
        'test_mIoU': 0.1886, 'val_mIoU': 0.1722,
        'test_IoU@0.25': 0.3333, 'test_IoU@0.5': 0.1111,
        'best_epoch': 11
    },
    'residual + IoU': {
        'test_mIoU': 0.2605, 'val_mIoU': 0.2510,
        'test_IoU@0.25': 0.3750, 'test_IoU@0.5': 0.2083,
        'best_epoch': 51
    },
    'residual + CIoU': {
        'test_mIoU': 0.2568, 'val_mIoU': 0.2451,
        'test_IoU@0.25': 0.3611, 'test_IoU@0.5': 0.2083,
        'best_epoch': 8
    },
    'residual + SmoothL1': {
        'test_mIoU': 0.1865, 'val_mIoU': 0.1692,
        'test_IoU@0.25': 0.3611, 'test_IoU@0.5': 0.0833,
        'best_epoch': 11
    },
}

# Per-class results for best model (plain + IoU)
per_class_best = {
    'Cardiomegaly': {'mIoU': 0.6203, 'IoU@0.25': 1.0, 'IoU@0.5': 0.85},
    'Effusion': {'mIoU': 0.1412, 'IoU@0.25': 0.125, 'IoU@0.5': 0.0},
    'Atelectasis': {'mIoU': 0.1103, 'IoU@0.25': 0.143, 'IoU@0.5': 0.0},
}

# VLM baselines
vlm_results = {
    'Zero-shot': {'mIoU': 0.006, 'IoU@0.25': 0.0, 'IoU@0.5': 0.0},
    'Few-shot (3-shot)': {'mIoU': 0.171, 'IoU@0.25': 0.25, 'IoU@0.5': 0.194},
}

# Load training histories
def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

train_histories = {
    'plain + IoU': load_json(os.path.join(BASE_PATH, 'output/train/train_plain_iou_alltoken/train_result.json'))['history'],
    'plain + CIoU': load_json(os.path.join(BASE_PATH, 'output/train/train_plain_ciou_alltoken/train_result.json'))['history'],
    'plain + SmoothL1': load_json(os.path.join(BASE_PATH, 'output/train/train_plain_smoothl1_alltoken/train_result.json'))['history'],
    'residual + IoU': load_json(os.path.join(BASE_PATH, 'output/train/train_residual_iou_alltoken/train_result.json'))['history'],
    'residual + CIoU': load_json(os.path.join(BASE_PATH, 'output/train/train_residual_ciou_alltoken/train_result.json'))['history'],
    'residual + SmoothL1': load_json(os.path.join(BASE_PATH, 'output/train/train_residual_smoothl1_alltoken/train_result.json'))['history'],
}

# ============== CHART 1: Test mIoU Comparison ==============
def plot_test_miou_comparison():
    fig, ax = plt.subplots(figsize=(12, 6))

    configs = list(mlp_results.keys())
    test_miou = [mlp_results[c]['test_mIoU'] for c in configs]
    val_miou = [mlp_results[c]['val_mIoU'] for c in configs]

    x = np.arange(len(configs))
    width = 0.35

    bars1 = ax.bar(x - width/2, val_miou, width, label='Val mIoU', color='#2E86AB', alpha=0.8)
    bars2 = ax.bar(x + width/2, test_miou, width, label='Test mIoU', color='#F18F01', alpha=0.8)

    ax.set_ylabel('mIoU', fontsize=12)
    ax.set_title('MLP Bbox Regressor: Test mIoU Comparison by Architecture and Loss', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=15, ha='right')
    ax.legend()
    ax.set_ylim(0, 0.35)

    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_PATH, 'test_miou_comparison.png'), dpi=150)
    plt.close()
    print(f"Saved test_miou_comparison.png")

# ============== CHART 2: Per-Class mIoU (Best Model) ==============
def plot_per_class_miou():
    fig, ax = plt.subplots(figsize=(10, 6))

    classes = list(per_class_best.keys())
    miou = [per_class_best[c]['mIoU'] for c in classes]
    iou_025 = [per_class_best[c]['IoU@0.25'] for c in classes]
    iou_05 = [per_class_best[c]['IoU@0.5'] for c in classes]

    x = np.arange(len(classes))
    width = 0.25

    bars1 = ax.bar(x - width, miou, width, label='mIoU', color='#2E86AB', alpha=0.8)
    bars2 = ax.bar(x, iou_025, width, label='IoU@0.25', color='#F18F01', alpha=0.8)
    bars3 = ax.bar(x + width, iou_05, width, label='IoU@0.5', color='#A23B72', alpha=0.8)

    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Performance (Best Model: Plain + IoU)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend()
    ax.set_ylim(0, 1.15)

    # Add value labels
    for bar in bars1:
        h = bar.get_height()
        ax.annotate(f'{h:.2f}', xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_PATH, 'per_class_miou.png'), dpi=150)
    plt.close()
    print(f"Saved per_class_miou.png")

# ============== CHART 3: Training Convergence Curves ==============
def plot_training_curves():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Val mIoU over epochs
    ax = axes[0]
    for config, history in train_histories.items():
        epochs = [h['epoch'] for h in history]
        val_ious = [h['val_mean_iou'] for h in history]
        color = COLORS.get(config.replace(' + ', '_').lower().replace('smoothl1', 'smoothl1'), '#888888')
        linewidth = 2 if 'IoU' in config else 1
        linestyle = '-' if 'Plain' in config or ('Plain' in config) else '--'
        ax.plot(epochs, val_ious, label=config, linewidth=linewidth, alpha=0.8)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Val mIoU', fontsize=12)
    ax.set_title('Validation mIoU Over Training', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim(1, 100)
    ax.set_ylim(0, 0.3)
    ax.grid(True, alpha=0.3)

    # Right: Train loss over epochs
    ax = axes[1]
    for config, history in train_histories.items():
        epochs = [h['epoch'] for h in history]
        train_loss = [h['train_loss'] for h in history]
        color = COLORS.get(config.replace(' + ', '_').lower().replace('smoothl1', 'smoothl1'), '#888888')
        linewidth = 2 if 'IoU' in config else 1
        ax.plot(epochs, train_loss, label=config, linewidth=linewidth, alpha=0.8)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Train Loss', fontsize=12)
    ax.set_title('Training Loss Over Training', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim(1, 100)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_PATH, 'training_curves.png'), dpi=150)
    plt.close()
    print(f"Saved training_curves.png")

# ============== CHART 4: VLM vs MLP Comparison ==============
def plot_vlm_mlp_comparison():
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = ['Zero-shot\nVLM', 'Few-shot\nVLM (3-shot)', 'MLP\nFine-tuned']
    miou_values = [0.006, 0.171, 0.2623]
    colors = ['#888888', '#F18F01', '#2E86AB']

    bars = ax.bar(methods, miou_values, color=colors, alpha=0.8, width=0.5)

    ax.set_ylabel('Test mIoU', fontsize=12)
    ax.set_title('VLM Baseline vs MLP Fine-tuning', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 0.35)

    # Add value labels
    for bar, val in zip(bars, miou_values):
        ax.annotate(f'{val:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, val),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_PATH, 'vlm_mlp_comparison.png'), dpi=150)
    plt.close()
    print(f"Saved vlm_mlp_comparison.png")

# ============== MAIN ==============
if __name__ == '__main__':
    print("Generating charts...")
    plot_test_miou_comparison()
    plot_per_class_miou()
    plot_training_curves()
    plot_vlm_mlp_comparison()
    print("All charts generated successfully!")