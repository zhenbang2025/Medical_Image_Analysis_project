import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


CLASS_NAMES = ["NORMAL", "PNEUMONIA"]


@dataclass
class Sample:
    path: str
    label: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def list_split_samples(dataset_dir: str, split: str) -> List[Sample]:
    root = Path(dataset_dir) / split
    samples: List[Sample] = []

    for label_idx, class_name in enumerate(CLASS_NAMES):
        class_dirs = [p for p in root.rglob(class_name) if p.is_dir()]
        if not class_dirs:
            class_dirs = [root / class_name]

        for class_dir in class_dirs:
            for ext in ("*.jpeg", "*.jpg", "*.png"):
                for img_path in sorted(class_dir.glob(ext)):
                    samples.append(Sample(str(img_path), label_idx))

    return samples


def make_train_val_split(
    train_samples: List[Sample],
    val_fraction: float,
    seed: int,
) -> Tuple[List[Sample], List[Sample]]:
    labels = [s.label for s in train_samples]
    train_idx, val_idx = train_test_split(
        list(range(len(train_samples))),
        test_size=val_fraction,
        random_state=seed,
        stratify=labels,
    )
    train_split = [train_samples[i] for i in train_idx]
    val_split = [train_samples[i] for i in val_idx]
    return train_split, val_split


def subsample_by_fraction(samples: List, fraction: float, seed: int) -> List:
    if fraction <= 0.0:
        return []
    if fraction >= 1.0:
        return samples
    rng = random.Random(seed)
    labels = [s.label if hasattr(s, "label") else s[1] for s in samples]
    idx_by_label: Dict[int, List[int]] = {0: [], 1: []}
    for idx, label in enumerate(labels):
        idx_by_label[label].append(idx)

    selected_idx: List[int] = []
    for label, idxs in idx_by_label.items():
        target = max(1, int(len(idxs) * fraction))
        selected_idx.extend(rng.sample(idxs, target))

    return [samples[i] for i in selected_idx]


def save_split_file(path: str, splits: Dict[str, List[Sample]]) -> None:
    payload = {
        name: [{"path": s.path, "label": s.label} for s in items]
        for name, items in splits.items()
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_split_file(path: str) -> Dict[str, List[Sample]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    splits: Dict[str, List[Sample]] = {}
    for name, items in payload.items():
        splits[name] = [Sample(item["path"], item["label"]) for item in items]
    return splits


class ImageDataset(Dataset):
    def __init__(self, samples: List[Sample], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        img = Image.open(sample.path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, sample.label, sample.path
