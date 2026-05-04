import csv
import random
from collections import defaultdict
from pathlib import Path


def patient_id_from_image_name(name: str) -> str:
    return name.split("_")[0]


def find_image_paths(data_root: str) -> dict[str, str]:
    root = Path(data_root)
    image_dirs = sorted([p for p in root.glob("images_*") if p.is_dir()])
    image_map: dict[str, str] = {}
    for d in image_dirs:
        inner = d / "images"
        if not inner.exists():
            continue
        for img in inner.glob("*.png"):
            image_map[img.name] = str(img)
    return image_map


def load_bbox_rows(bbox_csv: str, classes: list[str]) -> list[dict]:
    rows = []
    with open(bbox_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) < 6:
                continue
            cls = row[1].strip()
            if cls not in classes:
                continue
            img_name = row[0].strip()
            rows.append(
                {
                    "name": img_name,
                    "class": cls,
                    "x": float(row[2]),
                    "y": float(row[3]),
                    "w": float(row[4]),
                    "h": float(row[5]),
                    "patient_id": patient_id_from_image_name(img_name),
                }
            )
    return rows


def stratified_patient_split(
    rows: list[dict],
    classes: list[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, list[dict]]:
    random.seed(seed)
    by_class = defaultdict(list)
    for r in rows:
        by_class[r["class"]].append(r)

    split = {"train": [], "val": [], "test": []}
    for cls in classes:
        cls_rows = by_class[cls]
        by_pid = defaultdict(list)
        for r in cls_rows:
            by_pid[r["patient_id"]].append(r)
        pids = list(by_pid.keys())
        random.shuffle(pids)
        n_pid = len(pids)
        n_train = int(n_pid * train_ratio)
        n_val = int(n_pid * val_ratio)
        train_p = set(pids[:n_train])
        val_p = set(pids[n_train:n_train + n_val])
        test_p = set(pids[n_train + n_val:])

        for pid in train_p:
            split["train"].extend(by_pid[pid])
        for pid in val_p:
            split["val"].extend(by_pid[pid])
        for pid in test_p:
            split["test"].extend(by_pid[pid])
    return split


def cap_per_class(rows: list[dict], classes: list[str], max_per_class: int, seed: int = 42) -> list[dict]:
    random.seed(seed)
    by_class = defaultdict(list)
    for r in rows:
        by_class[r["class"]].append(r)
    out = []
    for cls in classes:
        arr = by_class[cls]
        if len(arr) > max_per_class:
            arr = random.sample(arr, max_per_class)
        out.extend(arr)
    return out
