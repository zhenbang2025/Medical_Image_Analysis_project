import json
import os
from datetime import datetime
from pathlib import Path

import torch


def resolve_device(device: str) -> tuple[torch.device, torch.dtype]:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda"), torch.float16
        if torch.backends.mps.is_available():
            return torch.device("mps"), torch.float16
        return torch.device("cpu"), torch.float32
    if device == "cuda":
        return torch.device("cuda"), torch.float16
    if device == "mps":
        return torch.device("mps"), torch.float16
    return torch.device("cpu"), torch.float32


def default_run_name(prefix: str) -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def prepare_run_dir(output_root: str, run_name: str | None, prefix: str, allow_overwrite: bool = False) -> str:
    out_root = Path(output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    run = run_name or default_run_name(prefix)
    run_dir = out_root / run
    if run_dir.exists() and not allow_overwrite:
        raise FileExistsError(
            f"Run directory already exists: {run_dir}. "
            "Use a different --run_name or pass --allow_overwrite."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    return str(run_dir)


def save_json(path: str, data: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_text(path: str, text: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def to_rel_if_possible(path: str) -> str:
    try:
        return os.path.relpath(path)
    except ValueError:
        return path
