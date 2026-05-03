import shutil
from pathlib import Path

import kagglehub


def find_chest_xray_dir(root: Path) -> Path:
	if (root / "chest_xray").exists():
		return root / "chest_xray"
	for path in root.rglob("chest_xray"):
		if path.is_dir():
			return path
	raise FileNotFoundError("Could not locate chest_xray directory in Kaggle cache")


def main() -> None:
	# Download latest version to Kaggle cache
	cache_path = Path(
		kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
	)
	print("Path to dataset files:", cache_path)

	dataset_dir = find_chest_xray_dir(cache_path)
	target_dir = Path(__file__).resolve().parent / "chest_xray"

	if target_dir.exists():
		print(f"Target directory already exists: {target_dir}")
		print("Skipping copy. Remove it first if you want to re-copy.")
		return

	target_dir.parent.mkdir(parents=True, exist_ok=True)
	shutil.copytree(dataset_dir, target_dir)
	print(f"Copied dataset to: {target_dir}")


if __name__ == "__main__":
	main()