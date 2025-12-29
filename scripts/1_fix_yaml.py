import yaml
from pathlib import Path

# Change this if your path differs
DATASET_DIR = Path(__file__).resolve().parents[1] / "dataset"

# Read original YAML
orig_yaml = DATASET_DIR / "data.yaml"
data = yaml.safe_load(orig_yaml.read_text())

# Build a correct YOLO YAML
fixed = {
    "path": str(DATASET_DIR),          # root folder
    "train": "train/images",
    "val": "valid/images",
    "test": "test/images",
    "nc": data["nc"],
    "names": data["names"],
}

fixed_yaml = DATASET_DIR / "data_fixed.yaml"
fixed_yaml.write_text(yaml.safe_dump(fixed, sort_keys=False, allow_unicode=True))

print("✅ Saved:", fixed_yaml)
print("Classes:", fixed["names"])
