"""
Pre-defined dataset configurations.

Users select a dataset by name (e.g. ``--dataset_type laion``).
All dataset-specific parameters (paths, caption keys, etc.) are
hardcoded here so they don't leak into CLI flags.
"""

DATASET_CONFIGS = {
    "laion": {
        "type": "laion",
        "data_dir": "data/laion-art",
        "split": "train",
    },
    "coco": {
        "type": "coco",
        "data_dir": "data/coco",
        "split": "train",
    },
}
