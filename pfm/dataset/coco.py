import io
import glob
import os
import random
from typing import Callable

from PIL import Image
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader

from pfm.dataset.transforms import resize_crop_normalize


class CocoDataset(Dataset):
    """
    Reads COCO parquet files and yields image-caption pairs.

    Each parquet row holds an ``image`` ({bytes, path}) and a ``caption``
    list (COCO has ~5 captions per image); one caption is sampled per fetch.

    Each sample is a dict with:
        pixel     – (C, H, W) float tensor in [-1, 1]
        caption   – str
        key       – str (row index)
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        target_size: tuple[int, int] = (512, 512),
        filter_fn: Callable = None,
    ):
        super().__init__()
        pattern = os.path.join(data_dir, f"{split}-*.parquet")
        parquet_files = sorted(glob.glob(pattern))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found matching {pattern}")

        self.df = pd.concat(
            [pd.read_parquet(f) for f in parquet_files],
            ignore_index=True,
        )
        self.target_size = target_size
        self.filter_fn = filter_fn

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        for _ in range(100):
            row = self.df.iloc[idx]
            image = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")

            captions = row["caption"]
            if captions is None or len(captions) == 0:
                caption = ""
            else:
                caption = str(random.choice(list(captions)))

            if self.filter_fn is None or self.filter_fn(image, caption):
                break
            idx = random.randint(0, len(self) - 1)

        pixel = TF.to_tensor(image) * 255.0
        pixel = resize_crop_normalize(pixel, self.target_size)

        return {
            "pixel": pixel,
            "caption": caption,
            "key": str(idx),
        }


def collate_fn(batch):
    return {
        "pixel": torch.stack([item["pixel"] for item in batch]),
        "caption": [item["caption"] for item in batch],
        "key": [item["key"] for item in batch],
    }


def build_coco_dataloader(
    data_dir: str,
    split: str = "train",
    target_size: tuple[int, int] = (512, 512),
    filter_fn: Callable = None,
    batch_size: int = 1,
    num_workers: int = 4,
    shuffle: bool = True,
    seed: int = 42,
):
    dataset = CocoDataset(
        data_dir=data_dir,
        split=split,
        target_size=target_size,
        filter_fn=filter_fn,
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True,
        generator=generator,
        pin_memory=True,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()

    dl = build_coco_dataloader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=0,
    )
    batch = next(iter(dl))
    print(f"Dataset size: {len(dl.dataset)}")
    print(f"pixel shape: {batch['pixel'].shape}, range: [{batch['pixel'].min():.2f}, {batch['pixel'].max():.2f}]")
    print(f"caption[0]: {batch['caption'][0][:80]}")
