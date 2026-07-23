import math

import torch
import torchvision.transforms.functional as TF


def resize_crop_normalize(x: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
    """Resize-and-center-crop a (C, H, W) uint8-range tensor to [-1, 1]."""
    h, w = x.shape[-2:]
    bh, bw = target_size
    scale = max(bh / h, bw / w)
    resize_h, resize_w = math.ceil(h * scale), math.ceil(w * scale)

    x = TF.resize(x, (resize_h, resize_w),
                  interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
    x = TF.center_crop(x, target_size)
    return x / 127.5 - 1.0
