import sys
import json
import random
import platform
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image

from config import IMAGENET_MEAN, IMAGENET_STD


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def log_environment(output_dir: Path):
    env_info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "numpy_version": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }
    with open(output_dir / "environment_versions.json", "w") as f:
        json.dump(env_info, f, indent=4)


def save_failure_visuals(failures, save_dir: Path, prefix: str):
    save_dir.mkdir(parents=True, exist_ok=True)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    palette = np.array([[255, 80, 80], [40, 40, 40], [80, 220, 100]], dtype=np.uint8)

    for idx, (iou_val, img_t, gt_t, pred_t) in enumerate(failures):
        img_vis = torch.clamp(img_t * std + mean, 0, 1).permute(1, 2, 0).numpy()
        img_vis = (img_vis * 255).astype(np.uint8)

        gt_np = np.clip(gt_t.numpy(), 0, 2)
        pred_np = np.clip(pred_t.numpy(), 0, 2)
        gt_rgb = palette[gt_np]
        pred_rgb = palette[pred_np]

        combined = np.concatenate([img_vis, gt_rgb, pred_rgb], axis=1)
        Image.fromarray(combined).save(save_dir / f"{prefix}_failure_{idx}_fgIoU_{iou_val:.2f}.png")