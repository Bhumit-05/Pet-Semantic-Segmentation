import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import IMG_SIZE, NUM_CLASSES


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int = NUM_CLASSES):
    model.eval()
    conf_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)
    sample_records = []

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        logits = model(images)
        preds = torch.argmax(logits, dim=1)

        valid = (targets >= 0) & (targets < num_classes)
        t_valid = targets[valid]
        p_valid = preds[valid]
        indices = t_valid * num_classes + p_valid
        conf_matrix += torch.bincount(indices, minlength=num_classes ** 2).reshape(num_classes, num_classes)

        for i in range(images.size(0)):
            v_i = (targets[i] >= 0) & (targets[i] < num_classes)
            inter = ((preds[i][v_i] == 0) & (targets[i][v_i] == 0)).sum().item()
            union = ((preds[i][v_i] == 0) | (targets[i][v_i] == 0)).sum().item()
            fg_iou = inter / max(union, 1)
            sample_records.append((fg_iou, images[i].cpu(), targets[i].cpu(), preds[i].cpu()))

    conf_np = conf_matrix.cpu().numpy()
    tp = np.diag(conf_np)
    fp = conf_np.sum(axis=0) - tp
    fn = conf_np.sum(axis=1) - tp
    denom = tp + fp + fn
    per_class_iou = np.where(denom > 0, tp / denom, 0.0)
    miou = float(np.mean(per_class_iou))

    sample_records.sort(key=lambda x: x[0])
    return {
        "per_class_iou": {
            "foreground_0": round(float(per_class_iou[0]) * 100, 2),
            "background_1": round(float(per_class_iou[1]) * 100, 2),
            "border_2": round(float(per_class_iou[2]) * 100, 2),
        },
        "mIoU": round(miou * 100, 2),
    }, sample_records[:4]


@torch.no_grad()
def measure_latency(model: nn.Module, device: torch.device, img_size: int = IMG_SIZE, runs: int = 50):
    model.eval()
    dummy = torch.randn(1, 3, img_size, img_size, device=device)

    for _ in range(5):
        _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(runs):
        _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed_ms = ((time.perf_counter() - start) / runs) * 1000.0
    return round(elapsed_ms, 4)