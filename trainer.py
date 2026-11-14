import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import (
    SEED,
    EPOCHS,
    PATIENCE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_CLASSES,
    IGNORE_INDEX,
)
from model import ControlledUNet
from metrics import evaluate_model, measure_latency
from utils import set_seed, save_failure_visuals


def run_variant(
    name: str,
    use_dilation: bool,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    epochs: int = EPOCHS,
    patience: int = PATIENCE,
    seed: int = SEED,
):
    print(f"\n{'='*60}\nRunning Experiment: {name} (Dilated Bottleneck={use_dilation})\n{'='*60}")
    set_seed(seed)

    model = ControlledUNet(num_classes=NUM_CLASSES, use_dilated_bottleneck=use_dilation).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    best_val_miou = -1.0
    epochs_no_improve = 0
    best_ckpt = output_dir / f"{name}_best.pt"
    training_log = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        val_metrics, _ = evaluate_model(model, val_loader, device)
        val_miou = val_metrics["mIoU"]

        log_entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_mIoU": val_miou,
            "val_per_class_iou": val_metrics["per_class_iou"],
        }
        training_log.append(log_entry)
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val mIoU: {val_miou:.2f}%")

        if val_miou > best_val_miou:
            best_val_miou = val_miou
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_ckpt)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    test_metrics, worst_failures = evaluate_model(model, test_loader, device)
    latency_ms = measure_latency(model, device)
    save_failure_visuals(worst_failures, output_dir / "failures", prefix=name)

    summary = {
        "variant": name,
        "use_dilated_bottleneck": use_dilation,
        "trainable_parameters": param_count,
        "latency_ms_batch1_fp32": latency_ms,
        "device": str(device),
        "test_results": test_metrics,
        "training_log": training_log,
    }
    with open(output_dir / f"{name}_report.json", "w") as f:
        json.dump(summary, f, indent=4)

    print(f"Result | {name} | Params: {param_count:,} | Latency: {latency_ms} ms | Test mIoU: {test_metrics['mIoU']}%")
    return summary