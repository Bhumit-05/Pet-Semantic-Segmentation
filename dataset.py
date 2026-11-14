import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.datasets import OxfordIIITPet
import torchvision.transforms.functional as TF

from config import (
    SEED,
    IMG_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    VAL_SPLIT,
    IGNORE_INDEX,
    DATA_DIR,
    IMAGENET_MEAN,
    IMAGENET_STD,
)


class PetSegmentationContract(Dataset):
    def __init__(self, base_dataset, img_size: int = IMG_SIZE, augment: bool = False):
        self.base_dataset = base_dataset
        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, mask = self.base_dataset[idx]

        img = TF.resize(img, [self.img_size, self.img_size], interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [self.img_size, self.img_size], interpolation=TF.InterpolationMode.NEAREST)

        if self.augment and random.random() < 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)

        img_tensor = TF.to_tensor(img)
        img_tensor = TF.normalize(img_tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)

        mask_np = np.array(mask, dtype=np.int64)
        remapped_mask = np.full_like(mask_np, fill_value=IGNORE_INDEX)
        remapped_mask[mask_np == 1] = 0
        remapped_mask[mask_np == 2] = 1
        remapped_mask[mask_np == 3] = 2

        return img_tensor, torch.from_numpy(remapped_mask)


def build_dataloaders(
    data_dir: Path = DATA_DIR,
    img_size: int = IMG_SIZE,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    val_split: float = VAL_SPLIT,
    seed: int = SEED,
):
    raw_trainval = OxfordIIITPet(root=str(data_dir), split="trainval", target_types="segmentation", download=True)
    raw_test = OxfordIIITPet(root=str(data_dir), split="test", target_types="segmentation", download=True)

    val_len = int(val_split * len(raw_trainval))
    train_len = len(raw_trainval) - val_len
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(raw_trainval, [train_len, val_len], generator=generator)

    train_ds = PetSegmentationContract(train_subset, img_size=img_size, augment=True)
    val_ds = PetSegmentationContract(val_subset, img_size=img_size, augment=False)
    test_ds = PetSegmentationContract(raw_test, img_size=img_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader