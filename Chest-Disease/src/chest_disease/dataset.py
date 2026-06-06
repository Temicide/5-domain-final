from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .config import ID_COLUMN, LABEL_COLUMNS


class ChestDiseaseDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transforms, include_targets: bool) -> None:
        self.frame = frame.reset_index(drop=True)
        self.transforms = transforms
        self.include_targets = include_targets

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        image = Image.open(Path(row["image_path"])).convert("RGB")
        image_tensor = self.transforms(image)
        filename = str(row[ID_COLUMN])
        if self.include_targets:
            target = torch.tensor(row[LABEL_COLUMNS].astype(float).to_numpy(), dtype=torch.float32)
            return image_tensor, target, filename
        return image_tensor, filename


def build_transforms(image_size: int, train: bool, model_family: str):
    ops = []
    if train:
        ops.extend(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )
    else:
        ops.append(transforms.Resize((image_size, image_size)))
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transforms.Compose(ops)
