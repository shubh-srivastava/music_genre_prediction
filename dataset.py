"""PyTorch dataset for generated spectrogram images."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def image_to_tensor(image: Image.Image, image_size: int) -> torch.Tensor:
    image = image.convert("RGB").resize((image_size, image_size))
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (tensor - mean) / std


class SpectrogramImageDataset(Dataset):
    def __init__(self, output_dir: Path, split: str, image_size: int = 224) -> None:
        self.output_dir = output_dir
        self.split = split
        self.image_size = image_size
        manifest_path = output_dir / "manifest.csv"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest: {manifest_path}")

        with manifest_path.open("r", newline="", encoding="utf-8") as fp:
            rows = list(csv.DictReader(fp))
        self.rows = [row for row in rows if row["split"] == split]
        if not self.rows:
            raise RuntimeError(f"No rows found for split={split!r} in {manifest_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        image = Image.open(self.output_dir / row["path"])
        tensor = image_to_tensor(image, self.image_size)
        label = torch.tensor(int(row["label_index"]), dtype=torch.long)
        return tensor, label
