"""Validate/evaluate a one-spectrogram-per-song checkpoint."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config import DEFAULT_REPORT_DIR
from dataset import image_to_tensor
from model import build_resnet18
from utils import ensure_dir


DEFAULT_OUTPUT_DIR = Path("spectrograms_1spec_per_1song")
DEFAULT_CHECKPOINT = Path("checkpoints_1spec_per_1song") / "best_model.pt"


class OneSpecPerSongDataset(Dataset):
    def __init__(self, output_dir: Path, split: str, image_size: int) -> None:
        self.output_dir = output_dir
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

    def __getitem__(self, index: int):
        row = self.rows[index]
        image = Image.open(self.output_dir / row["path"])
        return image_to_tensor(image, self.image_size), torch.tensor(int(row["label_index"]), dtype=torch.long)


def validate(args: argparse.Namespace) -> None:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(args.checkpoint, map_location=device)
    classes = checkpoint["classes"]
    image_size = int(checkpoint.get("image_size", 224))

    dataset = OneSpecPerSongDataset(args.output_dir, args.split, image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model = build_resnet18(num_classes=len(classes), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for inputs, targets in tqdm(loader, leave=False):
            outputs = model(inputs.to(device))
            y_pred.extend(outputs.argmax(dim=1).cpu().tolist())
            y_true.extend(targets.tolist())

    print(classification_report(y_true, y_pred, target_names=classes, zero_division=0))
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=classes)
    display.plot(xticks_rotation=45, cmap="Blues", values_format="d")
    plt.tight_layout()

    output_path = ensure_dir(args.report_dir) / f"confusion_matrix_1spec_per_1song_{args.split}.png"
    plt.savefig(output_path, dpi=180)
    print(f"Saved confusion matrix: {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    validate(args)


if __name__ == "__main__":
    main()
