"""Train ResNet-18 using one spectrogram image per song."""

from __future__ import annotations

import argparse
import csv
import os
import time
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from config import DEFAULT_REPORT_DIR
from dataset import image_to_tensor
from model import build_resnet18
from utils import ensure_dir, read_json, set_seed


DEFAULT_OUTPUT_DIR = Path("spectrograms_1spec_per_1song")
DEFAULT_CHECKPOINT_DIR = Path("checkpoints_1spec_per_1song")


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


def run_epoch(model, loader, criterion, device, optimizer=None) -> tuple[float, float]:
    from tqdm import tqdm

    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.set_grad_enabled(is_train):
        for inputs, targets in tqdm(loader, leave=False):
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            correct += (outputs.argmax(dim=1) == targets).sum().item()
            total += batch_size
    return total_loss / total, correct / total


def append_log(path: Path, row: dict[str, object]) -> None:
    ensure_dir(path.parent)
    fields = [
        "run_id",
        "event",
        "epoch",
        "epochs_ran",
        "train_loss",
        "train_acc",
        "val_loss",
        "val_acc",
        "best_val_loss",
        "best_val_acc",
        "epoch_seconds",
        "elapsed_seconds",
        "started_at",
        "ended_at",
        "checkpoint",
    ]
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fields})


def train(args: argparse.Namespace) -> Path:
    set_seed(args.seed)
    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)
    if args.torch_interop_threads is not None:
        torch.set_num_interop_threads(args.torch_interop_threads)

    run_start = time.perf_counter()
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = ensure_dir(args.report_dir) / "training_log_1spec_per_1song.csv"

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    metadata = read_json(args.output_dir / "metadata.json")
    classes = metadata["classes"]
    image_size = int(args.image_size or metadata.get("image_size", 224))
    num_workers = args.num_workers
    if num_workers is None:
        num_workers = 0 if device.type == "cuda" else min(10, max(1, (os.cpu_count() or 2) - 2))

    train_dataset = OneSpecPerSongDataset(args.output_dir, "train", image_size)
    val_dataset = OneSpecPerSongDataset(args.output_dir, "val", image_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )

    model = build_resnet18(num_classes=len(classes), pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_path = ensure_dir(args.checkpoint_dir) / "best_model.pt"
    last_path = ensure_dir(args.checkpoint_dir) / "last_model.pt"
    best_val_loss = float("inf")
    best_val_acc = 0.0
    stale_epochs = 0
    epochs_ran = 0

    print(f"Training one-spectrogram-per-song model on {device}")
    print(f"Classes: {classes}")
    print(f"Training log: {log_path}")

    for epoch in range(1, args.epochs + 1):
        epochs_ran = epoch
        epoch_start = time.perf_counter()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        epoch_seconds = time.perf_counter() - epoch_start
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} time={epoch_seconds:.1f}s"
        )

        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "classes": classes,
            "image_size": image_size,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "pretrained": not args.no_pretrained,
            "pipeline": "1spec_per_1song",
        }
        torch.save(checkpoint, last_path)

        if val_loss < best_val_loss - args.min_delta:
            best_val_loss = val_loss
            best_val_acc = val_acc
            stale_epochs = 0
            torch.save(checkpoint, best_path)
            print(f"  saved best checkpoint: {best_path}")
        else:
            stale_epochs += 1
            print(f"  no meaningful val_loss improvement ({stale_epochs}/{args.early_stopping_patience})")

        append_log(
            log_path,
            {
                "run_id": run_id,
                "event": "epoch",
                "epoch": epoch,
                "epochs_ran": epochs_ran,
                "train_loss": f"{train_loss:.6f}",
                "train_acc": f"{train_acc:.6f}",
                "val_loss": f"{val_loss:.6f}",
                "val_acc": f"{val_acc:.6f}",
                "best_val_loss": f"{best_val_loss:.6f}",
                "best_val_acc": f"{best_val_acc:.6f}",
                "epoch_seconds": f"{epoch_seconds:.3f}",
                "started_at": started_at,
                "checkpoint": str(best_path),
            },
        )

        if args.early_stopping_patience > 0 and stale_epochs >= args.early_stopping_patience:
            print("Early stopping triggered.")
            break

    elapsed_seconds = time.perf_counter() - run_start
    ended_at = datetime.now().astimezone().isoformat(timespec="seconds")
    append_log(
        log_path,
        {
            "run_id": run_id,
            "event": "summary",
            "epochs_ran": epochs_ran,
            "best_val_loss": f"{best_val_loss:.6f}",
            "best_val_acc": f"{best_val_acc:.6f}",
            "elapsed_seconds": f"{elapsed_seconds:.3f}",
            "started_at": started_at,
            "ended_at": ended_at,
            "checkpoint": str(best_path),
        },
    )
    print(f"Best checkpoint: {best_path} (val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.4f})")
    print(f"Epochs ran: {epochs_ran}")
    print(f"Training duration: {elapsed_seconds:.1f}s")
    return best_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=0.001)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--torch-threads", type=int, default=None)
    parser.add_argument("--torch-interop-threads", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    train(args)


if __name__ == "__main__":
    main()
