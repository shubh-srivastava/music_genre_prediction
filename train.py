"""Train the CNN genre classifier from spectrogram images."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from config import DEFAULT_CHECKPOINT_DIR, DEFAULT_OUTPUT_DIR, AudioConfig
from utils import ensure_dir, read_json, set_seed


def run_epoch(model, loader, criterion, device, optimizer=None) -> tuple[float, float]:
    import torch
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


def train(args: argparse.Namespace) -> Path:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    from dataset import SpectrogramImageDataset
    from model import build_resnet18

    set_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)
    if args.torch_interop_threads is not None:
        torch.set_num_interop_threads(args.torch_interop_threads)
    print(
        "PyTorch CPU threads: "
        f"compute={torch.get_num_threads()} interop={torch.get_num_interop_threads()}"
    )
    print(f"Training device: {device}")

    metadata = read_json(args.output_dir / "metadata.json")
    classes = metadata["classes"]

    num_workers = args.num_workers
    if num_workers is None:
        num_workers = 0 if device.type == "cuda" else min(10, max(1, (os.cpu_count() or 2) - 2))

    train_dataset = SpectrogramImageDataset(args.output_dir, "train", image_size=args.image_size)
    val_dataset = SpectrogramImageDataset(args.output_dir, "val", image_size=args.image_size)
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
    print(
        f"DataLoader: batch_size={args.batch_size} num_workers={num_workers} "
        f"pin_memory={device.type == 'cuda'}"
    )

    model = build_resnet18(num_classes=len(classes), pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = 0.0
    best_val_loss = float("inf")
    epochs_without_loss_improvement = 0
    best_path = ensure_dir(args.checkpoint_dir) / "best_model.pt"
    last_path = ensure_dir(args.checkpoint_dir) / "last_model.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "classes": classes,
            "audio_config": metadata.get("audio_config", AudioConfig().to_dict()),
            "image_size": args.image_size,
            "val_acc": val_acc,
            "pretrained": not args.no_pretrained,
        }
        torch.save(checkpoint, last_path)

        meaningful_loss_drop = val_loss < best_val_loss - args.min_delta
        if meaningful_loss_drop:
            best_val_loss = val_loss
            best_val_acc = val_acc
            epochs_without_loss_improvement = 0
            torch.save(checkpoint, best_path)
            print(f"  saved best checkpoint: val_loss improved to {best_val_loss:.4f}")
        else:
            epochs_without_loss_improvement += 1
            if args.early_stopping_patience > 0:
                print(
                    "  no meaningful val_loss improvement "
                    f"({epochs_without_loss_improvement}/{args.early_stopping_patience})"
                )
            else:
                print("  no meaningful val_loss improvement")

        if (
            args.early_stopping_patience > 0
            and epochs_without_loss_improvement >= args.early_stopping_patience
        ):
            print(
                "Early stopping: validation loss did not improve by at least "
                f"{args.min_delta} for {args.early_stopping_patience} epochs."
            )
            break

    print(
        f"Best checkpoint: {best_path} "
        f"(val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.4f})"
    )
    return best_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=5,
        help="Stop after this many epochs without meaningful validation loss improvement. Use 0 to disable.",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=1e-4,
        help="Minimum validation loss reduction required to count as meaningful improvement.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader workers. Defaults to a CPU-friendly value on CPU and 0 on CUDA.",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=None,
        help="Set torch.set_num_threads, e.g. 12 for a 12-thread CPU.",
    )
    parser.add_argument(
        "--torch-interop-threads",
        type=int,
        default=None,
        help="Set torch.set_num_interop_threads, e.g. 12 for a 12-thread CPU.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="Example: cuda, cpu, cuda:0")
    parser.add_argument("--no-pretrained", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    train(args)


if __name__ == "__main__":
    main()
