"""Train the CNN genre classifier from spectrogram images."""

from __future__ import annotations

import argparse
import csv
import os
import time
from datetime import datetime
from pathlib import Path

from config import DEFAULT_CHECKPOINT_DIR, DEFAULT_OUTPUT_DIR, DEFAULT_REPORT_DIR, AudioConfig
from utils import ensure_dir, read_json, set_seed


TRAINING_LOG_FIELDS = [
    "run_id",
    "event",
    "model",
    "started_at",
    "ended_at",
    "elapsed_seconds",
    "epoch",
    "epochs_ran",
    "epoch_seconds",
    "train_loss",
    "train_acc",
    "val_loss",
    "val_acc",
    "best_val_loss",
    "best_val_acc",
    "checkpoint",
    "device",
    "batch_size",
    "lr",
    "weight_decay",
    "image_size",
    "num_workers",
    "torch_threads",
    "torch_interop_threads",
    "pretrained",
    "early_stopped",
]


def append_training_log(log_path: Path, row: dict[str, object]) -> None:
    ensure_dir(log_path.parent)
    file_exists = log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=TRAINING_LOG_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in TRAINING_LOG_FIELDS})


def append_training_logs(log_paths: list[Path], row: dict[str, object]) -> None:
    for log_path in log_paths:
        append_training_log(log_path, row)


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

    run_start = time.perf_counter()
    now = datetime.now().astimezone()
    started_at = now.isoformat(timespec="seconds")
    run_id = now.strftime("%Y%m%d_%H%M%S")
    report_dir = ensure_dir(args.report_dir)
    aggregate_log_path = report_dir / "training_log.csv"
    run_log_path = report_dir / f"training_log_{run_id}.csv"
    log_paths = [aggregate_log_path, run_log_path]

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)
    if args.torch_interop_threads is not None:
        torch.set_num_interop_threads(args.torch_interop_threads)

    metadata = read_json(args.output_dir / "metadata.json")
    classes = metadata["classes"]

    num_workers = args.num_workers
    if num_workers is None:
        num_workers = 0 if device.type == "cuda" else min(10, max(1, (os.cpu_count() or 2) - 2))

    model_name = "resnet18"
    pretrained = not args.no_pretrained

    print(
        "PyTorch CPU threads: "
        f"compute={torch.get_num_threads()} interop={torch.get_num_interop_threads()}"
    )
    print(f"Training device: {device}")
    print(f"Training log: {aggregate_log_path}")
    print(f"This run log: {run_log_path}")

    best_path = ensure_dir(args.checkpoint_dir) / "best_model.pt"
    last_path = ensure_dir(args.checkpoint_dir) / "last_model.pt"

    append_training_logs(
        log_paths,
        {
            "run_id": run_id,
            "event": "start",
            "model": model_name,
            "started_at": started_at,
            "checkpoint": str(best_path),
            "device": str(device),
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "image_size": args.image_size,
            "num_workers": num_workers,
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "pretrained": pretrained,
            "early_stopped": False,
        },
    )

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

    model = build_resnet18(num_classes=len(classes), pretrained=pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = 0.0
    best_val_loss = float("inf")
    epochs_without_loss_improvement = 0
    epochs_ran = 0
    early_stopped = False

    for epoch in range(1, args.epochs + 1):
        epochs_ran = epoch
        epoch_start = time.perf_counter()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        epoch_seconds = time.perf_counter() - epoch_start

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"time={epoch_seconds:.1f}s"
        )

        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "classes": classes,
            "audio_config": metadata.get("audio_config", AudioConfig().to_dict()),
            "image_size": args.image_size,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "pretrained": pretrained,
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

        early_stopped = (
            args.early_stopping_patience > 0
            and epochs_without_loss_improvement >= args.early_stopping_patience
        )

        append_training_logs(
            log_paths,
            {
                "run_id": run_id,
                "event": "epoch",
                "model": model_name,
                "started_at": started_at,
                "epoch": epoch,
                "epochs_ran": epochs_ran,
                "epoch_seconds": f"{epoch_seconds:.3f}",
                "train_loss": f"{train_loss:.6f}",
                "train_acc": f"{train_acc:.6f}",
                "val_loss": f"{val_loss:.6f}",
                "val_acc": f"{val_acc:.6f}",
                "best_val_loss": f"{best_val_loss:.6f}",
                "best_val_acc": f"{best_val_acc:.6f}",
                "checkpoint": str(best_path),
                "device": str(device),
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "image_size": args.image_size,
                "num_workers": num_workers,
                "torch_threads": torch.get_num_threads(),
                "torch_interop_threads": torch.get_num_interop_threads(),
                "pretrained": pretrained,
                "early_stopped": early_stopped,
            },
        )

        if early_stopped:
            print(
                "Early stopping: validation loss did not improve by at least "
                f"{args.min_delta} for {args.early_stopping_patience} epochs."
            )
            break

    elapsed_seconds = time.perf_counter() - run_start
    ended_at = datetime.now().astimezone().isoformat(timespec="seconds")
    append_training_logs(
        log_paths,
        {
            "run_id": run_id,
            "event": "summary",
            "model": model_name,
            "started_at": started_at,
            "ended_at": ended_at,
            "elapsed_seconds": f"{elapsed_seconds:.3f}",
            "epochs_ran": epochs_ran,
            "best_val_loss": f"{best_val_loss:.6f}",
            "best_val_acc": f"{best_val_acc:.6f}",
            "checkpoint": str(best_path),
            "device": str(device),
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "image_size": args.image_size,
            "num_workers": num_workers,
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "pretrained": pretrained,
            "early_stopped": early_stopped,
        },
    )

    print(
        f"Best checkpoint: {best_path} "
        f"(val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.4f})"
    )
    print(f"Epochs ran: {epochs_ran}")
    print(f"Training duration: {elapsed_seconds:.1f}s")
    print(f"Training log saved to: {aggregate_log_path}")
    print(f"This run log saved to: {run_log_path}")
    return best_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
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
        help="Set torch.set_num_threads, e.g. 10 for a 12-thread CPU while leaving 2 threads free.",
    )
    parser.add_argument(
        "--torch-interop-threads",
        type=int,
        default=None,
        help="Set torch.set_num_interop_threads, e.g. 10 for a 12-thread CPU while leaving 2 threads free.",
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
