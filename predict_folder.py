"""Predict genres for every audio file in the predict folder."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from config import AUDIO_EXTENSIONS, DEFAULT_CHECKPOINT_DIR
from predict import predict_file


def discover_audio_files(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Prediction folder does not exist: {input_dir}")
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    )


def write_predictions_csv(output_path: Path, rows: list[dict[str, object]], classes: list[str]) -> None:
    fieldnames = [
        "file",
        "predicted_genre",
        "confidence",
        "vote_share",
        "chunks",
        *[f"prob_{class_name}" for class_name in classes],
    ]
    with output_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def predict_folder(
    input_dir: Path,
    checkpoint: Path,
    output_csv: Path,
    device: str | None,
) -> list[dict[str, object]]:
    files = discover_audio_files(input_dir)
    if not files:
        raise RuntimeError(f"No supported audio files found in {input_dir}")

    rows: list[dict[str, object]] = []
    classes: list[str] | None = None

    for audio_path in files:
        result = predict_file(audio_path, checkpoint, device)
        probabilities = result["class_probabilities"]
        if classes is None:
            classes = list(probabilities.keys())

        row: dict[str, object] = {
            "file": str(audio_path),
            "predicted_genre": result["genre"],
            "confidence": result["confidence"],
            "vote_share": result["vote_share"],
            "chunks": result["chunks"],
        }
        for class_name in classes:
            row[f"prob_{class_name}"] = probabilities[class_name]
        rows.append(row)

        print(f"\n{audio_path}")
        print(f"  predicted_genre: {result['genre']}")
        print(f"  confidence: {result['confidence']:.2%}")
        print(f"  vote_share: {result['vote_share']:.2%} across {result['chunks']} chunks")
        print("  class_probabilities:")
        for class_name in classes:
            print(f"    {class_name}: {probabilities[class_name]:.2%}")

    write_predictions_csv(output_csv, rows, classes or [])
    print(f"\nWrote predictions: {output_csv}")
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("predict"))
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_DIR / "best_model.pt")
    parser.add_argument("--output-csv", type=Path, default=Path("predictions.csv"))
    parser.add_argument("--device", default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        predict_folder(args.input_dir, args.checkpoint, args.output_csv, args.device)
    except (FileNotFoundError, RuntimeError) as exc:
        raise SystemExit(f"Error: {exc}") from exc


if __name__ == "__main__":
    main()
