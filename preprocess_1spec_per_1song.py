"""Create one Mel-spectrogram image per raw song."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import librosa
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

from config import AUDIO_EXTENSIONS, AudioConfig, DEFAULT_SOURCE_AUDIO_ROOT
from utils import ensure_dir


DEFAULT_OUTPUT_DIR = Path("spectrograms_1spec_per_1song")


def discover_audio_files(audio_root: Path) -> list[tuple[Path, str, str]]:
    records: list[tuple[Path, str, str]] = []
    for genre_dir in sorted(path for path in audio_root.iterdir() if path.is_dir()):
        genre = genre_dir.name
        for path in sorted(genre_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
                records.append((path, genre, path.stem))
    return records


def _stratify_or_none(labels: list[str]) -> list[str] | None:
    counts = {label: labels.count(label) for label in set(labels)}
    return labels if labels and min(counts.values()) >= 2 else None


def split_songs(
    files: list[tuple[Path, str, str]],
    val_size: float,
    test_size: float,
    seed: int,
) -> dict[Path, str]:
    paths = [path for path, _, _ in files]
    labels = [genre for _, genre, _ in files]

    train_val_paths, test_paths, train_val_labels, _ = train_test_split(
        paths,
        labels,
        test_size=test_size,
        random_state=seed,
        stratify=_stratify_or_none(labels),
    )
    val_ratio = val_size / (1.0 - test_size)
    train_paths, val_paths = train_test_split(
        train_val_paths,
        test_size=val_ratio,
        random_state=seed,
        stratify=_stratify_or_none(train_val_labels),
    )

    split_map = {path: "train" for path in train_paths}
    split_map.update({path: "val" for path in val_paths})
    split_map.update({path: "test" for path in test_paths})
    return split_map


def song_to_spectrogram_image(path: Path, config: AudioConfig, image_size: int) -> tuple[Image.Image, float]:
    signal, _ = librosa.load(path, sr=config.sample_rate, mono=True)
    duration_seconds = len(signal) / config.sample_rate
    mel = librosa.feature.melspectrogram(
        y=signal,
        sr=config.sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        n_mels=config.n_mels,
        fmin=config.fmin,
        fmax=config.fmax,
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max, top_db=config.top_db)
    normalized = np.clip((mel_db + config.top_db) / config.top_db, 0.0, 1.0)
    image_array = (np.flipud(normalized) * 255).astype(np.uint8)
    image = Image.fromarray(image_array, mode="L").convert("RGB")
    image = image.resize((image_size, image_size), Image.Resampling.BICUBIC)
    return image, duration_seconds


def write_manifest(output_dir: Path, rows: list[dict[str, str]]) -> Path:
    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "path",
                "label",
                "label_index",
                "split",
                "song_id",
                "source_path",
                "duration_seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


def preprocess_dataset(
    audio_root: Path,
    output_dir: Path,
    config: AudioConfig,
    image_size: int,
    val_size: float,
    test_size: float,
    seed: int,
    overwrite: bool,
) -> Path:
    if not audio_root.exists():
        raise FileNotFoundError(f"Audio root does not exist: {audio_root}")

    files = discover_audio_files(audio_root)
    if not files:
        raise RuntimeError(f"No supported audio files found under {audio_root}")

    ensure_dir(output_dir)
    images_dir = ensure_dir(output_dir / "images")
    genres = sorted({genre for _, genre, _ in files})
    label_to_index = {genre: index for index, genre in enumerate(genres)}
    split_map = split_songs(files, val_size=val_size, test_size=test_size, seed=seed)

    rows: list[dict[str, str]] = []
    skipped = 0
    genre_counts = {genre: 0 for genre in genres}
    for file_index, (audio_path, genre, song_id) in enumerate(files, start=1):
        split = split_map[audio_path]
        print(f"[{file_index}/{len(files)}] Processing {audio_path} | label={genre} | split={split}")
        genre_counts[genre] += 1
        image_name = f"{genre}{genre_counts[genre]:03d}.png"
        image_path = images_dir / image_name

        try:
            if overwrite or not image_path.exists():
                image, duration_seconds = song_to_spectrogram_image(audio_path, config, image_size)
                image.save(image_path)
            else:
                duration_seconds = librosa.get_duration(path=audio_path)
        except Exception as exc:
            skipped += 1
            print(f"  Skipped: {exc}")
            continue

        rows.append(
            {
                "path": str(image_path.relative_to(output_dir)),
                "label": genre,
                "label_index": str(label_to_index[genre]),
                "split": split,
                "song_id": song_id,
                "source_path": str(audio_path),
                "duration_seconds": f"{duration_seconds:.3f}",
            }
        )
        print(f"  Wrote 1 full-song spectrogram image: {image_path.name}")

    manifest_path = write_manifest(output_dir, rows)
    metadata = {
        "audio_root": str(audio_root),
        "classes": genres,
        "audio_config": asdict(config),
        "image_size": image_size,
        "num_audio_files": len(files),
        "num_images": len(rows),
        "skipped_files": skipped,
        "split_counts": {
            split: sum(1 for row in rows if row["split"] == split)
            for split in ["train", "val", "test"]
        },
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2)
    return manifest_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-root", type=Path, default=DEFAULT_SOURCE_AUDIO_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-rate", type=int, default=22_050)
    parser.add_argument("--n-mels", type=int, default=128)
    parser.add_argument("--n-fft", type=int, default=2_048)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = AudioConfig(
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
    )
    manifest_path = preprocess_dataset(
        audio_root=args.audio_root,
        output_dir=args.output_dir,
        config=config,
        image_size=args.image_size,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
