"""Convert labeled songs into leakage-safe Mel-spectrogram PNG images."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path

from config import AUDIO_EXTENSIONS, DEFAULT_AUDIO_ROOT, DEFAULT_OUTPUT_DIR, AudioConfig
from utils import ensure_dir, write_json


def discover_audio_files(audio_root: Path) -> list[tuple[Path, str, str]]:
    records: list[tuple[Path, str, str]] = []
    for genre_dir in sorted(path for path in audio_root.iterdir() if path.is_dir()):
        genre = genre_dir.name
        for path in sorted(genre_dir.rglob("*")):
            if path.suffix.lower() in AUDIO_EXTENSIONS:
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
    from sklearn.model_selection import train_test_split

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


def write_manifest(output_dir: Path, rows: list[dict[str, str]]) -> Path:
    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=["path", "label", "label_index", "split", "song_id", "chunk_index"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


def preprocess_dataset(
    audio_root: Path,
    output_dir: Path,
    config: AudioConfig,
    val_size: float,
    test_size: float,
    seed: int,
    overwrite: bool,
) -> Path:
    from audio import audio_to_mels, save_spectrogram_image

    if not audio_root.exists():
        raise FileNotFoundError(f"Audio root does not exist: {audio_root}")

    ensure_dir(output_dir)
    images_dir = ensure_dir(output_dir / "images")
    files = discover_audio_files(audio_root)
    if not files:
        raise RuntimeError(f"No supported audio files found under {audio_root}")

    genres = sorted({genre for _, genre, _ in files})
    label_to_index = {genre: index for index, genre in enumerate(genres)}
    split_map = split_songs(files, val_size=val_size, test_size=test_size, seed=seed)

    rows: list[dict[str, str]] = []
    skipped = 0
    for audio_path, genre, song_id in files:
        mels = audio_to_mels(audio_path, config)
        if not mels:
            skipped += 1
            continue

        for chunk_index, mel in enumerate(mels):
            rel_song = audio_path.relative_to(audio_root).with_suffix("")
            image_name = f"{rel_song.as_posix().replace('/', '__')}__chunk_{chunk_index:03d}.png"
            image_path = images_dir / image_name
            if overwrite or not image_path.exists():
                save_spectrogram_image(mel, image_path)

            rows.append(
                {
                    "path": str(image_path.relative_to(output_dir)),
                    "label": genre,
                    "label_index": str(label_to_index[genre]),
                    "split": split_map[audio_path],
                    "song_id": song_id,
                    "chunk_index": str(chunk_index),
                }
            )

    manifest_path = write_manifest(output_dir, rows)
    write_json(
        output_dir / "metadata.json",
        {
            "audio_root": str(audio_root),
            "classes": genres,
            "audio_config": asdict(config),
            "num_audio_files": len(files),
            "num_images": len(rows),
            "skipped_short_files": skipped,
            "split_counts": {
                split: sum(1 for row in rows if row["split"] == split)
                for split in ["train", "val", "test"]
            },
        },
    )
    return manifest_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-root", type=Path, default=DEFAULT_AUDIO_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-rate", type=int, default=22_050)
    parser.add_argument("--segment-seconds", type=float, default=3.0)
    parser.add_argument("--n-mels", type=int, default=128)
    parser.add_argument("--n-fft", type=int, default=2_048)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = AudioConfig(
        sample_rate=args.sample_rate,
        segment_seconds=args.segment_seconds,
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
    )
    manifest_path = preprocess_dataset(
        audio_root=args.audio_root,
        output_dir=args.output_dir,
        config=config,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
