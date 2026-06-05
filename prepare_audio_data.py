"""Prepare fixed 3-minute audio clips before spectrogram preprocessing.

Rules:
- edm, hiphop, indian_indie, punjabi, bollywood_new, and bollywood_old:
  write one centered 3-minute clip for each source file.
- classical and ghazhal:
  split source audio into exactly 100 sequential 3-minute clips.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from config import AUDIO_EXTENSIONS, DEFAULT_SOURCE_AUDIO_ROOT


@dataclass(frozen=True)
class PreparationConfig:
    sample_rate: int = 22_050
    clip_minutes: float = 3.0
    split_clip_count: int = 100

    @property
    def clip_seconds(self) -> float:
        return self.clip_minutes * 60.0

    @property
    def clip_samples(self) -> int:
        return int(self.clip_seconds * self.sample_rate)


ONE_CLIP_PER_FILE_GENRES = {
    "edm",
    "hiphop",
    "indian_indie",
    "punjabi",
    "bollywood_new",
    "bollywood_old",
}

SPLIT_TO_100_GENRES = {
    "classical",
    "ghazhal",
}


def normalize_label(name: str) -> str:
    label = name.strip().lower()
    label = re.sub(r"[^a-z0-9]+", "_", label)
    return label.strip("_")


def policy_for_genre(label: str) -> str | None:
    normalized = normalize_label(label)
    if normalized in ONE_CLIP_PER_FILE_GENRES:
        return "one_center_clip_per_file"
    if normalized in SPLIT_TO_100_GENRES:
        return "split_to_100"
    return None


def discover_audio_files(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    )


def load_audio(path: Path, sample_rate: int) -> np.ndarray:
    signal, _ = librosa.load(path, sr=sample_rate, mono=True)
    return signal.astype(np.float32, copy=False)


def center_crop_or_pad(signal: np.ndarray, target_samples: int) -> np.ndarray:
    if len(signal) >= target_samples:
        start = (len(signal) - target_samples) // 2
        return signal[start : start + target_samples]

    padded = np.zeros(target_samples, dtype=np.float32)
    start = (target_samples - len(signal)) // 2
    padded[start : start + len(signal)] = signal
    return padded


def write_clip(path: Path, signal: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, signal, sample_rate)


def add_manifest_row(
    rows: list[dict[str, str]],
    label: str,
    policy: str,
    source_path: Path,
    output_path: Path,
    source_index: int,
    clip_index: int,
    config: PreparationConfig,
) -> None:
    rows.append(
        {
            "label": label,
            "policy": policy,
            "source_path": str(source_path),
            "output_path": str(output_path),
            "source_index": str(source_index),
            "clip_index": str(clip_index),
            "duration_seconds": f"{config.clip_seconds:.3f}",
        }
    )


def prepare_one_clip_per_file(
    genre_dir: Path,
    output_root: Path,
    config: PreparationConfig,
) -> list[dict[str, str]]:
    label = normalize_label(genre_dir.name)
    policy = "one_center_clip_per_file"
    files = discover_audio_files(genre_dir)
    output_dir = output_root / label
    rows: list[dict[str, str]] = []

    print(f"\nPreparing genre: {label}")
    print(f"  policy={policy} source_files={len(files)} output_clips={len(files)}")

    for index, source_path in enumerate(files, start=1):
        signal = load_audio(source_path, config.sample_rate)
        clip = center_crop_or_pad(signal, config.clip_samples)
        output_path = output_dir / f"{label}{index:03d}.wav"
        write_clip(output_path, clip, config.sample_rate)
        add_manifest_row(rows, label, policy, source_path, output_path, index, index, config)
        print(f"  [{index}/{len(files)}] {source_path.name} -> {output_path.name}")

    return rows


def prepare_split_to_100(
    genre_dir: Path,
    output_root: Path,
    config: PreparationConfig,
) -> list[dict[str, str]]:
    label = normalize_label(genre_dir.name)
    policy = "split_to_100"
    files = discover_audio_files(genre_dir)
    output_dir = output_root / label
    rows: list[dict[str, str]] = []
    output_index = 1

    print(f"\nPreparing genre: {label}")
    print(f"  policy={policy} target_clips={config.split_clip_count} source_files={len(files)}")

    for source_index, source_path in enumerate(files, start=1):
        if output_index > config.split_clip_count:
            break

        signal = load_audio(source_path, config.sample_rate)
        full_clip_count = len(signal) // config.clip_samples
        print(f"  [{source_index}/{len(files)}] {source_path.name} full_3min_clips={full_clip_count}")

        for chunk_index in range(full_clip_count):
            if output_index > config.split_clip_count:
                break

            start = chunk_index * config.clip_samples
            clip = signal[start : start + config.clip_samples]
            output_path = output_dir / f"{label}{output_index:03d}.wav"
            write_clip(output_path, clip, config.sample_rate)
            add_manifest_row(
                rows,
                label,
                policy,
                source_path,
                output_path,
                source_index,
                output_index,
                config,
            )
            print(f"    wrote {output_path.name}")
            output_index += 1

    if len(rows) < config.split_clip_count:
        print(f"  warning: only created {len(rows)}/{config.split_clip_count} clips")

    return rows


def prepare_genre(
    genre_dir: Path,
    output_root: Path,
    config: PreparationConfig,
) -> list[dict[str, str]]:
    label = normalize_label(genre_dir.name)
    policy = policy_for_genre(label)
    if policy is None:
        print(f"\nSkipping genre: {label} (no preparation policy)")
        return []
    if policy == "one_center_clip_per_file":
        return prepare_one_clip_per_file(genre_dir, output_root, config)
    if policy == "split_to_100":
        return prepare_split_to_100(genre_dir, output_root, config)
    raise ValueError(f"Unknown policy for {label}: {policy}")


def prepare_all(
    source_dir: Path,
    output_dir: Path,
    config: PreparationConfig,
    overwrite: bool,
) -> None:
    if not source_dir.exists():
        raise FileNotFoundError(f"Source data folder does not exist: {source_dir}")

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output folder already exists: {output_dir}. Use --overwrite to rebuild it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    genre_dirs = sorted(path for path in source_dir.iterdir() if path.is_dir())
    if not genre_dirs:
        raise RuntimeError(f"No genre folders found in {source_dir}")

    all_rows: list[dict[str, str]] = []
    for genre_dir in genre_dirs:
        all_rows.extend(prepare_genre(genre_dir, output_dir, config))

    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "label",
                "policy",
                "source_path",
                "output_path",
                "source_index",
                "clip_index",
                "duration_seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    metadata = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "config": asdict(config),
        "total_clips": len(all_rows),
        "clips_by_label": {},
        "minutes_by_label": {},
    }
    for row in all_rows:
        metadata["clips_by_label"].setdefault(row["label"], 0)
        metadata["clips_by_label"][row["label"]] += 1
        metadata["minutes_by_label"].setdefault(row["label"], 0.0)
        metadata["minutes_by_label"][row["label"]] += float(row["duration_seconds"]) / 60.0

    with (output_dir / "metadata.json").open("w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2)

    print(f"\nWrote processed audio manifest: {manifest_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_AUDIO_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("data_processed"))
    parser.add_argument("--clip-minutes", type=float, default=3.0)
    parser.add_argument("--split-clip-count", type=int, default=100)
    parser.add_argument("--sample-rate", type=int, default=22_050)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = PreparationConfig(
        sample_rate=args.sample_rate,
        clip_minutes=args.clip_minutes,
        split_clip_count=args.split_clip_count,
    )
    prepare_all(args.source_dir, args.output_dir, config, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
