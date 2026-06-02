"""Prepare balanced genre audio folders before spectrogram preprocessing.

This script reads labeled audio folders from data/, trims or splits songs by
genre policy, renames the resulting clips, and writes them into data_processed/.
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
import soundfile as sf

from config import AUDIO_EXTENSIONS, DEFAULT_SOURCE_AUDIO_ROOT


@dataclass(frozen=True)
class PreparationConfig:
    sample_rate: int = 22_050
    target_minutes_per_genre: float = 300.0
    clip_minutes: float = 3.0
    long_song_minutes: float = 4.0
    min_clip_seconds: float = 30.0

    @property
    def target_seconds(self) -> float:
        return self.target_minutes_per_genre * 60.0

    @property
    def clip_seconds(self) -> float:
        return self.clip_minutes * 60.0

    @property
    def long_song_seconds(self) -> float:
        return self.long_song_minutes * 60.0


SPLIT_GENRES = {
    "classical",
    # "carnatic",
    # "gazal",
    # "gazals",
    "ghazhal",
    # "ghazals",
    # "semiclassical",
}

CENTER_TRIM_GENRES = {
    "bollywood",
    # "bollypop",
    "edm",
    "hiphop",
    # "hh",
    # "indian_indie",
    "indian_indie",
    # "indie",
    # "indina_indie",
    "punjabi",
    # "sufi",
}


def normalize_label(name: str) -> str:
    label = name.strip().lower()
    label = re.sub(r"[^a-z0-9]+", "_", label)
    return label.strip("_")


def policy_for_genre(label: str) -> str:
    normalized = normalize_label(label)
    if normalized in {"ghazal", "ghazals", "gazal", "gazals"}:
        return "split"
    if normalized in SPLIT_GENRES:
        return "split"
    if normalized in CENTER_TRIM_GENRES:
        return "center_trim"
    return "center_trim"


def discover_audio_files(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    )


def load_audio(path: Path, sample_rate: int):
    signal, _ = librosa.load(path, sr=sample_rate, mono=True)
    return signal


def center_crop(signal, sample_rate: int, seconds: float):
    target_samples = min(len(signal), int(seconds * sample_rate))
    start = max(0, (len(signal) - target_samples) // 2)
    end = start + target_samples
    return signal[start:end]


def iter_split_clips(signal, sample_rate: int, clip_seconds: float):
    clip_samples = int(clip_seconds * sample_rate)
    for start in range(0, len(signal), clip_samples):
        end = min(start + clip_samples, len(signal))
        yield signal[start:end]


def clip_duration(signal, sample_rate: int) -> float:
    return len(signal) / sample_rate


def write_clip(path: Path, signal, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, signal, sample_rate)


def prepare_genre(
    genre_dir: Path,
    output_root: Path,
    config: PreparationConfig,
) -> list[dict[str, str]]:
    label = normalize_label(genre_dir.name)
    policy = policy_for_genre(label)
    files = discover_audio_files(genre_dir)
    output_dir = output_root / label
    rows: list[dict[str, str]] = []
    total_seconds = 0.0
    output_index = 1

    print(f"\nPreparing genre: {label}")
    print(f"  source_files={len(files)} policy={policy} target={config.target_minutes_per_genre:.0f} minutes")

    for file_index, source_path in enumerate(files, start=1):
        if total_seconds >= config.target_seconds:
            break
        if config.target_seconds - total_seconds < config.min_clip_seconds:
            break

        signal = load_audio(source_path, config.sample_rate)
        source_seconds = clip_duration(signal, config.sample_rate)
        remaining_seconds = config.target_seconds - total_seconds
        print(f"  [{file_index}/{len(files)}] {source_path.name} duration={source_seconds / 60:.2f} min")

        if policy == "split":
            candidate_clips = iter_split_clips(signal, config.sample_rate, config.clip_seconds)
        else:
            if source_seconds > config.long_song_seconds:
                seconds = min(config.clip_seconds, remaining_seconds)
                candidate_clips = [center_crop(signal, config.sample_rate, seconds)]
            else:
                seconds = min(source_seconds, remaining_seconds)
                candidate_clips = [center_crop(signal, config.sample_rate, seconds)]

        for clip in candidate_clips:
            if total_seconds >= config.target_seconds:
                break

            remaining_seconds = config.target_seconds - total_seconds
            if remaining_seconds < config.min_clip_seconds:
                break
            current_seconds = clip_duration(clip, config.sample_rate)
            if current_seconds > remaining_seconds:
                clip = center_crop(clip, config.sample_rate, remaining_seconds)
                current_seconds = clip_duration(clip, config.sample_rate)

            if current_seconds < config.min_clip_seconds:
                print(f"    skipped short remainder={current_seconds:.1f}s")
                continue

            output_path = output_dir / f"{label}{output_index:03d}.wav"
            write_clip(output_path, clip, config.sample_rate)
            total_seconds += current_seconds
            rows.append(
                {
                    "label": label,
                    "policy": policy,
                    "source_path": str(source_path),
                    "output_path": str(output_path),
                    "duration_seconds": f"{current_seconds:.3f}",
                }
            )
            print(f"    wrote {output_path.name} duration={current_seconds / 60:.2f} min")
            output_index += 1

    print(f"  total_written={total_seconds / 60:.2f} minutes clips={len(rows)}")
    shortfall_seconds = config.target_seconds - total_seconds
    if shortfall_seconds > 1.0:
        print(
            "  warning: genre did not reach target; "
            f"short by {shortfall_seconds / 60:.2f} minutes"
        )
    return rows


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
            fieldnames=["label", "policy", "source_path", "output_path", "duration_seconds"],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    metadata = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "config": asdict(config),
        "total_clips": len(all_rows),
        "minutes_by_label": {},
    }
    for row in all_rows:
        metadata["minutes_by_label"].setdefault(row["label"], 0.0)
        metadata["minutes_by_label"][row["label"]] += float(row["duration_seconds"]) / 60.0

    with (output_dir / "metadata.json").open("w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2)

    print(f"\nWrote processed audio manifest: {manifest_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_AUDIO_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("data_processed"))
    parser.add_argument("--target-minutes", type=float, default=300.0)
    parser.add_argument("--clip-minutes", type=float, default=3.0)
    parser.add_argument("--long-song-minutes", type=float, default=4.0)
    parser.add_argument("--min-clip-seconds", type=float, default=30.0)
    parser.add_argument("--sample-rate", type=int, default=22_050)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = PreparationConfig(
        sample_rate=args.sample_rate,
        target_minutes_per_genre=args.target_minutes,
        clip_minutes=args.clip_minutes,
        long_song_minutes=args.long_song_minutes,
        min_clip_seconds=args.min_clip_seconds,
    )
    prepare_all(args.source_dir, args.output_dir, config, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
