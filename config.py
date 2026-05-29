"""Shared configuration for the music genre classifier."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 22_050
    segment_seconds: float = 3.0
    n_fft: int = 2_048
    hop_length: int = 512
    n_mels: int = 128
    fmin: float = 20.0
    fmax: float | None = None
    top_db: float = 80.0

    @property
    def segment_samples(self) -> int:
        return int(self.sample_rate * self.segment_seconds)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "AudioConfig":
        if not data:
            return cls()
        valid_keys = cls.__dataclass_fields__.keys()
        return cls(**{key: value for key, value in data.items() if key in valid_keys})


AUDIO_EXTENSIONS = {".au", ".flac", ".m4a", ".mp3", ".ogg", ".wav"}

DEFAULT_AUDIO_ROOT = Path("data")
DEFAULT_OUTPUT_DIR = Path("spectrograms")
DEFAULT_CHECKPOINT_DIR = Path("checkpoints")
DEFAULT_REPORT_DIR = Path("reports")
