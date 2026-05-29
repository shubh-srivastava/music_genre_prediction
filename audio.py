"""Audio loading and Mel-spectrogram image helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from config import AudioConfig


def load_audio(path: Path, config: AudioConfig) -> np.ndarray:
    import librosa

    signal, _ = librosa.load(path, sr=config.sample_rate, mono=True)
    return signal.astype(np.float32, copy=False)


def iter_segments(signal: np.ndarray, config: AudioConfig) -> list[np.ndarray]:
    segment_len = config.segment_samples
    usable_len = (len(signal) // segment_len) * segment_len
    if usable_len == 0:
        return []
    return [
        signal[start : start + segment_len]
        for start in range(0, usable_len, segment_len)
    ]


def mel_spectrogram(segment: np.ndarray, config: AudioConfig) -> np.ndarray:
    import librosa

    mel = librosa.feature.melspectrogram(
        y=segment,
        sr=config.sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        n_mels=config.n_mels,
        fmin=config.fmin,
        fmax=config.fmax,
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max, top_db=config.top_db)
    normalized = (mel_db + config.top_db) / config.top_db
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def audio_to_mels(path: Path, config: AudioConfig) -> list[np.ndarray]:
    signal = load_audio(path, config)
    return [mel_spectrogram(segment, config) for segment in iter_segments(signal, config)]


def mel_to_image(mel: np.ndarray) -> Image.Image:
    image_array = (np.flipud(mel) * 255).astype(np.uint8)
    return Image.fromarray(image_array, mode="L").convert("RGB")


def save_spectrogram_image(mel: np.ndarray, path: Path) -> None:
    """Save a normalized Mel-spectrogram as an RGB PNG image."""
    mel_to_image(mel).save(path)
