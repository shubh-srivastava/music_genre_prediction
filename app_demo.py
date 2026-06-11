"""Streamlit demo for one-spectrogram-per-song genre prediction."""

from __future__ import annotations

import tempfile
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

from config import AUDIO_EXTENSIONS, AudioConfig
from dataset import image_to_tensor
from model import build_resnet18


DEFAULT_CHECKPOINT = Path("checkpoints_1spec_per_1song") / "best_model.pt"


def save_upload_to_temp(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as fp:
        fp.write(uploaded_file.getbuffer())
        return Path(fp.name)


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


def probability_frame(probabilities: dict[str, float]) -> pd.DataFrame:
    rows = [
        {"genre": genre, "probability": probability}
        for genre, probability in probabilities.items()
    ]
    return pd.DataFrame(rows).sort_values("probability", ascending=False)


def predict_one_spectrogram(audio_path: Path, checkpoint_path: Path, progress_callback=None) -> dict:
    def log(message: str) -> None:
        if progress_callback is not None:
            progress_callback(message)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    classes = checkpoint["classes"]
    image_size = int(checkpoint.get("image_size", 224))
    config = AudioConfig.from_dict(checkpoint.get("audio_config"))

    log("Creating one full-song Mel-spectrogram image")
    spectrogram_image, duration_seconds = song_to_spectrogram_image(audio_path, config, image_size)

    log(f"Converting spectrogram to {image_size}x{image_size} tensor")
    tensor = image_to_tensor(spectrogram_image, image_size).unsqueeze(0).to(device)

    log(f"Building CNN model on device: {device}")
    model = build_resnet18(num_classes=len(classes), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    log("Running the one spectrogram image through the CNN")
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).squeeze(0).cpu().tolist()

    winner_index = int(np.argmax(probs))
    probabilities = {
        class_name: float(probs[index])
        for index, class_name in enumerate(classes)
    }

    return {
        "genre": classes[winner_index],
        "confidence": float(probs[winner_index]),
        "duration_seconds": duration_seconds,
        "spectrogram_image": spectrogram_image,
        "class_probabilities": probabilities,
    }


def main() -> None:
    st.set_page_config(
        page_title="Music Genre Prediction Demo",
        layout="centered",
    )

    st.title("Music Genre Prediction Demo")
    st.caption("Upload a song and classify it with the one-song/one-spectrogram CNN.")

    checkpoint_path = Path(
        st.text_input("Checkpoint path", value=str(DEFAULT_CHECKPOINT))
    )
    uploaded_file = st.file_uploader(
        "Upload an audio file",
        type=[extension.lstrip(".") for extension in sorted(AUDIO_EXTENSIONS)],
    )

    if uploaded_file is not None:
        st.audio(uploaded_file)

    predict_clicked = st.button(
        "Predict genre",
        type="primary",
        disabled=uploaded_file is None,
    )

    if not predict_clicked:
        return

    if uploaded_file is None:
        st.error("Upload an audio file first.")
        return

    temp_path = save_upload_to_temp(uploaded_file)
    status = st.status("Starting one-spectrogram prediction...", expanded=True)
    try:
        def log_step(message: str) -> None:
            status.write(message)

        result = predict_one_spectrogram(temp_path, checkpoint_path, progress_callback=log_step)
        status.update(label="Prediction complete", state="complete", expanded=False)
    except Exception as exc:
        status.update(label="Prediction failed", state="error", expanded=True)
        st.error(f"Prediction failed: {exc}")
        return
    finally:
        if temp_path.exists():
            temp_path.unlink()

    st.subheader("Prediction")
    st.metric("Predicted genre", result["genre"])
    st.write(f"Confidence: **{result['confidence']:.2%}**")
    st.write(f"Song duration: **{result['duration_seconds'] / 60.0:.2f} minutes**")

    st.subheader("Generated full-song spectrogram")
    st.image(result["spectrogram_image"], use_container_width=True)

    probabilities = probability_frame(result["class_probabilities"])
    st.subheader("Class probabilities")
    st.bar_chart(probabilities, x="genre", y="probability")

    table = probabilities.copy()
    table["probability"] = table["probability"].map(lambda value: f"{value:.2%}")
    st.dataframe(table, hide_index=True, use_container_width=True)


if __name__ == "__main__":
    main()
