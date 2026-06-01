"""Streamlit frontend for CNN music genre prediction."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from config import AUDIO_EXTENSIONS, DEFAULT_CHECKPOINT_DIR
from predict import predict_file


DEFAULT_CHECKPOINT = DEFAULT_CHECKPOINT_DIR / "best_model.pt"


def save_upload_to_temp(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as fp:
        fp.write(uploaded_file.getbuffer())
        return Path(fp.name)


def probability_frame(probabilities: dict[str, float]) -> pd.DataFrame:
    rows = [
        {"genre": genre, "probability": probability}
        for genre, probability in probabilities.items()
    ]
    return pd.DataFrame(rows).sort_values("probability", ascending=False)


def main() -> None:
    st.set_page_config(
        page_title="Music Genre Prediction",
        layout="centered",
    )

    st.title("Music Genre Prediction")
    st.caption("Upload a song and classify it using the trained CNN checkpoint.")

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

    if not checkpoint_path.exists():
        st.error(f"Checkpoint does not exist: {checkpoint_path}")
        return

    if uploaded_file is None:
        st.error("Upload an audio file first.")
        return

    temp_path = save_upload_to_temp(uploaded_file)
    status = st.status("Starting prediction...", expanded=True)
    try:
        def log_step(message: str) -> None:
            status.write(message)

        result = predict_file(temp_path, checkpoint_path, progress_callback=log_step)
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
    st.write(f"Vote share: **{result['vote_share']:.2%}** across **{result['chunks']}** chunks")

    probabilities = probability_frame(result["class_probabilities"])
    st.subheader("Class probabilities")
    st.bar_chart(probabilities, x="genre", y="probability")

    table = probabilities.copy()
    table["probability"] = table["probability"].map(lambda value: f"{value:.2%}")
    st.dataframe(table, hide_index=True, use_container_width=True)


if __name__ == "__main__":
    main()
