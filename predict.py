"""Predict the genre of one song with chunk-level majority voting."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from config import DEFAULT_CHECKPOINT_DIR, AudioConfig


def predict_file(audio_path: Path, checkpoint_path: Path, device_name: str | None = None) -> dict:
    import torch

    from audio import audio_to_mels, mel_to_image
    from dataset import image_to_tensor
    from model import build_resnet18

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file does not exist: {audio_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    classes = checkpoint["classes"]
    config = AudioConfig.from_dict(checkpoint.get("audio_config"))
    image_size = int(checkpoint.get("image_size", 224))

    model = build_resnet18(num_classes=len(classes), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    mels = audio_to_mels(audio_path, config)
    if not mels:
        raise RuntimeError(f"Audio is shorter than one {config.segment_seconds}s segment: {audio_path}")

    tensors = [image_to_tensor(mel_to_image(mel), image_size) for mel in mels]
    batch = torch.stack(tensors).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(batch), dim=1)

    predicted_indices = probs.argmax(dim=1).cpu().tolist()
    votes = Counter(predicted_indices)
    winner_index, winner_votes = votes.most_common(1)[0]
    mean_confidence = probs[:, winner_index].mean().item()

    return {
        "genre": classes[winner_index],
        "confidence": mean_confidence,
        "chunks": len(mels),
        "vote_share": winner_votes / len(mels),
        "votes": {classes[index]: count for index, count in votes.items()},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio_file", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_DIR / "best_model.pt")
    parser.add_argument("--device", default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        result = predict_file(args.audio_file, args.checkpoint, args.device)
    except (FileNotFoundError, RuntimeError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
    print(f"Predicted genre: {result['genre']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Vote share: {result['vote_share']:.2%} across {result['chunks']} chunks")
    print(f"Votes: {result['votes']}")


if __name__ == "__main__":
    main()
