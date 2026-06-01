# Music Genre Prediction

PyTorch music genre classifier that converts songs into Mel-spectrogram images and trains a CNN on those images.

The folders inside `data/` are the labels. For example, songs in `data/bollypop/` are labeled `bollypop`, songs in `data/carnatic/` are labeled `carnatic`, and so on.

The pipeline avoids data leakage: songs are split into train/validation/test sets before they are sliced into 3-second spectrogram images, so chunks from the same song never appear in multiple splits.

## Project Structure

```text
preprocess.py                   # Convert audio to Mel-spectrogram PNG images
train.py                        # Train the ResNet-18 CNN classifier
evaluate.py                     # Evaluate a checkpoint
predict.py                      # Predict one audio file
predict_folder.py               # Predict every audio file in predict/
audio.py                        # Audio loading and Mel-spectrogram image creation
config.py                       # Shared paths and audio settings
dataset.py                      # PyTorch Dataset for generated PNG spectrograms
model.py                        # ResNet-18 CNN model factory
utils.py                        # Shared helpers
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Install the CUDA-enabled PyTorch build if you want GPU training on a compatible machine. Follow the selector at https://pytorch.org/get-started/locally/ and then install the remaining packages from `requirements.txt`.

## Dataset Layout

Place songs directly under genre folders in `data/`:

```text
data/
  bollypop/
    bp01.mp3
  carnatic/
    c01.mp3
  ghazal/
    g01.mp3
  semiclassical/
    sc01.mp3
  sufi/
    s01.mp3
```

Supported extensions: `.au`, `.flac`, `.m4a`, `.mp3`, `.ogg`, `.wav`.

## Run

Preprocess songs into PNG Mel-spectrogram images:

```bash
python preprocess.py --audio-root data --output-dir spectrograms
```

Train the CNN:

```bash
python train.py --output-dir spectrograms --epochs 20 --batch-size 32
```

Evaluate on the held-out test split:

```bash
python evaluate.py --output-dir spectrograms --checkpoint checkpoints/best_model.pt
```

Predict one song:

```bash
python predict.py path\to\song.wav --checkpoint checkpoints/best_model.pt
```

Predict every supported audio file inside the `predict/` folder and write a CSV:

```bash
python predict_folder.py --input-dir predict --checkpoint checkpoints/best_model.pt --output-csv predictions.csv
```

Run the Streamlit frontend:

```bash
streamlit run app.py
```

The frontend shows the prediction progress step by step: checkpoint loading, song slicing, spectrogram image creation, tensor conversion, CNN inference, and majority-vote aggregation.

## Generated Files

- `spectrograms/images/` contains generated spectrogram PNG files.
- `spectrograms/manifest.csv` maps each spectrogram image to its label and split.
- `spectrograms/metadata.json` stores class names and audio preprocessing settings.
- `checkpoints/best_model.pt` stores the best trained CNN checkpoint.
- `predictions.csv` stores batch prediction results when using `predict_folder.py`.

## Notes

- Training uses ImageNet-pretrained ResNet-18 by default. The first run may download model weights through `torchvision`.
- Use `--no-pretrained` for offline smoke tests or when you explicitly want random initialization.
- The source songs remain in `data/`; preprocessing writes generated images to `spectrograms/`.
