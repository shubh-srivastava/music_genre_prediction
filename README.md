# Music Genre Prediction

PyTorch music genre classifier that converts songs into Mel-spectrogram images and trains a CNN on those images.

The folders inside `data_raw/` are the labels. For example, songs in `data_raw/bollypop/` are labeled `bollypop`, songs in `data_raw/carnatic/` are labeled `carnatic`, and so on.

The pipeline avoids data leakage: songs are split into train/validation/test sets before they are sliced into 3-second spectrogram images, so chunks from the same song never appear in multiple splits.

## Project Structure

```text
prepare_audio_data.py           # Trim/split raw songs into data_processed/
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

Place songs directly under genre folders in `data_raw/`:

```text
data_raw/
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

Prepare balanced processed audio. This reads `data_raw/`, trims or splits songs to keep each genre near 300 minutes, renames the clips, and writes them to `data_processed/`:

```bash
python prepare_audio_data.py --source-dir data_raw --output-dir data_processed --overwrite
```

Preprocess the processed audio into PNG Mel-spectrogram images:

```bash
python preprocess.py --audio-root data_processed --output-dir spectrograms --overwrite
```

Train the CNN:

```bash
python train.py --output-dir spectrograms --epochs 20 --batch-size 32
```

Training uses validation-loss early stopping by default. It stops after 5 epochs without a meaningful validation loss reduction. You can tune it:

```bash
python train.py --output-dir spectrograms --epochs 50 --early-stopping-patience 7 --min-delta 0.001
```

For CPU training on a 12-thread machine, keep two threads free for system responsiveness and use 10 training threads:

```powershell
$env:OMP_NUM_THREADS=10
$env:MKL_NUM_THREADS=10
$env:OPENBLAS_NUM_THREADS=10
python train.py --output-dir spectrograms --epochs 50 --batch-size 128 --num-workers 10 --torch-threads 10 --torch-interop-threads 10
```

If memory allows, try `--batch-size 256`. If loading images becomes the bottleneck, try `--num-workers 6`, `8`, or `10` and compare epoch time.

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

- `data_processed/` contains trimmed/split and renamed WAV clips used for spectrogram preprocessing.
- `spectrograms/images/` contains generated spectrogram PNG files.
- `spectrograms/manifest.csv` maps each spectrogram image to its label and split.
- `spectrograms/metadata.json` stores class names and audio preprocessing settings.
- `checkpoints/best_model.pt` stores the best trained CNN checkpoint.
- `predictions.csv` stores batch prediction results when using `predict_folder.py`.

## Notes

- Training uses ImageNet-pretrained ResNet-18 by default. The first run may download model weights through `torchvision`.
- Use `--no-pretrained` for offline smoke tests or when you explicitly want random initialization.
- The source songs remain in `data_raw/`; `prepare_audio_data.py` writes normalized audio clips to `data_processed/`.
- Classical-style folders such as `classical`, `carnatic`, `ghazal`, and `semiclassical` are split into 3-minute sections until the genre reaches about 300 minutes.
- Folders such as `bollywood`, `bollypop`, `edm`, `hiphop`, and `indian_indie` use center trimming for songs longer than 4 minutes, keeping about 3 minutes from the middle.
