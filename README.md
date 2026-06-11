# Music Genre Prediction

This project classifies songs into music genres using a CNN trained on Mel-spectrogram images. Raw songs are first normalized into a balanced processed dataset, then converted into spectrogram PNG images, and finally used to train a ResNet-18 classifier.

The raw dataset is expected to use folder names as labels:

```text
data_raw/
  bollywood_new/
  bollywood_old/
  classical/
  edm/
  ghazhal/
  hiphop/
  indian_indie/
  punjabi/
```

## Project Files

```text
prepare_audio_data.py    # Trim/split raw songs into data_processed/
preprocess.py            # Convert processed audio to Mel-spectrogram PNG images
train.py                 # Train the CNN model
evaluate.py              # Evaluate a checkpoint and save a confusion matrix
predict.py               # Predict one audio file
predict_folder.py        # Predict all songs inside predict/
app.py                   # Streamlit upload frontend

audio.py                 # Audio loading and spectrogram conversion helpers
config.py                # Shared paths and audio settings
dataset.py               # PyTorch Dataset for spectrogram images
model.py                 # ResNet-18 model factory
utils.py                 # Shared utility functions
requirements.txt         # Python dependencies
```

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If you want GPU training, install the CUDA-enabled PyTorch build that matches your system from the official PyTorch install selector, then install the rest of the requirements.

## Data Preparation

The project keeps raw data and generated data separate.

```text
data_raw/         # original audio dataset, ignored by git
data_processed/   # trimmed/split WAV clips, ignored by git
spectrograms/     # generated PNG spectrogram images, ignored by git
checkpoints/      # trained model checkpoints, can be committed
```

Prepare fixed 3-minute audio clips:

```powershell
python prepare_audio_data.py --source-dir data_raw --output-dir data_processed --overwrite
```

This step follows two preparation rules:

- `edm`, `hiphop`, `indian_indie`, `punjabi`, `bollywood_new`, and `bollywood_old`: create one centered 3-minute WAV clip from each source file.
- `classical` and `ghazhal`: split source audio into exactly 100 sequential 3-minute WAV clips.

Other folders are skipped unless a policy is added in `prepare_audio_data.py`.

Convert processed audio into Mel-spectrogram images:

```powershell
python preprocess.py --audio-root data_processed --output-dir spectrograms --overwrite
```

This writes:

```text
spectrograms/images/
spectrograms/manifest.csv
spectrograms/metadata.json
```

The manifest maps every spectrogram image to its label, split, source song id, and chunk index.

## Training

## One Spectrogram Per Song Experiment

If the chunk-level model is not performing well, use this alternate pipeline.
It creates one full-song Mel-spectrogram image for each song in `data_raw/`,
then trains and validates a separate model from those images.

Create one spectrogram image per raw song:

```powershell
python preprocess_1spec_per_1song.py --audio-root data_raw --output-dir spectrograms_1spec_per_1song --overwrite
```

Train the alternate model:

```powershell
python train_1spec_per_1song.py --output-dir spectrograms_1spec_per_1song --epochs 50 --early-stopping-patience 10 --min-delta 0.001
```

Validate or test the alternate model:

```powershell
python validate_1spec_per_1song.py --output-dir spectrograms_1spec_per_1song --checkpoint checkpoints_1spec_per_1song/best_model.pt --split test
```

This keeps all artifacts separate from the existing chunk-based pipeline:

```text
spectrograms_1spec_per_1song/
checkpoints_1spec_per_1song/
reports/training_log_1spec_per_1song.csv
```

Train the CNN:

```powershell
python train.py --output-dir spectrograms --epochs 50 --batch-size 128
```

For a 12-thread CPU, keep two threads free for system responsiveness:

```powershell
$env:OMP_NUM_THREADS=10
$env:MKL_NUM_THREADS=10
$env:OPENBLAS_NUM_THREADS=10

python train.py --output-dir spectrograms --epochs 50 --batch-size 128 --num-workers 10 --torch-threads 10 --torch-interop-threads 10
```

If memory allows, try `--batch-size 256`. If image loading is the bottleneck, compare `--num-workers 6`, `8`, and `10`.

Training uses validation-loss early stopping by default:

```powershell
python train.py --output-dir spectrograms --epochs 50 --early-stopping-patience 7 --min-delta 0.001
```

Disable early stopping with:

```powershell
python train.py --output-dir spectrograms --epochs 50 --early-stopping-patience 0
```

The best model is saved to:

```text
checkpoints/best_model.pt
```

## Evaluation

Evaluate the held-out test split:

```powershell
python evaluate.py --output-dir spectrograms --checkpoint checkpoints/best_model.pt
```

This prints a classification report and writes a confusion matrix to `reports/`.

## Prediction

Predict one song:

```powershell
python predict.py path\to\song.mp3 --checkpoint checkpoints\best_model.pt
```

The script prints:

- predicted genre
- confidence
- chunk vote share
- votes by class
- probabilities for every class

Predict all songs in the `predict/` folder:

```powershell
python predict_folder.py --input-dir predict --checkpoint checkpoints\best_model.pt --output-csv predictions.csv
```

This prints results for every supported audio file and writes `predictions.csv`.

## Streamlit Frontend

Run the upload UI:

```powershell
streamlit run app.py
```

The frontend lets you upload a song and shows each prediction step: checkpoint loading, song slicing, spectrogram image creation, tensor conversion, CNN inference, and majority-vote aggregation. It then displays the predicted genre and per-class probabilities.

## Model

The classifier uses ResNet-18 from `torchvision`. ResNet-18 is a CNN originally designed for image classification. This project converts audio into spectrogram images, so the genre task becomes an image classification task. The model uses transfer learning by default and replaces the final classification layer with one output per music genre.

Use `--no-pretrained` if you want to train without pretrained ImageNet weights.

## Git Policy

Data files are intentionally ignored:

- `data/`
- `data_raw/`
- `data_processed/`
- `spectrograms/`
- `predict/`
- `reports/`
- audio files such as `.mp3`, `.wav`, `.flac`, `.ogg`, `.m4a`, `.au`

Model checkpoints are not ignored, so `checkpoints/best_model.pt` can be committed when you want to include a trained model in the repository.
