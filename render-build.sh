#!/usr/bin/env bash
# exit on error
set -o errexit

echo "Installing dependencies..."
python -m pip install --upgrade pip
pip install -r requirements.txt

echo "Downloading models from Google Drive..."
mkdir -p model

# Download audio_model.pkl
if [ ! -f "model/audio_model.pkl" ]; then
    echo "Downloading audio_model.pkl..."
    gdown 1TdxCG5qOlElFiNtu85XzBp88uY5HnU0V -O model/audio_model.pkl
fi

# Download deepfake_model_94acc.h5 (Restored for high accuracy)
if [ ! -f "model/deepfake_model_94acc.h5" ]; then
    echo "Downloading deepfake_model_94acc.h5..."
    gdown 1ny15DmXESVvnmpas2nQdvrkfDUcZUv7j -O model/deepfake_model_94acc.h5
fi

echo "Build complete."
