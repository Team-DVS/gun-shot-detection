import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

DATASET_DIR = "dataset"
OUTPUT_DIR = "spectrograms"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_and_save_mel_spectrogram(file_path, output_path):
    y, sr = librosa.load(file_path, sr=22050)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    plt.figure(figsize=(3, 3))
    librosa.display.specshow(mel_spec_db, sr=sr)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

for class_folder in ['gunshot', 'not_gunshot']:
    input_folder = os.path.join(DATASET_DIR, class_folder)
    output_class_folder = os.path.join(OUTPUT_DIR, class_folder)
    os.makedirs(output_class_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith('.wav'):
            file_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_class_folder, filename.replace('.wav', '.png'))

            try:
                extract_and_save_mel_spectrogram(file_path, output_path)
                print(f"Saved: {output_path}")
            except Exception as e:
                print(f"Failed to process {file_path}: {e}")
