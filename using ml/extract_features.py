import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

# Paths
DATASET_PATH = "dataset"
CLASSES = ["gunshot", "not_gunshot"]
FEATURES_CSV = "features.csv"

# Parameters
SR = 16000
DURATION = 4  # seconds
N_MFCC = 13

# Storage
features = []
labels = []

# Loop over each class
for label in CLASSES:
    folder = os.path.join(DATASET_PATH, label)
    for filename in tqdm(os.listdir(folder), desc=f"Processing {label}"):
        file_path = os.path.join(folder, filename)
        try:
            y, _ = librosa.load(file_path, sr=SR, duration=DURATION)
            mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC)
            mfcc_mean = np.mean(mfcc.T, axis=0)  # average over time
            features.append(mfcc_mean)
            labels.append(label)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

# Create DataFrame
df = pd.DataFrame(features)
df['label'] = labels

# Save to CSV
df.to_csv(FEATURES_CSV, index=False)
print(f"✅ Features saved to {FEATURES_CSV}")
