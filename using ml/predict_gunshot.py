import librosa
import numpy as np
import joblib
import sounddevice as sd
from scipy.io.wavfile import write
import os

# Constants
SR = 16000
DURATION = 4
N_MFCC = 13
TEMP_FILE = "temp.wav"

# Load model and label encoder
model = joblib.load("gunshot_model.pkl")
encoder = joblib.load("label_encoder.pkl")

# Record from mic
def record_audio():
    print(f"🎙️ Recording {DURATION} seconds of audio...")
    recording = sd.rec(int(DURATION * SR), samplerate=SR, channels=1)
    sd.wait()
    write(TEMP_FILE, SR, recording)
    print("✅ Recording complete")

# Extract MFCCs from audio file
def extract_features(file_path):
    y, _ = librosa.load(file_path, sr=SR, duration=DURATION)
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    return mfcc_mean.reshape(1, -1)

# Predict
def predict(file_path):
    features = extract_features(file_path)
    prediction = model.predict(features)[0]
    label = encoder.inverse_transform([prediction])[0]
    print(f"🔊 Predicted: {label.upper()}")

# Main
if __name__ == "__main__":
    print("Choose input method:")
    print("1. 🎙️ Record from microphone")
    print("2. 📂 Use existing audio file")
    choice = input("Enter 1 or 2: ")

    if choice == "1":
        record_audio()
        predict(TEMP_FILE)
        os.remove(TEMP_FILE)
    elif choice == "2":
        path = input("Enter path to .wav file: ")
        predict(path)
    else:
        print("❌ Invalid choice.")
