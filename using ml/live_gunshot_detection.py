import sounddevice as sd
import numpy as np
import librosa
from scipy.io.wavfile import write
import joblib
import os

# === Constants ===
SAMPLE_RATE = 16000     # 16 kHz
DURATION = 4            # seconds to record
TEMP_FILENAME = "temp_live.wav"
N_MFCC = 13             # same as during training

# === Load Model and Label Encoder ===
model = joblib.load("gunshot_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# === Step 1: Record Audio from Microphone ===
def record_audio():
    print(f"🎙️ Recording {DURATION} seconds from mic...")
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    write(TEMP_FILENAME, SAMPLE_RATE, audio)
    print("✅ Recording saved temporarily.")

# === Step 2: Extract Features from Recorded Audio ===
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc_scaled = np.mean(mfcc.T, axis=0)
    return mfcc_scaled.reshape(1, -1)

# === Step 3: Predict ===
def predict_gunshot():
    features = extract_features(TEMP_FILENAME)
    prediction = model.predict(features)[0]
    label = label_encoder.inverse_transform([prediction])[0]
    print(f"\n🔊 Predicted Sound: **{label.upper()}**")

# === Step 4: Run ===
if __name__ == "__main__":
    record_audio()
    predict_gunshot()
    os.remove(TEMP_FILENAME)
