import sounddevice as sd
import numpy as np
import librosa
from scipy.io.wavfile import write
import joblib
import os
import time

SAMPLE_RATE = 16000
DURATION = 4
TEMP_FILE = "live_audio.wav"
N_MFCC = 13
SLEEP_GAP = 0.5

model = joblib.load("gunshot_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

def record_audio(filename):
    print("🎙️ Listening...")
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    write(filename, SAMPLE_RATE, audio)

def extract_features(filename):
    y, sr = librosa.load(filename, sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc_scaled = np.mean(mfcc.T, axis=0)
    return mfcc_scaled.reshape(1, -1)

def detect_gunshot():
    record_audio(TEMP_FILE)
    features = extract_features(TEMP_FILE)
    prediction = model.predict(features)[0]
    label = label_encoder.inverse_transform([prediction])[0]
    print(f"🔊 Detected: {label.upper()}")
    os.remove(TEMP_FILE)

if __name__ == "__main__":
    print("🚨 Gunshot Detection Started (Press Ctrl+C to stop)")
    try:
        while True:
            detect_gunshot()
            time.sleep(SLEEP_GAP)
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")
