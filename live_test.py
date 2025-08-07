import sounddevice as sd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import load_model
import queue
import time

model = load_model("cnn_gunshot_model.h5")

sr = 22050
chunk_duration = 2.0
chunk_samples = int(sr * chunk_duration)
confidence_threshold = 0.9
classes = ["Gunshot", "NotGunshot"]

audio_q = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    if status:
        print("Audio Status:", status)
    audio_q.put(indata.copy())

print("🎙️ Listening for gunshots...")
stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=sr, blocksize=chunk_samples)
stream.start()

try:
    window_index = 0
    while True:
        if not audio_q.empty():
            audio_chunk = audio_q.get()
            audio_chunk = audio_chunk.flatten()

            S = librosa.feature.melspectrogram(y=audio_chunk, sr=sr, n_mels=128)
            S_DB = librosa.power_to_db(S, ref=np.max)

            resized = cv2.resize(S_DB, (128, 128))
            normalized = (resized - resized.min()) / (resized.max() - resized.min())
            image_input = np.stack([normalized]*3, axis=-1)
            image_input = np.expand_dims(image_input, axis=0)

            pred = model.predict(image_input, verbose=0)[0][0]
            predicted_class = classes[1] if pred >= confidence_threshold else classes[0]

            print(f"[Window {window_index}] Prediction: {predicted_class} (Confidence: {pred*100:.2f}%)")
            if predicted_class == "Gunshot":
                print(f"🚨 Gunshot detected in window {window_index}!")

            window_index += 1
        else:
            time.sleep(0.1)
except KeyboardInterrupt:
    print("🛑 Stopped by user.")
finally:
    stream.stop()
    stream.close()
