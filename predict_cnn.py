import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from tensorflow.keras.models import load_model
import cv2

# Load trained model
model = load_model("cnn_gunshot_model.h5")

# File path to test audio
file_path = "sample/sample_1.wav"
y, sr = librosa.load(file_path)

# Display full mel spectrogram
plt.figure(figsize=(10, 4))
S_full = librosa.feature.melspectrogram(y=y, sr=sr)
S_dB_full = librosa.power_to_db(S_full, ref=np.max)
librosa.display.specshow(S_dB_full, sr=sr, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram of Full Audio')
plt.tight_layout()
plt.show()

# Parameters
chunk_duration = 2.0  # seconds
hop_duration = 1.0    # seconds
chunk_samples = int(sr * chunk_duration)
hop_samples = int(sr * hop_duration)

# Class labels for binary classifier
classes = ["NotGunshot", "Gunshot"]
confidence_threshold = 0.9  # Only classify as gunshot if confidence > 90%

# Sliding window prediction
detected = False
window_index = 0

for start in range(0, len(y) - chunk_samples + 1, hop_samples):
    end = start + chunk_samples
    y_chunk = y[start:end]

    # Mel spectrogram
    S = librosa.feature.melspectrogram(y=y_chunk, sr=sr)
    S_DB = librosa.power_to_db(S, ref=np.max)

    # Resize and normalize
    resized = cv2.resize(S_DB, (128, 128))
    normalized = (resized - resized.min()) / (resized.max() - resized.min())
    image_input = np.stack([normalized]*3, axis=-1)
    image_input = np.expand_dims(image_input, axis=0)

    # Predict (sigmoid output for binary classification)
    pred = model.predict(image_input, verbose=0)[0][0]
    predicted_class = classes[1] if pred >= confidence_threshold else classes[0]

    print(f"[Window {window_index}] Prediction: {predicted_class} (Confidence: {pred*100:.2f}%)")

    if predicted_class == "Gunshot":
        print(f"🚨 Gunshot detected in window {window_index}!")
        detected = True

    window_index += 1

if not detected:
    print("✅ No gunshot detected in any window.")
