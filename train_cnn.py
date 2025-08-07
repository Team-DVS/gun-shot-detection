import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os

# Constants
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 15  # Increased for better training

# Check class balance and structure:
# spectrograms/
# ├── Gunshot/
# └── NotGunshot/

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    horizontal_flip=True,     # New: Data Augmentation
    zoom_range=0.1,
    rotation_range=5
)

train_generator = train_datagen.flow_from_directory(
    'spectrograms',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',  # Binary classification
    subset='training',
    shuffle=True
)

val_generator = train_datagen.flow_from_directory(
    'spectrograms',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

# Define model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary output
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Train
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)

model.save('cnn_gunshot_model.h5')
print("✅ Model trained and saved as cnn_gunshot_model.h5")
