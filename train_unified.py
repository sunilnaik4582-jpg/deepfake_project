import os
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import random

# ─── CONFIGURATION ────────────────────────────────────────────────
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
MAX_IMAGES_PER_CLASS = 4000 

MODEL_SAVE_PATH = "model/unified_model.h5"

# ─── DATA LOADING LOGIC (Same as before) ──────────────────────────
def get_image_paths(directory):
    paths = []
    if not os.path.exists(directory): return paths
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                paths.append(os.path.join(root, file))
    return paths

print("[*] Gathering dataset paths...")
ai_paths = get_image_paths("dataset/ai_fake/AI-Generated Images") + get_image_paths("dataset/ai_fake/Human Faces Dataset/AI-Generated Images")
df_paths = get_image_paths("dataset/fake")
real_paths = get_image_paths("dataset/real") + get_image_paths("dataset/ai_fake/Human Faces Dataset/Real Images")

random.shuffle(ai_paths); ai_final = ai_paths[:MAX_IMAGES_PER_CLASS]
random.shuffle(df_paths); df_final = df_paths[:MAX_IMAGES_PER_CLASS]
random.shuffle(real_paths); real_final = real_paths[:MAX_IMAGES_PER_CLASS]

all_paths = ai_final + df_final + real_final
all_labels = ([0] * len(ai_final)) + ([1] * len(df_final)) + ([2] * len(real_final))

temp_zip = list(zip(all_paths, all_labels))
random.shuffle(temp_zip)
all_paths, all_labels = zip(*temp_zip)

train_paths, val_paths, train_labels, val_labels = train_test_split(
    list(all_paths), list(all_labels), test_size=0.2, random_state=42, stratify=all_labels
)

def process_path(file_path, label):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = img / 255.0
    return img, label

def configure_dataset(paths, labels, batch_size, shuffle_data=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle_data: ds = ds.shuffle(buffer_size=1000)
    ds = ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds

train_ds = configure_dataset(train_paths, train_labels, BATCH_SIZE, shuffle_data=True)
val_ds = configure_dataset(val_paths, val_labels, BATCH_SIZE, shuffle_data=False)

# ─── BUILD CUSTOM CNN MODEL (Old Architecture but 3 Classes) ──────
# This architecture was better at catching deepfake artifacts
def build_custom_cnn():
    print("\n[*] Building Custom CNN (Artifact-Sensitive Architecture)...")
    model = Sequential([
        # Input layer
        tf.keras.layers.Input(shape=(224, 224, 3)),
        
        Conv2D(32, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(),
        
        Conv2D(64, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(),
        
        Conv2D(128, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(),
        
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        # 3 output neurons for AI, Deepfake, Real
        Dense(3, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

if __name__ == "__main__":
    model = build_custom_cnn()
    os.makedirs("model", exist_ok=True)
    
    callbacks = [
        ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_accuracy', patience=6, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    ]
    
    print("\n[START] Training Super-Unified Model (V2)...")
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)
    
    print("\n[DONE] New Model saved at:", MODEL_SAVE_PATH)
