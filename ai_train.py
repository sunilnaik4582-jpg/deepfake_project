import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

# ----- CONFIGURATION -----
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 5  # Increase this to 10 or 20 when you have more images

REAL_DIR = "dataset/real"
AI_FAKE_DIR = "dataset/AI-Generated Images"
MODEL_SAVE_PATH = "model/ai_model.h5"

def load_data():
    X = []
    y = []

    print(f"Loading REAL images from {REAL_DIR}...")
    if os.path.exists(REAL_DIR):
        for img_name in os.listdir(REAL_DIR):
            img_path = os.path.join(REAL_DIR, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, IMAGE_SIZE)
                X.append(img)
                y.append(1)  # 1 = REAL
    else:
        print(f"Directory {REAL_DIR} not found.")

    print(f"Loading AI FAKE images from {AI_FAKE_DIR}...")
    if os.path.exists(AI_FAKE_DIR):
        for img_name in os.listdir(AI_FAKE_DIR):
            img_path = os.path.join(AI_FAKE_DIR, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, IMAGE_SIZE)
                X.append(img)
                y.append(0)  # 0 = AI FAKE
    else:
        print(f"Directory {AI_FAKE_DIR} not found.")

    if len(X) == 0:
        raise ValueError("No images loaded! Please check your dataset folders.")

    X = np.array(X, dtype="float32") / 255.0  # Normalize to [0, 1]
    y = np.array(y, dtype="float32")

    print(f"Total dataset size: {len(X)} images")
    return X, y

def build_model():
    print("Building MobileNetV2 Transfer Learning Model...")
    # Load Pre-trained MobileNetV2
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    
    # Freeze the base model
    base_model.trainable = False 
    
    # Add custom classification layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(optimizer='adam', 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    return model

if __name__ == "__main__":
    X, y = load_data()
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = build_model()
    
    print("\nStarting Training...")
    model.fit(X_train, y_train, 
              validation_data=(X_val, y_val), 
              epochs=EPOCHS, 
              batch_size=BATCH_SIZE)
    
    model.save(MODEL_SAVE_PATH)
    print(f"\n✅ Training Complete. Model saved at {MODEL_SAVE_PATH}")
