import tensorflow as tf
from model import create_model
import os

# ─── Dataset Path ───────────────────────────────────────────────
DATASET_DIR = "dataset"
IMAGE_SIZE  = (224, 224)
BATCH_SIZE  = 16
EPOCHS      = 20

# ─── Count actual class sizes ────────────────────────────────────
fake_count = len(os.listdir(os.path.join(DATASET_DIR, "fake")))
real_count = len(os.listdir(os.path.join(DATASET_DIR, "real")))
total      = fake_count + real_count

print(f"Dataset: Fake={fake_count}, Real={real_count}, Total={total}")

# ─── Auto-calculate Class Weights ────────────────────────────────
# Formula: weight = Total / (num_classes * class_count)
# Class 0 = fake, Class 1 = real (alphabetical order)
class_weight = {
    0: total / (2 * fake_count),   # Fake weight
    1: total / (2 * real_count),   # Real weight
}
print(f"Class Weights: Fake(0)={class_weight[0]:.3f}, Real(1)={class_weight[1]:.3f}")

# ─── Load Dataset with Validation Split ──────────────────────────
train_data = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="training",
    seed=42
)

val_data = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="validation",
    seed=42
)

# ─── Train ───────────────────────────────────────────────────────
model = create_model()
model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    class_weight=class_weight
)

model.save("model/deepfake_model.h5")
print("MODEL TRAINING COMPLETE")
