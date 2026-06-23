import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping
)

from dataset import load_dataset
from model import build_model


# ============================================
# LOAD DATASET
# ============================================

print("Loading dataset...")

X, y = load_dataset()

print("Dataset Loaded")
print("X Shape:", X.shape)
print("y Shape:", y.shape)


# ============================================
# TRAIN TEST SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42,

    stratify=y
)

print("Train Shape:", X_train.shape)
print("Test Shape:", X_test.shape)

np.save("data/X_test.npy", X_test)
np.save("data/y_test.npy", y_test)
# ============================================
# CLASS WEIGHTS
# ============================================

class_weights = compute_class_weight(

    class_weight='balanced',

    classes=np.unique(y_train),

    y=y_train
)

class_weights = dict(
    enumerate(class_weights)
)

print("Class Weights:")
print(class_weights)


# ============================================
# DATA AUGMENTATION
# ============================================

datagen = tf.keras.preprocessing.image.ImageDataGenerator(

    rotation_range=10,

    width_shift_range=0.05,

    height_shift_range=0.05,

    zoom_range=0.1,

    brightness_range=[0.9, 1.1],

    horizontal_flip=True
)

datagen.fit(X_train)


# ============================================
# BUILD MODEL
# ============================================

model = build_model()

model.summary()


# ============================================
# CALLBACKS
# ============================================

os.makedirs("models", exist_ok=True)

checkpoint = ModelCheckpoint(

    "models/best_model.h5",

    monitor="val_accuracy",

    save_best_only=True,

    mode="max",

    verbose=1
)

earlystop = EarlyStopping(

    monitor="val_accuracy",

    patience=5,

    restore_best_weights=True,

    verbose=1
)


# ============================================
# TRAIN MODEL
# ============================================

history = model.fit(

    datagen.flow(
        X_train,
        y_train,
        batch_size=16
    ),

    validation_data=(X_test, y_test),

    epochs=15,

    class_weight=class_weights,

    callbacks=[
        checkpoint,
        earlystop
    ]
)


# Plot Accuracy
plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title("Model Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend(['Train','Validation'])

plt.show()
plt.savefig("model_accuracy.png", bbox_inches="tight")
# Plot Loss
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title("Model Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend(['Train','Validation'])

plt.show()
plt.savefig("model_loss.png", bbox_inches="tight")
# ============================================
# SAVE FINAL MODEL
# ============================================

model.save(
    "models/final_model.h5"
)

print("Training Completed")