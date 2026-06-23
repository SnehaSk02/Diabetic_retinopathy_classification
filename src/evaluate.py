import numpy as np

from tensorflow.keras.models import load_model

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
X_test = np.load("data/X_test.npy")
y_test = np.load("data/y_test.npy")
# ============================================
# LOAD MODEL
# ============================================

model = load_model("models/best_model.h5")

# ============================================
# PREDICT
# ============================================

y_pred = model.predict(X_test)

y_pred_classes = np.argmax(y_pred, axis=1)

# ============================================
# ACCURACY
# ============================================

accuracy = accuracy_score(
    y_test,
    y_pred_classes
)

print(f"\nAccuracy: {accuracy:.4f}")

# ============================================
# CLASSIFICATION REPORT
# ============================================

class_names = [
    "Mild",
    "Moderate",
    "No_DR",
    "Proliferate_DR",
    "Severe"
]

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred_classes,
        target_names=class_names
    )
)

# ============================================
# CONFUSION MATRIX
# ============================================

cm = confusion_matrix(
    y_test,
    y_pred_classes
)

print("\nConfusion Matrix:\n")

print(cm)