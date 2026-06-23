import os
import cv2
import numpy as np

from tensorflow.keras.applications.densenet import preprocess_input

from preprocessing import crop_black
from preprocessing import apply_clahe


def load_dataset(dataset_path="data\\colored_images", img_size=224):

    classes = [
        "Mild",
        "Moderate",
        "No_DR",
        "Proliferate_DR",
        "Severe"
    ]

    images = []
    labels = []

    for label, class_name in enumerate(classes):

        class_folder = os.path.join(dataset_path, class_name)

        print("Loading:", class_name)

        for img_name in os.listdir(class_folder):

            img_path = os.path.join(class_folder, img_name)

            img = cv2.imread(img_path)

            if img is not None:

                # BGR → RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Preprocessing
                img = crop_black(img)
                img = apply_clahe(img)

                # Resize
                img = cv2.resize(img, (img_size, img_size))

                # DenseNet preprocessing
                img = preprocess_input(img)

                images.append(img)
                labels.append(label)

    return np.array(images), np.array(labels) 