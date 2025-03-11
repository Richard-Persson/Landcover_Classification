import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import cv2

DATASET_PATH_RGB = "data/raw/EuroSAT_RGB"
DATASET_PATH_MULTISPECTRAL = "data/raw/EuroSAT_MS"

IMG_SIZE = 128  # Resize images to 128x128
NUM_CLASSES = 10  # Adjust based on dataset

CLASS_NAMES = sorted(os.listdir(DATASET_PATH_RGB))  # e.g., ["Forest", "Urban", "Water", ...]
CLASS_TO_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def load_images_from_folder(folder, img_size=IMG_SIZE, channels=3):
    """Loads images and assigns numeric labels based on folder names."""
    images, labels = [], []

    for class_name in os.listdir(folder):
        class_path = os.path.join(folder, class_name)
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # Load image
                if img is not None:
                    img = cv2.resize(img, (img_size, img_size))  # Resize
                    if channels == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
                    images.append(img)
                    labels.append(CLASS_TO_INDEX[class_name])  # Convert class name to index

    return np.array(images), np.array(labels)


def get_dataset(img_size=IMG_SIZE, data_type=" "):
    """Loads and prepares the dataset basert på datatype."""

    # Split RGB data
    if (data_type == "RGB"):

        # Last inn RGB bilder
        images, labels = load_images_from_folder(DATASET_PATH_RGB, IMG_SIZE, channels=3)

        # Normalser data
        images = images.astype("float32") / 255.0

    # Split Multispectral data
    elif (data_type == "MS"):

        # Last inn Multispectral bilder
        images, labels = load_images_from_folder(DATASET_PATH_MULTISPECTRAL, IMG_SIZE, channels=13)

        # Normalser data
        images = images.astype("float32") / 10000.0  # Endre denne basert på MS rekkevidde
    else:
        raise ValueError('Invalid data type')

    # One-hot encode etikettene
    labels = to_categorical(labels, NUM_CLASSES)

    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_dataset()
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
