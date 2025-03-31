import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import cv2
import rasterio

DATASET_PATH_RGB = "data/raw/EuroSAT_RGB"
DATASET_PATH_MULTISPECTRAL = "data/raw/EuroSAT_MS"

IMG_SIZE = 128  # Resize images to 128x128
NUM_CLASSES = 10

CLASS_NAMES = sorted(os.listdir(DATASET_PATH_RGB))  # ["PermanentCrop", "AnnualCrop", "River", "Industrial", ...]
CLASS_TO_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def load_images_from_folder(folder, img_size=IMG_SIZE, channels=3, max_images_per_class=750, selected_bands = None):
    """Laster inn bilder og gir de numeriske etiketter basert på mappenavn.
        Setter begrensning på 750 bilder pr klasse fordi det er det vi maks får trent MS bilder på """
    images, labels = [], []

    print(f"💾 Loading from: {folder}")
    for class_name in os.listdir(folder):
        class_path = os.path.join(folder, class_name)
        if os.path.isdir(class_path):
            print(f"📂 Processing class: {class_name}")

            image_files = os.listdir(class_path)
            random.shuffle(image_files)  # Velger forskjellige bilder når vi henter fra datasettet 
            image_files = os.listdir(class_path)[:max_images_per_class] # Begrenser antall bilder vi henter ut
            for img_name in image_files:
                img_path = os.path.join(class_path, img_name)

                if channels == 3:  # RGB
                    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # Load image
                    if img is not None:
                        img = cv2.resize(img, (img_size, img_size))  # Resize
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
                else:  # Multispectral
                    with rasterio.open(img_path) as src:
                        img = src.read()  # Shape: (channels, height, width)
                        img = np.transpose(img, (1, 2, 0))  # Convert to (height, width, channels)

                        if selected_bands is not None:
                            img = np.stack([cv2.resize(img[..., i], (img_size, img_size))
                                            for i in selected_bands], axis=-1)
                        else:
                            img = np.stack([cv2.resize(img[..., i], (img_size, img_size)) 
                                for i in range(channels)], axis=-1)

                images.append(img)
                labels.append(CLASS_TO_INDEX[class_name])

    return np.array(images), np.array(labels)


def get_dataset(img_size=IMG_SIZE, data_type=" ", bands=None):
    """Laster og forbereder datasetet basert på datatype.
        | bands = 1: RGB | bands = 2: NDVI | bands = 3: SWIR |"""

    # Split RGB data
    if (data_type == "RGB"):

        # Last inn RGB bilder
        images, labels = load_images_from_folder(DATASET_PATH_RGB, IMG_SIZE, channels=3)

        # Normalser data
        images = images.astype("float32") / 255.0

    # Split MultRGBispectral data
    elif (data_type == "MS"):

        # Standard RGB bands
        if bands == 1:
            selectedBands = [1, 2, 3]

        # Vegetation Health bands
        elif bands == 2:
            selectedBands = [7, 3, 2]

        # Urban Soil Analysis bands
        elif bands == 3:
            selectedBands = [1, 7, 10]
        else:
            selectedBands = None
        # Last inn Multispectral bilder
        images, labels = load_images_from_folder(DATASET_PATH_MULTISPECTRAL, IMG_SIZE, channels=13, selected_bands=selectedBands)

        # Normalser data
        images = images.astype("float32", copy=False)  # Unngå ekstra kopi
        np.divide(images, 10000.0, out=images)  # In-place operasjon
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
