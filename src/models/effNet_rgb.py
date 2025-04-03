import sys
import os

# Legg til prosjektets rotmappe i sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import EfficientNetB3, preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.utils.data_loader import get_dataset, IMG_SIZE, NUM_CLASSES


def build_efficientnet_model(input_shape, num_classes):
    """Bygger en EfficientNetB3-modell for satellittklassifisering med RGB-bilder."""
    base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=input_shape) #startet med b0 må gå over til b3 siden b0 overfitter for mye
    base_model.trainable = True  # Sett til False for feature extraction kun

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    return model


if __name__ == "__main__":
    save_path = "models/EfficientNet/landcover_effnet_rgb.h5"

    # Last inn RGB-datasettet
    X_train, X_test, y_train, y_test = get_dataset(data_type="RGB")

    X_train = preprocess_input(X_train) #modellen viste tegn på overfitting, med høye tall på trening og lave ellers. 
    X_test = preprocess_input(X_test)

    # Data Augmentation for satellittbilder
    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.2
    )

    train_generator = datagen.flow(X_train, y_train, batch_size=8)

    # Sjekk om modellen allerede finnes
    if os.path.exists(save_path):
        print("Laster inn tidligere trent EfficientNet-modell...")
        model = load_model(save_path)
        model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    else:
        print("Trener en ny EfficientNet-modell...")
        model = build_efficientnet_model((IMG_SIZE, IMG_SIZE, 3), NUM_CLASSES)
        model.fit(train_generator, validation_data=(X_test, y_test), epochs=20, batch_size=16)

        # Lagre den nye modellen
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.save(save_path)
        print(f"Modellen er lagret i: {save_path}")

    # Evaluer modellen
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {accuracy:.4f}, Test loss: {loss:.4f}")
