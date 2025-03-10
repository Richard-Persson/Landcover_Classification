import os
import sys
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.utils.data_loader import get_dataset, IMG_SIZE, NUM_CLASSES


def build_model(input_shape, num_classes):
    """Bygger en sekvensiell CNN modell"""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


if __name__ == "__main__":

    # Definerer trening og test set med get_dataset
    X_train, X_test, y_train, y_test = get_dataset(data_type="RGB")

    model = build_model((IMG_SIZE, IMG_SIZE, 3), NUM_CLASSES)
    model.summary()

    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

    y_pred = model.predict(X_test)

    np.save("models/CNN/y_true_rgb.npy", y_test)
    np.save("models/CNN/y_pred_rgb.npy", y_pred)

    model.save("models/CNN/landcover_cnn_rgb.h5")
