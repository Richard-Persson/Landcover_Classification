import sys
import os

# Legg til prosjektets rotmappe i sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from src.utils.data_loader import get_dataset, IMG_SIZE, NUM_CLASSES


def build_resnet_model(input_shape, num_classes):
    """Bygger en ResNet-modell for multispektrale bilder."""
    base_model = ResNet50(weights=None, include_top=False, input_shape=input_shape) #droppet imagenet da den ikke støtter mer enn 3 kanaler, b
    base_model.trainable = True

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# brightness, contrast, saturation,: Disse justerer fargebalansen, men multispektrale kanaler representerer fysiske verdier
#innfører zoom og crop, rotasjon krever en tf - addon

def processData(X, y, batch_size=8, augment=True):
    def augment(x, y):
        # Flipper
        x = tf.image.random_flip_left_right(x)
        x = tf.image.random_flip_up_down(x)

        # Random zoom, crop + resize
        scale = tf.random.uniform([], 0.8, 1.0)
        input_shape = tf.shape(x)
        crop_size = tf.cast(tf.cast(input_shape[:2], tf.float32) * scale, tf.int32)
        x = tf.image.random_crop(x, size=[crop_size[0], crop_size[1], 13])
        x = tf.image.resize(x, [IMG_SIZE, IMG_SIZE])

        return x, y
    
    dataset = tf.data.Dataset.from_tensor_slices((X, y))

    if augment:
        dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset


if __name__ == "__main__":
    save_path = "models/ResNet/landcover_resnet_multispectral.h5"

    # Last inn multispektralt datasett
    X_train, X_test, y_train, y_test = get_dataset(data_type="MS")

    # Lag tf.data.Datasets
    train_dataset = processData(X_train, y_train, batch_size=8, augment=True)
    test_dataset =  processData(X_test, y_test, batch_size=8)

    # Last inn eller tren modell
    if os.path.exists(save_path):
        print("Laster tidligere modell...")
        model = load_model(save_path)
        model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    else:
        print("Trener ny modell...")
        model = build_resnet_model((IMG_SIZE, IMG_SIZE, 13), NUM_CLASSES)
        model.fit(train_dataset, validation_data=test_dataset, epochs=10)

        # Lagre modellen
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.save(save_path)
        print(f"Modell lagret i: {save_path}")

    # Evaluer modell
    loss, accuracy = model.evaluate(test_dataset)
    print(f"Test accuracy: {accuracy:.4f}, Test loss: {loss:.4f}")
