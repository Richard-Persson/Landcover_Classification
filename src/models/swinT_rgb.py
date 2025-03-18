# fungerer ikke, nekter å ta imot pikselverdier

import sys
import os

# Legg til prosjektets rotmappe i sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import tensorflow as tf
from transformers import TFSwinModel, SwinConfig, AutoImageProcessor
from tensorflow.keras.layers import Dense, GlobalAveragePooling1D, Dropout, Input, Lambda # We could also use `Conv2d` layer to do the same as a 2-D Convolution uses kernel size (which in our case will be the patch size) and the stride also will be equal to the patch size as we want the windows to not be overlapping.  
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src.utils.data_loader import get_dataset, IMG_SIZE, NUM_CLASSES
MODEL_NAME = "microsoft/swin-tiny-patch4-window7-224"

# Last inn Swin Transformer-modell
config = SwinConfig.from_pretrained(MODEL_NAME)
config.image_size = IMG_SIZE  # Tilpasset bildet ditt

base_model = TFSwinModel.from_pretrained(MODEL_NAME, config=config)
base_model.trainable = True  # Tillater finjustering

# Image processor for Swin Transformer
image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

def preprocess_images_tf(inputs):
    """Forhåndsprosesserer bilder i en TensorFlow-kompatibel Lambda-funksjon."""
    return image_processor(inputs, return_tensors="tf")["pixel_values"]

# Definer input
inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
preprocessed_inputs = Lambda(preprocess_images_tf)(inputs)

# Swin Transformer krever batch-dimensjon
swin_outputs = base_model(pixel_values=preprocessed_inputs, training=True).last_hidden_state

# Global gjennomsnittspooling
x = GlobalAveragePooling1D()(swin_outputs)

# Klassifikasjonslag
outputs = Dense(NUM_CLASSES, activation='softmax')(x)

# Definer modell
model = Model(inputs, outputs)

# Kompiler modell
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Vis modelloversikt
model.summary()
