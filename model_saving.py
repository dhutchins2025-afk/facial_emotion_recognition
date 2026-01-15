"""
model_saving.py
Load MobileNetV2 pretrained model and add classification head for emotion detection
This script creates the emotion_model.h5 file required for the Flask application
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np


def build_emotion_model(model_name='emotion_model.h5'):
    """
    Build a transfer learning model using MobileNetV2 pretrained on ImageNet
    for emotion classification (7 emotions)

    Args:
        model_name (str): Name to save the model

    Returns:
        model: Compiled Keras model
    """

    print("Building emotion detection model...")

    # Load pretrained MobileNetV2 (without top classification layer)
    base_model = keras.applications.MobileNetV2(
        input_shape=(48, 48, 1),
        include_top=False,
        weights='imagenet'
    )

    # Freeze base model weights (transfer learning)
    base_model.trainable = False

    # Build custom classification head
    model = models.Sequential([
        # Resize input to proper dimensions for MobileNetV2
        layers.Resizing(224, 224),

        # Convert grayscale to RGB (MobileNetV2 expects 3 channels)
        layers.experimental.preprocessing.RandomContrast(0.1),
        layers.Lambda(lambda x: tf.image.grayscale_to_rgb(x)
                      if x.shape[-1] == 1 else x),

        # Preprocessing for MobileNetV2
        layers.experimental.preprocessing.Normalization(
            mean=0.5, variance=0.25),

        # Base model
        base_model,

        # Global average pooling
        layers.GlobalAveragePooling2D(),

        # Dense layers with dropout
        layers.Dense(256, activation='relu',
                     kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        layers.Dense(128, activation='relu',
                     kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        # Output layer (7 emotions)
        layers.Dense(7, activation='softmax')
    ])

    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("\nModel Architecture:")
    model.summary()

    return model


def create_dummy_model(model_name='emotion_model.h5'):
    """
    Create a simple CNN model for emotion detection if transfer learning fails
    This is a fallback option with manageable size

    Args:
        model_name (str): Name to save the model

    Returns:
        model: Compiled Keras model
    """

    print("Building emotion detection model (CNN)...")

    model = models.Sequential([
        # Input layer
        layers.Input(shape=(48, 48, 1)),

        # Convolutional blocks
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Global pooling
        layers.GlobalAveragePooling2D(),

        # Dense layers
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        # Output layer
        layers.Dense(7, activation='softmax')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("\nModel Architecture:")
    model.summary()

    return model


def save_model(model, model_name='emotion_model.h5'):
    """Save the trained model"""
    try:
        model.save(model_name)
        print(f"\n✓ Model saved successfully as '{model_name}'")
        return True
    except Exception as e:
        print(f"\n✗ Error saving model: {e}")
        return False


def verify_model(model_name='emotion_model.h5'):
    """Verify the saved model can be loaded"""
    try:
        loaded_model = tf.keras.models.load_model(model_name)
        print(f"✓ Model verification successful")
        print(f"  Input shape: {loaded_model.input_shape}")
        print(f"  Output shape: {loaded_model.output_shape}")
        return True
    except Exception as e:
        print(f"✗ Model verification failed: {e}")
        return False


if __name__ == '__main__':
    print("=" * 60)
    print("Face Emotion Detection Model - Setup Script")
    print("=" * 60)

    model_filename = 'emotion_model.h5'

    try:
        # Try to build transfer learning model first
        print("\nAttempting to build MobileNetV2-based model...")
        model = build_emotion_model(model_filename)
    except Exception as e:
        print(f"MobileNetV2 model failed: {e}")
        print("Falling back to custom CNN model...")
        model = create_dummy_model(model_filename)

    # Save the model
    if save_model(model, model_filename):
        # Verify the model
        if verify_model(model_filename):
            print("\n" + "=" * 60)
            print("✓ Model setup complete!")
            print(f"✓ Model saved as: {model_filename}")
            print("✓ Ready to use with Flask application")
            print("=" * 60)
        else:
            print("\n✗ Model verification failed")
    else:
        print("\n✗ Model saving failed")
