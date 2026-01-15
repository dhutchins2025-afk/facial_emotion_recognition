"""
model_retrain.py
Train a CNN model on FER2013 emotion detection dataset
This script downloads/uses the FER2013 dataset and trains the model
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import urllib.request
import zipfile
import pandas as pd
import os

# Emotion labels
EMOTIONS = ['Angry', 'Disgusted', 'Fearful',
            'Happy', 'Neutral', 'Sad', 'Surprised']
EMOTION_MAP = {i: emotion for i, emotion in enumerate(EMOTIONS)}


def download_fer2013(data_dir='data'):
    """
    Download FER2013 dataset from a public source
    Note: FER2013 is now available through Kaggle API
    This is a helper function - users may need to download manually
    """
    print("Note: FER2013 dataset requires Kaggle API or manual download")
    print("Visit: https://www.kaggle.com/datasets/msambare/fer2013")
    print("Place fer2013.csv in the 'data' directory")

    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)

    csv_path = data_path / 'fer2013.csv'
    return csv_path


def load_fer2013_data(csv_path, test_size=0.2, validation_split=0.2):
    """
    Load FER2013 data from CSV file
    Format: emotion,pixels,Usage (emotion: 0-6, pixels: space-separated pixel values)
    """
    print(f"Loading FER2013 dataset from {csv_path}...")

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: {csv_path} not found!")
        print(
            "Please download FER2013 from Kaggle and place fer2013.csv in 'data' directory")
        return None, None, None, None

    # Parse pixel data
    print("Parsing image data...")
    X = []
    y = []

    for idx, row in df.iterrows():
        if idx % 1000 == 0:
            print(f"Processed {idx}/{len(df)} images")

        # Parse pixel values (space-separated string -> array)
        pixels = np.array(row['pixels'].split(), dtype='uint8').reshape(48, 48)
        emotion = row['emotion']

        X.append(pixels)
        y.append(emotion)

    X = np.array(X, dtype='float32') / 255.0  # Normalize
    X = np.expand_dims(X, axis=-1)  # Add channel dimension
    y = keras.utils.to_categorical(y, num_classes=7)  # One-hot encode

    print(f"Loaded {len(X)} images with shape {X.shape}")

    # Split data
    from sklearn.model_selection import train_test_split

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(test_size + validation_split), random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=test_size / (test_size + validation_split),
        random_state=42
    )

    print(f"\nData split:")
    print(f"  Train: {X_train.shape[0]} images")
    print(f"  Validation: {X_val.shape[0]} images")
    print(f"  Test: {X_test.shape[0]} images")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), df


def build_emotion_model():
    """Build CNN model for emotion classification"""

    model = models.Sequential([
        # Input layer
        layers.Input(shape=(48, 48, 1)),

        # Block 1
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 3
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 4
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Global pooling and dense layers
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        # Output layer
        layers.Dense(7, activation='softmax')
    ])

    return model


def compile_model(model):
    """Compile the model"""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def train_model(model, train_data, val_data, epochs=50, batch_size=32):
    """Train the model"""

    X_train, y_train = train_data
    X_val, y_val = val_data

    # Data augmentation
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2
    )

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        ),
        keras.callbacks.ModelCheckpoint(
            'emotion_model_best.h5',
            monitor='val_accuracy',
            save_best_only=True
        )
    ]

    # Train with data augmentation
    history = model.fit(
        train_datagen.flow(X_train, y_train, batch_size=batch_size),
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )

    return history


def evaluate_model(model, test_data):
    """Evaluate model on test set"""
    X_test, y_test = test_data

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

    print(f"\nTest Set Performance:")
    print(f"  Loss: {loss:.4f}")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    return loss, accuracy


def plot_training_history(history, save_path='training_history.png'):
    """Plot and save training history"""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Accuracy
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.legend()
    ax1.grid(True)

    # Loss
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Val Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Model Loss')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"Training history saved to {save_path}")
    plt.close()


def save_model(model, model_path='emotion_model.h5'):
    """Save the trained model"""
    try:
        model.save(model_path)
        print(f"✓ Model saved as {model_path}")
        return True
    except Exception as e:
        print(f"✗ Error saving model: {e}")
        return False


if __name__ == '__main__':
    print("=" * 70)
    print("Face Emotion Detection - FER2013 Training Script")
    print("=" * 70)

    # Download/locate FER2013
    csv_path = download_fer2013()

    # Load data
    train_data, val_data, test_data, df = load_fer2013_data(csv_path)

    if train_data is None:
        print("\nCannot proceed without data. Please download FER2013 dataset.")
        exit(1)

    # Build model
    print("\nBuilding model architecture...")
    model = build_emotion_model()
    model = compile_model(model)

    print("\nModel Summary:")
    model.summary()

    # Train model
    print("\nStarting training...")
    history = train_model(model, train_data, val_data,
                          epochs=50, batch_size=32)

    # Plot training history
    plot_training_history(history)

    # Evaluate on test set
    evaluate_model(model, test_data)

    # Save final model
    print("\nSaving final model...")
    if save_model(model, 'emotion_model.h5'):
        print("\n" + "=" * 70)
        print("✓ Training complete!")
        print("✓ Model ready for deployment")
        print("=" * 70)
