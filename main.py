import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.backend import clear_session

print(f"TensorFlow version: {tf.__version__}")

# Directories
train_dir = 'train'  # Make sure this points to your training data directory
test_dir = 'test'    # Make sure this points to your test data directory

# Parameters
img_height, img_width = 224, 224
batch_size = 32
epochs = 10
model_path = 'plant_model.keras'

# Function to load images and labels from the training directory
def load_train_images_and_labels(directory):
    images = []
    labels = []
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            filepath = os.path.join(subdir, file)
            image = cv2.imread(filepath)
            if image is not None:
                image = cv2.resize(image, (img_height, img_width))
                images.append(image)
                labels.append(os.path.basename(subdir))
            else:
                print(f"Failed to load image: {filepath}")
    return np.array(images), np.array(labels)

# Function to load multiple test images
def load_test_images(directory):
    images = []
    filenames = []
    for file in os.listdir(directory):
        filepath = os.path.join(directory, file)
        image = cv2.imread(filepath)
        if image is not None:
            image = cv2.resize(image, (img_height, img_width))
            images.append(image)
            filenames.append(file)
        else:
            print(f"Failed to load image: {filepath}")
    if not images:
        raise ValueError("No valid images found in the test directory.")
    return np.array(images), filenames

# Function to create the model
def create_model(num_classes):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(1024, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Load and preprocess the training data
train_images, train_labels = load_train_images_and_labels(train_dir)
if train_images.size == 0 or train_labels.size == 0:
    raise ValueError("Training directory is empty or does not contain valid images.")

train_images = train_images / 255.0  # Normalize images

# Encode labels
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)
train_labels_encoded = to_categorical(train_labels_encoded)

# Split into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(train_images, train_labels_encoded, test_size=0.2, random_state=42)

# Clear Keras backend session
clear_session()

# Check if model exists
if not os.path.exists(model_path):
    print("Training new model...")
    model = create_model(len(label_encoder.classes_))
    model.fit(x_train, y_train, epochs=epochs, validation_data=(x_val, y_val), batch_size=batch_size)
    model.save(model_path)
    print(f"Model saved to {model_path}")
else:
    print(f"Loading existing model from {model_path}")
    try:
        model = load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Creating a new model instead...")
        model = create_model(len(label_encoder.classes_))
        if os.path.exists(model_path):
            os.remove(model_path)
        model.fit(x_train, y_train, epochs=epochs, validation_data=(x_val, y_val), batch_size=batch_size)
        model.save(model_path)
        print(f"New model saved to {model_path}")

# Load and preprocess multiple test images
test_images, test_filenames = load_test_images(test_dir)
test_images = test_images / 255.0  # Normalize images

# Predict the class of each test image
predictions = model.predict(test_images)
predicted_class_indices = np.argmax(predictions, axis=1)
predicted_labels = label_encoder.inverse_transform(predicted_class_indices)

# Print results for each image
for filename, label in zip(test_filenames, predicted_labels):
    print(f'Image: {filename} - Predicted label: {label}')

# Additional debugging: Check number of images and their shapes
print(f"Number of test images: {len(test_filenames)}")
for i, filename in enumerate(test_filenames):
    print(f"Image {filename} shape: {test_images[i].shape}")