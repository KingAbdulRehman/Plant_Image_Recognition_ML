import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Directories
train_dir = 'train'
test_dir = 'test'

# Parameters
img_height, img_width = 224, 224
batch_size = 32
epochs = 10

# Function to load images and labels from the training directory
def load_train_images_and_labels(directory):
    images = []
    labels = []
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            filepath = os.path.join(subdir, file)
            image = cv2.imread(filepath)
            if image is not None:
                image = cv2.resize(image, (img_height, img_width))  # Resize image
                images.append(image)
                labels.append(os.path.basename(subdir))  # Use folder name as label
            else:
                print(f"Failed to load image: {filepath}")
    return np.array(images), np.array(labels)

# Function to load a single test image
def load_test_image(directory):
    for file in os.listdir(directory):
        filepath = os.path.join(directory, file)
        image = cv2.imread(filepath)
        if image is not None:
            image = cv2.resize(image, (img_height, img_width))  # Resize image
            return np.array([image]), file  # Return the image and the filename
        else:
            print(f"Failed to load image: {filepath}")
    raise ValueError("No valid images found in the test directory.")

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

# Load and preprocess the test image
test_image, test_filename = load_test_image(test_dir)
test_image = test_image / 255.0  # Normalize image

# Load the pre-trained model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# Add new layers on top of the pre-trained model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(1024, activation='relu'),
    Dense(len(label_encoder.classes_), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=epochs, validation_data=(x_val, y_val), batch_size=batch_size)

# Predict the class of the test image
prediction = model.predict(test_image)
predicted_class_index = np.argmax(prediction, axis=1)
predicted_label = label_encoder.inverse_transform(predicted_class_index)

# Print the result
print(f'Image: {test_filename} - Predicted label: {predicted_label[0]}')
