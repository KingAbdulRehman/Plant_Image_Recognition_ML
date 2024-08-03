import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import os
import pickle
from pathlib import Path
import cv2

# Parameters
img_height, img_width = 150, 150
batch_size = 32
epochs = 15

# File paths
model_path = 'simple_image_model.keras'
class_indices_path = 'class_indices.pickle'

# Directories
train_dir = 'train'  # Update this
validation_dir = 'train'  # Update this
test_dir = 'test'  # Update this

def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = cv2.resize(img, (img_width, img_height))
    img = img.astype(np.float32) / 255.0
    return img

def image_generator(directory, batch_size):
    class_dirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    class_indices = {class_name: i for i, class_name in enumerate(class_dirs)}
    
    image_paths = []
    labels = []
    for class_name in class_dirs:
        class_path = os.path.join(directory, class_name)
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            image_paths.append(img_path)
            labels.append(class_indices[class_name])
    
    num_samples = len(image_paths)
    print(f"Found {num_samples} images in {directory} across {len(class_dirs)} classes.")
    
    while True:
        indices = np.random.permutation(num_samples)
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_indices = indices[start:end]
            
            batch_images = [load_and_preprocess_image(image_paths[i]) for i in batch_indices]
            batch_labels = [labels[i] for i in batch_indices]
            
            yield np.array(batch_images), tf.keras.utils.to_categorical(batch_labels, num_classes=len(class_dirs))

def count_images_in_directory(directory):
    num_images = sum(len(files) for _, _, files in os.walk(directory) if any(f.lower().endswith(('png', 'jpg', 'jpeg')) for f in files))
    return num_images

def create_and_train_model():
    train_generator = image_generator(train_dir, batch_size)
    validation_generator = image_generator(validation_dir, batch_size)

    num_classes = len([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    print(f"Number of classes detected: {num_classes}")

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    num_train_samples = count_images_in_directory(train_dir)
    num_validation_samples = count_images_in_directory(validation_dir)

    steps_per_epoch = num_train_samples // batch_size
    validation_steps = num_validation_samples // batch_size

    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Validation steps: {validation_steps}")

    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_steps
    )

    class_indices = {i: class_name for i, class_name in enumerate([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])}
    return model, class_indices

# Check if model and class indices already exist
if os.path.exists(model_path) and os.path.exists(class_indices_path):
    print("Loading existing model and class indices...")
    model = load_model(model_path)
    with open(class_indices_path, 'rb') as handle:
        class_indices = pickle.load(handle)
else:
    print("Creating and training new model...")
    model, class_indices = create_and_train_model()

    # Save the model and class indices
    model.save(model_path)
    with open(class_indices_path, 'wb') as handle:
        pickle.dump(class_indices, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Model saved to {model_path}")
    print(f"Class indices saved to {class_indices_path}")

# Function to predict image class
def predict_image(image_path):
    img = load_and_preprocess_image(image_path)
    img_array = np.expand_dims(img, 0)  # Create batch axis

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    class_names = list(class_indices.values())
    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)

    return predicted_class, confidence

# Test the model on all images in the test directory
test_images = list(Path(test_dir).glob('*.*'))
supported_formats = {'.jpg', '.jpeg', '.png', '.bmp'}

print(f"Found {len(test_images)} files in the test directory.")
print("Testing the model on images...")

for image_path in test_images:
    if image_path.suffix.lower() in supported_formats:
        predicted_class, confidence = predict_image(str(image_path))
        print(f"Image: {image_path.name}")
        print(f"Predicted class: {predicted_class}")
        print(f"Confidence: {confidence:.2f}%")
        print()
    else:
        print(f"Skipping unsupported file: {image_path.name}")

print("Testing complete.")
