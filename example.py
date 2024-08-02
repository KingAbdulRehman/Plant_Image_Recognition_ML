import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import pickle
from pathlib import Path

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

def create_and_train_model():
    # Data generators
    train_datagen = ImageDataGenerator(rescale=1./255)
    validation_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical'
    )

    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical'
    )

    # Automatically determine the number of classes
    num_classes = len(train_generator.class_indices)
    print(f"Number of classes detected: {num_classes}")

    # Create the model
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

    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size
    )

    return model, train_generator.class_indices

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
    img = tf.keras.preprocessing.image.load_img(
        image_path, target_size=(img_height, img_width)
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis
    img_array /= 255.  # Rescale the image

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    class_names = list(class_indices.keys())
    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)

    return predicted_class, confidence

# Test the model on all images in the test directory
test_images = list(Path(test_dir).glob('*.*'))
supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}

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