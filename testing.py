import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import pickle
from pathlib import Path

# Parameters
img_height, img_width = 1000, 1000

# File paths
model_path = 'simple_image_model.keras'
class_indices_path = 'class_indices.pickle'

# Directories
test_dir = 'test'

def predict_image(model, class_indices, image_path):
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

if __name__ == "__main__":
    # Load the model and class indices
    if os.path.exists(model_path) and os.path.exists(class_indices_path):
        print("Loading existing model and class indices...")
        model = load_model(model_path)
        with open(class_indices_path, 'rb') as handle:
            class_indices = pickle.load(handle)
    else:
        print("Error: Model or class indices not found. Please run the training script first.")
        exit(1)

    # Test the model on all images in the test directory
    test_images = list(Path(test_dir).glob('*.*'))
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}

    print(f"Found {len(test_images)} files in the test directory.")
    print("Testing the model on images...")

    for image_path in test_images:
        if image_path.suffix.lower() in supported_formats:
            predicted_class, confidence = predict_image(model, class_indices, str(image_path))
            print(f"Image: {image_path.name}")
            print(f"Predicted class: {predicted_class}")
            print(f"Confidence: {confidence:.2f}%")
            print()
        else:
            print(f"Skipping unsupported file: {image_path.name}")

    print("Testing complete.")