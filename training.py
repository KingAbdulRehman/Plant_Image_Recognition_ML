import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import pickle

# Parameters
img_height, img_width = 1000, 1000
batch_size = 128
epochs = 150
additional_epochs = 50

# File paths
model_path = 'simple_image_model.keras'
class_indices_path = 'class_indices.pickle'

# Directories
train_dir = 'train'
validation_dir = 'train'

def create_data_generators():
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

    return train_generator, validation_generator

def create_model(num_classes):
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

    return model

def train_model(model, train_generator, validation_generator, epochs_to_train):
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=epochs_to_train,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size
    )
    return history

if __name__ == "__main__":
    train_generator, validation_generator = create_data_generators()
    num_classes = len(train_generator.class_indices)
    print(f"Number of classes detected: {num_classes}")

    if os.path.exists(model_path) and os.path.exists(class_indices_path):
        with open(class_indices_path, 'rb') as handle:
            old_class_indices = pickle.load(handle)
        
        if len(old_class_indices) != num_classes:
            print("Number of classes has changed. Creating a new model...")
            model = create_model(num_classes)
            history = train_model(model, train_generator, validation_generator, epochs)
        else:
            print("Loading existing model and continuing training...")
            model = load_model(model_path)
            history = train_model(model, train_generator, validation_generator, additional_epochs)
    else:
        print("Creating and training new model...")
        model = create_model(num_classes)
        history = train_model(model, train_generator, validation_generator, epochs)

    # Save the model and class indices
    model.save(model_path)
    with open(class_indices_path, 'wb') as handle:
        pickle.dump(train_generator.class_indices, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Model saved to {model_path}")
    print(f"Class indices saved to {class_indices_path}")

    # You can add code here to plot training history if desired