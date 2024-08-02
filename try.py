import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
import os
import pickle

# Sample data
texts = [
    "I love this product",
    "This is terrible",
    "Great service",
    "Poor quality",
    "Highly recommended",
    "Do not buy",
    "Amazing experience",
    "Waste of money"
]
labels = [1, 0, 1, 0, 1, 0, 1, 0]  # 1 for positive, 0 for negative

# File paths
model_path = 'simple_text_model.keras'
tokenizer_path = 'tokenizer.pickle'

# Function to create and train the model
def create_and_train_model(X, y):
    model = Sequential([
        Embedding(input_dim=1000, output_dim=16, input_length=10),
        GlobalAveragePooling1D(),
        Dense(24, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X, y, epochs=50, verbose=1)
    return model

# Check if model and tokenizer already exist
if os.path.exists(model_path) and os.path.exists(tokenizer_path):
    print("Loading existing model and tokenizer...")
    model = load_model(model_path)
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
else:
    print("Creating and training new model...")
    # Tokenize the text
    tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    # Pad sequences
    padded = pad_sequences(sequences, maxlen=10, padding='post', truncating='post')

    # Convert to numpy arrays
    X = np.array(padded)
    y = np.array(labels)

    # Create and train the model
    model = create_and_train_model(X, y)

    # Save the model and tokenizer
    model.save(model_path)
    with open(tokenizer_path, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Model saved to {model_path}")
    print(f"Tokenizer saved to {tokenizer_path}")

# Function to predict sentiment
def predict_sentiment(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=10, padding='post', truncating='post')
    prediction = model.predict(padded)
    return "Positive" if prediction[0][0] > 0.5 else "Negative"

# Test the model
test_texts = [
    "This is awesome",
    "I hate it",
    "Not bad at all",
    "Disappointing product"
]

for text in test_texts:
    sentiment = predict_sentiment(text)
    print(f"Text: '{text}' - Sentiment: {sentiment}")