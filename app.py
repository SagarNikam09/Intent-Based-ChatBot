import json
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D, Dropout
from tensorflow.keras.utils import to_categorical
import pickle

# Load and preprocess the data
def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # Extract patterns and tags
    patterns = []
    tags = []
    for intent in data['intents']:
        for pattern in intent['patterns']:
            patterns.append(pattern)
            tags.append(intent['tag'])
    
    return patterns, tags

# Preprocess text
def preprocess_text(texts):
    # Create tokenizer
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    
    # Convert texts to sequences
    sequences = tokenizer.texts_to_sequences(texts)
    
    # Pad sequences
    max_len = max(len(seq) for seq in sequences)
    padded_sequences = pad_sequences(sequences, maxlen=max_len)
    
    return tokenizer, padded_sequences, max_len

# Build the model
def build_model(vocab_size, max_len, num_classes):
    model = Sequential([
        Embedding(vocab_size, 128, input_length=max_len),
        GlobalAveragePooling1D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    return model

# Main execution
def main():
    # Load data
    patterns, tags = load_data('intents_corrected.json')
    
    # Preprocess patterns
    tokenizer, X, max_len = preprocess_text(patterns)
    
    # Encode tags
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(tags)
    y = to_categorical(y)
    
    # Build model
    vocab_size = len(tokenizer.word_index) + 1
    num_classes = len(set(tags))
    
    model = build_model(vocab_size, max_len, num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Train model
    model.fit(X, y, epochs=100, batch_size=32, validation_split=0.2)
    
    # Save the model and preprocessing objects
    model.save('chatbot_model.h5')
    
    # Save tokenizer and label encoder
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open('label_encoder.pickle', 'wb') as handle:
        pickle.dump(label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()

