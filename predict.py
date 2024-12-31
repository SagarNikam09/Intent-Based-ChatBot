import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import random

def load_response_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return {intent['tag']: intent['responses'] for intent in data['intents']}

def predict_intent(text, model, tokenizer, label_encoder, max_len):
    # Preprocess the input text
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=max_len)
    
    # Make prediction
    prediction = model.predict(padded)
    predicted_class = label_encoder.inverse_transform([np.argmax(prediction)])
    
    return predicted_class[0]

def get_response(tag, responses_dict):
    return random.choice(responses_dict[tag])

def chat():
    # Load the saved model and preprocessing objects
    model = load_model('chatbot_model.h5')
    
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    
    with open('label_encoder.pickle', 'rb') as handle:
        label_encoder = pickle.load(handle)
    
    # Load responses
    responses_dict = load_response_data('intents_corrected.json')
    
    # Get max_len from the model
    max_len = model.input_shape[1]
    
    print("Bot: Hi! How can I help you today? (type 'quit' to exit)")
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() == 'quit':
            print("Bot: Goodbye!")
            break
        
        # Predict intent
        predicted_tag = predict_intent(user_input, model, tokenizer, label_encoder, max_len)
        
        # Get response
        response = get_response(predicted_tag, responses_dict)
        print("Bot:", response)

if __name__ == "__main__":
    chat() 