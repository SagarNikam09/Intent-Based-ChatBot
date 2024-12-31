import streamlit as st
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import random
from exp import CustomLayer

# Load the model and preprocessing objects
@st.cache_resource
def load_chatbot_resources():
    model = load_model('chatbot_model.h5', custom_objects={'CustomLayer': CustomLayer})
    #model = load_model('chatbot_model.h5')
    
    try:
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
            print("Tokenizer loaded successfully.")
    except Exception as e:
        print("Error loading tokenizer:", e)

    try:
        with open('label_encoder.pickle', 'rb') as handle:
            label_encoder = pickle.load(handle)
            print("Label encoder loaded successfully.")
    except Exception as e:
        print("Error loading label encoder:", e)
        
    with open('intents_corrected.json', 'r') as file:
        data = json.load(file)
        responses_dict = {intent['tag']: intent['responses'] for intent in data['intents']}
    
    return model, tokenizer, label_encoder, responses_dict

def predict_intent(text, model, tokenizer, label_encoder, max_len):
    # Preprocess the input text
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=max_len)
    
    # Make prediction
    prediction = model.predict(padded, verbose=0)
    predicted_class = label_encoder.inverse_transform([np.argmax(prediction)])
    
    return predicted_class[0]

def get_response(tag, responses_dict):
    return random.choice(responses_dict[tag])

def main():
    st.title("AI Chatbot 🤖")
    st.write("Hello! I'm your AI assistant. Ask me anything!")

    # Load resources
    model, tokenizer, label_encoder, responses_dict = load_chatbot_resources()
    max_len = model.input_shape[1]

    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("What's on your mind?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get chatbot response
        predicted_tag = predict_intent(prompt, model, tokenizer, label_encoder, max_len)
        response = get_response(predicted_tag, responses_dict)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

    # Sidebar with additional information
    with st.sidebar:
        st.title("About")
        st.write("""
        This is an AI chatbot trained on various topics including:
        - Greetings and conversations
        - Technical topics
        - Health and wellness
        - Education and career
        - And much more!
        """)
        
        # Add a clear chat button
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()

if __name__ == "__main__":
    main() 