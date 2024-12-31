# Intent-Based AI Chatbot with Neural Networks

A sophisticated chatbot implementation using neural networks for intent classification and pattern matching. The system features a clean web interface built with Streamlit, preprocessed data handling, and a robust model architecture.

## Architecture Overview

### Preprocessing Pipeline
- Text tokenization for converting raw input into processable tokens
- Sequence conversion for neural network compatibility
- Uniform sequence padding for consistent input dimensions
- Intent label encoding for classification

### Model Architecture
- **Embedding Layer**: Learns dense word representations from input sequences
- **GlobalAveragePooling1D**: Reduces sequence dimensions efficiently
- **Dense Layers**: Multiple layers with dropout for regularization
- **Softmax Activation**: Enables multi-class intent classification

## Project Structure

```
chatbot/
├── app.py               # Training script
├── predict.py           # Command-line interface
├── streamlit_app.py     # Streamlit web interface
├── requirements.txt     # Dependencies
├── intents_corrected.json    # Training data
├── chatbot_model.h5         # Saved model (generated)
├── tokenizer.pickle        # Saved tokenizer (generated)
└── label_encoder.pickle    # Saved label encoder (generated)
```

## Features

### Core Functionality
- Intent recognition through pattern matching
- Response generation from predefined templates
- Multi-intent support
- Randomized response selection for variety
- Model and preprocessing object persistence

### User Interface
- Clean, modern chat interface
- Distinct user and bot message styling
- Persistent chat history within session
- Informative sidebar
- One-click chat reset
- Emoji integration for enhanced user experience

### Technical Features
- Model caching for improved performance
- Session state management
- Streamlined message processing pipeline
- Responsive design

## Installation

1. Clone the repository:
```bash
git clone https://github.com/SagarNikam09/Intent-Based-ChatBot.git
cd Intent-Based-ChatBot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model
```bash
python app.py
```

### Command Line Interface
```bash
python predict.py
```

### Web Interface
```bash
streamlit run streamlit_app.py
```

## Model Performance Optimization

To improve the model's performance, consider:
- Expanding the training dataset
- Adjusting model architecture (layer sizes, dropout rates)
- Fine-tuning hyperparameters
- Adding data augmentation
- Implementing cross-validation

## UI Components

### Main Interface
- Title and welcome message
- Scrollable chat history
- Message input field
- Clear chat functionality

### Message Display
- Right-aligned user messages
- Left-aligned bot messages
- Visual message distinction
- Smooth scrolling behavior

### Sidebar
- Project information
- Usage instructions
- Additional resources

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
