# 🎓 College Enquiry Chatbot

This is a deep learning-based chatbot designed to answer common questions about your college. It uses natural language processing (NLP) to understand user inputs and reply with relevant information trained on predefined intents.

## 📋 Table of Contents
- [Features](#features)
- [Technologies Used](#technologies-used)
- [How It Works](#how-it-works)
- [Setup Instructions](#setup-instructions)
- [Folder Structure](#folder-structure)
- [Sample Intents](#sample-intents)

## ✨ Features

- 🔍 Understands user queries with NLP (tokenization + lemmatization)
- 🤖 Trained with TensorFlow on intent-tagged data
- 🧠 Uses a bag-of-words model to classify user input
- 🎯 High accuracy response prediction with softmax classifier
- 💬 Engaging and interactive terminal-based chat interface
- ✅ Handles unknown inputs gracefully

## ⚙️ Technologies Used

- Python 3.10+
- NLTK (tokenization & lemmatization)
- TensorFlow / Keras
- NumPy
- JSON (for intent dataset)
- Pickle (for saving NLP structures)

## 🧠 How It Works

1. **Train the model** using `train_chatbot.py`:
   - Loads `intents.json` patterns/tags
   - Prepares vocabulary (words & classes)
   - Builds and trains a neural network model
   - Saves the trained model (`chatbotmodel.h5`), words, and classes

2. **Chat with the bot** using `chat.py`:
   - Loads model and data
   - Takes user input, classifies intent
   - Responds with a random message from the matching intent tag

## 🚀 Setup Instructions

### 1. Clone the Repo ##Install Required Packages
'''bash 
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

### 2. Install Required Packages
'''bash 
pip install numpy tensorflow nltk

### 3. Train the Chatbot (if not already trained)
'''bash 
python train_chatbot.py

### 4. Start Chatting!
'''bash
python chat.py
(Type your questions in the terminal. To exit, type exit, bye, or quit.)

## 📁 Folder Structure
📦 chatbot/
├── 📄 chat.py              # Main chatbot script (user interface)
├── 📄 train_chatbot.py     # Trains the deep learning model
├── 📄 intents.json         # Intent dataset (patterns, responses)
├── 📄 chatbotmodel.h5      # Trained model (auto-generated)
├── 📄 words.pkl            # NLP vocabulary (auto-generated)
├── 📄 classes.pkl          # Intent classes (auto-generated)
└── 📄 README.md            # Project documentation

## 🧾 Sample Intents (intents.json)
json
{
  "intents": [
    {
      "tag": "greeting",
      "patterns": ["Hi", "Hello", "Is anyone there?"],
      "responses": ["Hello!", "Hi there, how can I help you?"]
    },
    {
      "tag": "admission",
      "patterns": ["How to apply?", "Admission process?"],
      "responses": ["You can apply online through our official portal."]
    }
  ]
}

## 🙌 Credits
Developed by Mukul Rajput
Project: College Enquiry Chatbot using Deep Learning







