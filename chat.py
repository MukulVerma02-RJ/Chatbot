import random
import json
import pickle
import numpy as np
import nltk
import time
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load data and model
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbotmodel.h5')

# Clean up input sentence
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Create bag-of-words
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# Predict the intent of a sentence
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

# Generate a response based on intent
def get_response(intents_list, intents_json):
    if not intents_list:
        return "I'm sorry, I didn't understand that. Can you try again?"
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

# Chat loop
if __name__ == "__main__":
    print("\n|=========== Welcome to College Enquiry Chatbot System! ===========|")
    print("|======================== Feel Free ========================|")
    print("|======================== To ========================|")
    print("|=========== Ask your any query about our college ===========|\n")

    while True:
        message = input("You: ")
        if message.lower() in ["exit", "quit", "bye"]:
            print("Bot: Goodbye! Have a great day!")
            break
        ints = predict_class(message)
        res = get_response(ints, intents)
        time.sleep(0.5)  # Optional delay to feel natural
        print("Bot:", res)
        time.sleep(0.5)  # Optional delay to feel natural
            