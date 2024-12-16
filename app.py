import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from sklearn.preprocessing import LabelEncoder
import json
import tkinter as tk
from tkinter import scrolledtext

# Install NLTK data for tokenization
nltk.download('punkt')

# Load intents data
with open("C:\python\python\AI\intents.json", 'r') as file:
    data = json.load(file)

# Prepare the data for training
questions = []
responses = []
for intent in data["intents"]:
    questions.append(intent["question"])
    responses.append(intent["response"])

tokenizer = Tokenizer()
tokenizer.fit_on_texts(questions)
X = tokenizer.texts_to_sequences(questions)
max_len = max([len(x) for x in X])

# Pad sequences to make sure all input sequences have the same length
X = pad_sequences(X, maxlen=max_len, padding='post')

# Encode the responses using LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(responses)
y = np.array(y)

# Define the LSTM model architecture
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64, input_length=max_len))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(len(np.unique(y)), activation='softmax'))

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=500, batch_size=16)

# Function to generate a response
def respond(input_text):
    # Tokenize the input
    input_seq = tokenizer.texts_to_sequences([input_text])
    input_seq = pad_sequences(input_seq, maxlen=max_len, padding='post')

    # Predict the response
    prediction = model.predict(input_seq)
    predicted_class = np.argmax(prediction, axis=1)[0]
    response = label_encoder.inverse_transform([predicted_class])

    return response[0]

# Create the main Tkinter window
root = tk.Tk()
root.title("Chatbot")

# Create the conversation area (scrollable)
conversation_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=50, height=20, state='disabled')
conversation_area.grid(row=0, column=0, padx=10, pady=10)

# Create the entry box for user input
user_input_box = tk.Entry(root, width=40)
user_input_box.grid(row=1, column=0, padx=10, pady=10)

# Function to update the conversation and get a response
def on_send():
    user_input = user_input_box.get()
    if user_input.lower() == "bye":
        conversation_area.config(state='normal')
        conversation_area.insert(tk.END, "You: " + user_input + "\n")
        conversation_area.insert(tk.END, "Bot: Goodbye!\n")
        conversation_area.config(state='disabled')
        user_input_box.delete(0, tk.END)
        root.quit()
    else:
        conversation_area.config(state='normal')
        conversation_area.insert(tk.END, "You: " + user_input + "\n")
        response = respond(user_input)
        conversation_area.insert(tk.END, "Bot: " + response + "\n")
        conversation_area.config(state='disabled')
        user_input_box.delete(0, tk.END)

# Create the Send button
send_button = tk.Button(root, text="Send", command=on_send)
send_button.grid(row=2, column=0, padx=10, pady=10)

# Start the Tkinter main loop
root.mainloop()
