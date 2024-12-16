import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from sklearn.preprocessing import LabelEncoder
import json

# Install NLTK data for tokenization
nltk.download('punkt')

with open("intents.json", 'r') as file:
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

print("Hello! How can I assist you with?")

while True:
    user_input = input("You: ")
    if user_input.lower() == "bye":
        print("Bot: Goodbye!")
        break
    else:
        response = respond(user_input)
        print(f"Bot: {response}")
