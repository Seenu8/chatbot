import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from sklearn.preprocessing import LabelEncoder
import json

nltk.download('punkt')

# read file
with open("C:\python\python\AI\intents.json", 'r') as file:
    data = json.load(file)

# separate QA
questions = []
responses = []
for intent in data["intents"]:
    questions.append(intent["question"])
    responses.append(intent["response"])

# tokenization (spliting the data)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(questions)
X = tokenizer.texts_to_sequences(questions)
max_len = max([len(x) for x in X])

# pad sequence (checks the len of the sentance)
X = pad_sequences(X, maxlen=max_len, padding='post')

# ecoding
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

# Function for response
def respond(input_text):
    input_seq = tokenizer.texts_to_sequences([input_text])
    input_seq = pad_sequences(input_seq, maxlen=max_len, padding='post')
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


#
# import mysql.connector
# import spacy
# from transformers import pipeline
#
# # Load NLP model for entity recognition
# nlp = spacy.load("en_core_web_sm")
# qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased")
#
# # Connect to MySQL database
# db = mysql.connector.connect(
#     host="localhost",
#     user="root",
#     password="root",
#     database="printer_management"
# )
#
# cursor = db.cursor()
#
#
# def extract_entities(user_input):
#     """ Extracts relevant entities from the user's input """
#     doc = nlp(user_input)
#     entities = [ent.text for ent in doc.ents]
#     return entities
#
#
# def get_printer_details(printer_name):
#     conn = mysql.connector.connect(
#         host="localhost",
#         user="root",
#         password="password",
#         database="printers_db"
#     )
#     cursor = conn.cursor()
#
#     # Ensure you're using the correct column name
#     query = "SELECT printer_name, status FROM printers WHERE printer_name LIKE %s"
#     cursor.execute(query, (f"%{printer_name}%",))
#
#     result = cursor.fetchone()
#     conn.close()
#
#     if result:
#         return f"Printer: {result[0]}, Status: {result[1]}"
#     else:
#         return "Printer not found."
#
#
# def chatbot():
#     print("Chatbot: Hello! Ask me about printers. Type 'exit' to quit.")
#
#     while True:
#         user_input = input("You: ")
#         if user_input.lower() == "exit":
#             print("Chatbot: Goodbye!")
#             break
#
#         # Extract key entities
#         entities = extract_entities(user_input)
#         if entities:
#             printer_name = entities[0]  # Assume first entity is the printer name
#             details = get_printer_details(printer_name)
#
#             if details:
#                 response_text = f"Printer Details:\n"
#                 for name, model, status in details:
#                     response_text += f"- {name} (Model: {model}, Status: {status})\n"
#             else:
#                 response_text = "Sorry, no details found for that printer."
#
#         else:
#             response_text = "Could you clarify which printer you're asking about?"
#
#         # Make response more conversational using Hugging Face
#         response = qa_pipeline({
#             "question": user_input,
#             "context": response_text
#         })
#
#         print(f"Chatbot: {response['answer']}")
#
#
# chatbot()
