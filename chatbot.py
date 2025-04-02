import json
import re
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Read the intents file
with open("C:\\python\\AI\\intents.json", 'r') as file:
    data = json.load(file)

# Extract questions and responses
questions = []
responses = {}

# Process the data
for intent in data["intents"]:
    questions.append(intent["question"])
    responses[intent["question"]] = intent["response"]

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer(
    tokenizer=lambda x: re.findall(r'\w+', x.lower()),
    stop_words='english'
)

# Fit the vectorizer on the questions
question_vectors = vectorizer.fit_transform(questions)


def get_response(user_input):
    # Vectorize the user input
    user_vector = vectorizer.transform([user_input.lower()])

    # Calculate similarity with all questions
    similarities = cosine_similarity(user_vector, question_vectors)[0]

    # Find the most similar question
    best_match_index = similarities.argmax()
    best_match_score = similarities[best_match_index]

    # If the similarity is too low, return a default response
    if best_match_score < 0.3:
        return "I'm not sure I understand. Could you rephrase that?"

    # Return the response for the most similar question
    best_match_question = questions[best_match_index]
    return responses[best_match_question]


# Interactive chat loop
print("Chatbot initialized. Type 'bye' to exit.")
print("Hello! How can I assist you?")

while True:
    user_input = input("You: ")
    if user_input.lower() == "bye":
        print("Bot: Goodbye!")
        break
    else:
        try:
            response = get_response(user_input)
            print(f"Bot: {response}")
        except Exception as e:
            print(f"Error: {e}")
            print("Bot: I'm having trouble understanding. Could you try again?")