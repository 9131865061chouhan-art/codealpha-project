import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK data
nltk.download('punkt')

# -----------------------------
# FAQ DATA
# -----------------------------
faqs = [
    {
        "question": "What is your product?",
        "answer": "Our product is an AI-based chatbot that helps users get answers to frequently asked questions."
    },
    {
        "question": "How can I create an account?",
        "answer": "Click on the Sign Up button and enter your name, email, and password."
    },
    {
        "question": "How can I reset my password?",
        "answer": "Click on 'Forgot Password' on the login page and follow the instructions."
    },
    {
        "question": "How can I contact customer support?",
        "answer": "You can contact customer support through email or the Help section of our website."
    },
    {
        "question": "Is the product free?",
        "answer": "Yes, a free version is available. Some advanced features may require a paid plan."
    },
    {
        "question": "How do I delete my account?",
        "answer": "Go to Account Settings and select the Delete Account option."
    }
]

# Extract questions and answers
questions = [faq["question"] for faq in faqs]
answers = [faq["answer"] for faq in faqs]


# -----------------------------
# TEXT PREPROCESSING
# -----------------------------
def preprocess(text):
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation and special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    # Tokenize
    tokens = nltk.word_tokenize(text)

    return " ".join(tokens)


processed_questions = [preprocess(q) for q in questions]


# -----------------------------
# TF-IDF VECTORIZATION
# -----------------------------
vectorizer = TfidfVectorizer()

faq_vectors = vectorizer.fit_transform(processed_questions)


# -----------------------------
# CHATBOT FUNCTION
# -----------------------------
def chatbot(user_question):

    # Preprocess user question
    processed_input = preprocess(user_question)

    # Convert user question into vector
    user_vector = vectorizer.transform([processed_input])

    # Calculate cosine similarity
    similarity_scores = cosine_similarity(
        user_vector,
        faq_vectors
    )

    # Find most similar FAQ
    best_match = similarity_scores.argmax()
    best_score = similarity_scores[0][best_match]

    # Minimum similarity threshold
    if best_score < 0.2:
        return "Sorry, I don't understand your question. Please try asking something else."

    return answers[best_match]


# -----------------------------
# CHAT LOOP
# -----------------------------
print("FAQ Chatbot")
print("Type 'exit' to stop the chatbot.\n")

while True:

    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Bot: Goodbye!")
        break

    response = chatbot(user_input)

    print("Bot:", response)