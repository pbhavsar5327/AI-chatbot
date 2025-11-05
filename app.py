from flask import Flask, render_template, request, jsonify
import json, random, os, pickle, sqlite3, numpy as np
from nltk.stem import WordNetLemmatizer
import nltk

# Initialize Flask
app = Flask(__name__)

# File paths
DB_PATH = os.path.join("database", "chatbot.db")
INTENTS_PATH = "intents.json"
MODEL_PATH = "chatbot_model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

# Make sure database folder exists
os.makedirs("database", exist_ok=True)

# Load model and vectorizer
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

# Load intents
with open(INTENTS_PATH, "r", encoding="utf-8") as f:
    intents = json.load(f)

# Initialize NLTK tools
nltk.download('punkt')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

# Database setup
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_message TEXT,
            bot_reply TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_db()

# Log each chat
def log_chat(user, bot):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT INTO chat_history (user_message, bot_reply) VALUES (?,?)", (user, bot))
    conn.commit()
    conn.close()

# Generate response using ML model
def get_response(msg):
    msg_clean = msg.lower().strip()
    words = nltk.word_tokenize(msg_clean)
    words = [lemmatizer.lemmatize(w) for w in words]
    user_input = ' '.join(words)

    X = vectorizer.transform([user_input])
    prediction = model.predict(X)[0]

    # Find corresponding intent from JSON
    for intent in intents["intents"]:
        if intent["tag"] == prediction:
            reply = random.choice(intent["responses"])
            log_chat(msg, reply)
            return reply

    # Default fallback reply
    reply = random.choice([
        "🤔 I didn’t quite get that. Try asking about Admissions, Courses, or Fees.",
        "💡 Try asking like 'B.Tech fees' or 'Hostel details'."
    ])
    log_chat(msg, reply)
    return reply

# Home route
@app.route("/")
def home():
    categories = sorted({i["tag"].replace("_", " ").title() for i in intents["intents"]})
    return render_template("index.html", categories=categories)

# AJAX route for chatbot
@app.route("/get", methods=["POST"])
def get_bot_reply():
    data = request.get_json()
    msg = data.get("message", "")
    return jsonify({"reply": get_response(msg)})

# Run app
if __name__ == "__main__":
    app.run(debug=True)
