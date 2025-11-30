import json
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import nltk
from nltk.stem import WordNetLemmatizer


with open("intents.json", "r", encoding="utf-8") as f:
    data = json.load(f)

lemmatizer = WordNetLemmatizer()
nltk.download('punkt')
nltk.download('wordnet')


X = []
y = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        tokens = nltk.word_tokenize(pattern.lower())
        tokens = [lemmatizer.lemmatize(w) for w in tokens]
        X.append(' '.join(tokens))
        y.append(intent["tag"])


vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)


model = MultinomialNB()
model.fit(X_vec, y)


with open("chatbot_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Model trained and saved successfully!")

