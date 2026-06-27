from flask import Flask, render_template, request, jsonify
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

with open("faq_data.json","r") as file:
    faqs = json.load(file)

questions = [faq["question"] for faq in faqs]

vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(questions)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():

    user_question = request.json["message"]

    user_vector = vectorizer.transform([user_question])

    similarity = cosine_similarity(
        user_vector,
        question_vectors
    )

    best_match = similarity.argmax()

    score = similarity.max()

    if score < 0.7:
        answer = "Sorry, I don't have an answer for that question."
    else:
        answer = faqs[best_match]["answer"]

    return jsonify({"reply": answer})


if __name__ == "__main__":
    app.run(debug=True)