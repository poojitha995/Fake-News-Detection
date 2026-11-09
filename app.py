
from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained machine learning model
model = joblib.load("model/fake_news_pipeline.pkl")


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    confidence = None
    news_text = ""

    if request.method == "POST":

        news_text = request.form.get("news", "").strip()

        if news_text:

            # Predict the news category
            result = model.predict([news_text])[0]

            # Get prediction confidence
            probabilities = model.predict_proba([news_text])[0]
            confidence = round(max(probabilities) * 100, 2)

            if result == 0:
                prediction = "Fake News"
            else:
                prediction = "Real News"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        news_text=news_text
    )


if __name__ == "__main__":
    app.run(debug=True)
