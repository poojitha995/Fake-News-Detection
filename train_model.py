
import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# 1. Load datasets
fake = pd.read_csv("dataset/Fake.csv")
real = pd.read_csv("dataset/True.csv")

# 2. Add labels
fake["label"] = 0
real["label"] = 1

# 3. Combine datasets
data = pd.concat([fake, real], ignore_index=True)

# 4. Prepare text
data["title"] = data["title"].fillna("")
data["text"] = data["text"].fillna("")

data["content"] = data["title"] + " " + data["text"]

# 5. Remove empty and duplicate articles
data = data[data["content"].str.strip() != ""]
data = data.drop_duplicates(subset=["content"])

X = data["content"]
y = data["label"]

# 6. Split training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 7. Create ML pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        max_features=50000,
        ngram_range=(1, 2)
    )),
    ("classifier", LogisticRegression(max_iter=1000))
])

# 8. Train the model
print("Training Fake News Detection Model...")
model.fit(X_train, y_train)

# 9. Test the model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    target_names=["Fake News", "Real News"]
))

# 10. Save the trained model
os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/fake_news_pipeline.pkl")

print("\nModel saved successfully!")
