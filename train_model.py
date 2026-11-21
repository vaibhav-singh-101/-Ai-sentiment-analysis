
import os
import json
import re
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Create models directory
os.makedirs("models", exist_ok=True)

# Load dataset
data = pd.read_csv("dataset/reviews.csv")

# Remove missing and duplicate records
data = data.dropna(subset=["review", "sentiment"])
data = data.drop_duplicates()

# Clean review text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

data["review"] = data["review"].apply(clean_text)

# Separate features and labels
X = data["review"]
y = data["sentiment"].str.lower()

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create ML pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )),
    ("classifier", LogisticRegression(
    max_iter=2000,
    C=1.0,
    class_weight="balanced",
    random_state=42
))
])

# Train model
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nMODEL TRAINING COMPLETED")
print("------------------------")
print("Total records:", len(data))
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    zero_division=0
))

# Save trained model
joblib.dump(model, "models/sentiment_model.pkl")

# Save evaluation information
results = {
    "accuracy": round(float(accuracy), 4),
    "total_records": len(data),
    "training_records": len(X_train),
    "testing_records": len(X_test),
    "labels": sorted(y.unique().tolist())
}

with open("models/metrics.json", "w") as file:
    json.dump(results, file, indent=4)

print("\nModel saved successfully!")
print("Location: models/sentiment_model.pkl")
