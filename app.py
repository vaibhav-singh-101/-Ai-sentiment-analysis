
import os
import re
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="AI Sentiment Analyzer",
    page_icon="🤖",
    layout="wide"
)

MODEL_PATH = "models/sentiment_model.pkl"
METRICS_PATH = "models/metrics.json"

st.title("🤖 AI Customer Review Sentiment Analyzer")
st.write(
    "Analyze customer reviews using Machine Learning "
    "and Natural Language Processing."
)

# Check model availability
if not os.path.exists(MODEL_PATH):
    st.error("Model not found. Please run train_model.py first.")
    st.stop()

# Load model
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# Model information
if os.path.exists(METRICS_PATH):
    import json

    with open(METRICS_PATH, "r") as file:
        metrics = json.load(file)

    col1, col2, col3 = st.columns(3)

    col1.metric("Dataset Records", metrics["total_records"])
    col2.metric("Model Accuracy", f'{metrics["accuracy"] * 100:.2f}%')
    col3.metric("Testing Records", metrics["testing_records"])

st.divider()

st.subheader("Analyze a Customer Review")

review = st.text_area(
    "Enter your customer review:",
    placeholder="Example: This product is excellent and amazing!"
)

if st.button("Analyze Sentiment", type="primary"):
    if not review.strip():
        st.warning("Please enter a review first.")

    else:
        cleaned_review = clean_text(review)

        if not cleaned_review:
            st.warning("Please enter a review containing letters.")
        else:
            prediction = model.predict([cleaned_review])[0]

            probabilities = model.predict_proba([cleaned_review])[0]

            confidence = max(probabilities) * 100

            st.subheader("Prediction Result")

            if prediction == "positive":
                st.success("😊 Positive Sentiment")
            elif prediction == "negative":
                st.error("😞 Negative Sentiment")
            else:
                st.info("😐 Neutral Sentiment")

            st.write("Predicted Sentiment:", prediction.title())
            st.write(f"Prediction Confidence: {confidence:.2f}%")

            st.progress(float(confidence / 100))

            probability_data = pd.DataFrame({
                "Sentiment": model.classes_,
                "Probability": probabilities * 100
            })

            st.bar_chart(
                probability_data.set_index("Sentiment")
            )

st.divider()

st.subheader("Analyze Multiple Reviews")

uploaded_file = st.file_uploader(
    "Upload a CSV file with a review column",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        uploaded_data = pd.read_csv(uploaded_file)

        if "review" not in uploaded_data.columns:
            st.error("Your CSV must contain a column named 'review'.")

        else:
            uploaded_data = uploaded_data.dropna(
                subset=["review"]
            ).copy()

            if uploaded_data.empty:
                st.warning("No valid reviews found.")
            else:
                cleaned_reviews = uploaded_data["review"].apply(
                    clean_text
                )

                uploaded_data["sentiment"] = model.predict(
                    cleaned_reviews
                )

                st.success("Reviews analyzed successfully!")

                st.dataframe(
                    uploaded_data,
                    use_container_width=True
                )

                st.subheader("Sentiment Distribution")

                counts = uploaded_data["sentiment"].value_counts()

                st.bar_chart(counts)

                csv_data = uploaded_data.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label="Download Prediction Results",
                    data=csv_data,
                    file_name="sentiment_results.csv",
                    mime="text/csv"
                )

    except Exception as error:
        st.error(f"Error processing CSV: {error}")

st.divider()

st.caption(
    "Developed using Python, Scikit-learn, TF-IDF "
    "and Streamlit | AI/ML Internship Project"
)
