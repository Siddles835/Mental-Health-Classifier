import streamlit as st
import joblib

st.set_page_config(
    page_title="Mental Health Classifier",
    page_icon="",
    layout="centered"
)

model = joblib.load("models/logreg_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
encoder = joblib.load("models/label_encoder.pkl")

st.title("Mental Health Classifier")

st.write(
    "Enter a piece of text and the AI will predict the mental health category."
)

text = st.text_area(
    "Enter text:",
    height=200,
    placeholder="Example: I've been feeling overwhelmed and anxious lately..."
)

if st.button("Predict"):

    if text.strip() == "":
        st.warning("Please enter some text.")

    else:

        X = vectorizer.transform([text])

        prediction = model.predict(X)

        category = encoder.inverse_transform(prediction)[0]

        st.success(f"### Predicted Category: **{category}**")