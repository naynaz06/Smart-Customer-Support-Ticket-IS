import re
import joblib
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Smart Customer Support Ticket Classifier",
    page_icon="🎫",
    layout="centered"
)

# Load saved model and vectorizer
@st.cache_resource
def load_artifacts():
    model = joblib.load("models/final_svm_model.pkl")
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_artifacts()

# Text cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# App interface
st.title("🎫 Smart Customer Support Ticket Classifier")

st.write(
    "Enter a customer support ticket to predict its category."
)

ticket = st.text_area(
    "Customer Ticket",
    placeholder="Example: My card is not working..."
)

if st.button("Predict Category", type="primary"):
    if ticket.strip():
        cleaned_ticket = clean_text(ticket)
        ticket_features = vectorizer.transform([cleaned_ticket])
        prediction = model.predict(ticket_features)[0]

        st.success("Prediction completed!")
        st.subheader("Predicted Category")
        st.code(prediction)
    else:
        st.warning("Please enter a customer ticket first.")