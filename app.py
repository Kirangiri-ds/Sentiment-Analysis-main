import streamlit as st
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk
from ntscraper import Nitter
from PIL import Image

# Download stopwords once, using Streamlit's caching
@st.cache_resource
def load_stopwords():
    nltk.download('stopwords')
    return stopwords.words('english')

# Load model and vectorizer once
@st.cache_resource
def load_model_and_vectorizer():
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer

# Define sentiment prediction function
def predict_sentiment(text, model, vectorizer, stop_words):
    # Preprocess text
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [word for word in text if word not in stop_words]
    text = ' '.join(text)
    text = [text]
    text = vectorizer.transform(text)
    
    # Predict sentiment
    sentiment = model.predict(text)
    return "Negative" if sentiment == 0 else "Positive"

# Initialize Nitter scraper
@st.cache_resource
def initialize_scraper():
    return Nitter(log_level=1)

# Function to create a colored card
def create_card(tweet_text, sentiment):
    color = "green" if sentiment == "Positive" else "red"
    card_html = f"""
    <div style="background-color: {color}; padding: 10px; border-radius: 5px; margin: 10px 0;">
        <h5 style="color: white;">{sentiment} Sentiment</h5>
        <p style="color: white;">{tweet_text}</p>
    </div>
    """
    return card_html

# Main app logic
def main():
    st.set_page_config(page_title="Text Sentiment Predictor", page_icon="💬", layout="centered")
    st.markdown("""
        <style>
        .main-title {font-size:2.5em; font-weight:bold; color:#4F8BF9; text-align:center; margin-bottom:0.2em;}
        .desc {font-size:1.2em; color:#555; text-align:center; margin-bottom:1em;}
        .footer {font-size:1em; color:#888; text-align:center; margin-top:2em;}
        .stButton>button {background-color:#4F8BF9; color:white; font-size:1.1em; border-radius:8px;}
        </style>
    """, unsafe_allow_html=True)

    st.sidebar.title("About")
    try:
        img = Image.open("images/kiran_giri.jpg")
        img_rotated = img.rotate(90, expand=True)
        st.sidebar.image(img_rotated, caption="Kiran Giri", use_container_width=True)
    except Exception as e:
        st.sidebar.warning(f"Image could not be loaded: {e}")
    st.sidebar.info("This app uses machine learning to predict sentiment (positive/negative) from your text input.")
    st.sidebar.markdown("Made by **Kiran Giri** 💙")

    st.markdown('<div class="main-title">Text Sentiment Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="desc">Enter any text below to analyze its sentiment.</div>', unsafe_allow_html=True)

    stop_words = load_stopwords()
    model, vectorizer = load_model_and_vectorizer()

    text_input = st.text_area("Text to analyze", "", height=120)
    analyze_col, result_col = st.columns([1,2])
    with analyze_col:
        analyze = st.button("🔍 Analyze", use_container_width=True)
    with result_col:
        if analyze:
            if text_input.strip():
                sentiment = predict_sentiment(text_input, model, vectorizer, stop_words)
                emoji = "😊" if sentiment == "Positive" else "😞"
                card_html = create_card(text_input, sentiment)
                st.markdown(card_html, unsafe_allow_html=True)
                st.markdown(f"<h4 style='text-align:center;'>Result: {sentiment} {emoji}</h4>", unsafe_allow_html=True)
            else:
                st.warning("Please enter some text to analyze.")

    st.markdown('<div class="footer">Made by <b>Kiran Giri</b> © 2025</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
