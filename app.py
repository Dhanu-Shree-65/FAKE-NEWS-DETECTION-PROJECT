import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop = stopwords.words("english")

model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

def clean_text(text):
    return " ".join([word for word in text.split() if word.lower() not in stop])

st.title("📰 Fake News Detection App")

input_text = st.text_area("Enter News Article Content Here:")

if st.button("Predict"):
    cleaned = clean_text(input_text)
    vector_input = tfidf.transform([cleaned])
    result = model.predict(vector_input)[0]
    st.success("✅ Real News" if result else "⚠️ Fake News")
