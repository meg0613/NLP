import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from src.language_detection import detect_language
from src.translator import translate_to_english
from src.nlp_pipeline import analyze_text
from src.summarizer import summarize
from src.sentiment import sentiment_analysis

st.set_page_config(page_title="Multi-language NLP Pipeline", layout="wide")
st.title("Multi-language NLP Pipeline")

user_input = st.text_area("Enter text in any language:", height=200)

if st.button("Run Analysis") and user_input.strip():
    lang = detect_language(user_input)
    st.markdown(f"**Detected Language:** `{lang}`")

    if lang != 'en':
        try:
            translated = translate_to_english(user_input, lang)
            st.markdown("**Translated to English:**")
            st.write(translated)
        except:
            translated = user_input
            st.warning("Translation failed, using original text.")
    else:
        translated = user_input

    st.subheader("NLP Analysis")
    analysis = analyze_text(translated)
    st.write("**Tokens:**", analysis["tokens"])
    st.write("**Entities:**", analysis["entities"])

    st.subheader("Sentiment")
    sentiment = sentiment_analysis(translated)
    st.json(sentiment)

    st.subheader("Summary")
    summary = summarize(translated)
    st.write(summary)