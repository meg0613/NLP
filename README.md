# Multi-language NLP Pipeline

This project processes text in various languages by detecting the language, translating to English, performing sentiment analysis, NER, tokenization, and summarization.

## Features
- Language Detection (`langdetect`)
- Translation (`Helsinki-NLP` MarianMT models)
- Tokenization, POS tagging, NER (`spaCy`)
- Sentiment Analysis (`XLM-Roberta`)
- Summarization (`mBART`)

## How to Run

```bash
pip install -r requirements.txt
python -m spacy download xx_ent_wiki_sm
streamlit run app/main.py
```

## Folder Structure

- `src/` – Core modules
- `app/` – Streamlit app
- `models/`, `data/` – Optional data/model storage

