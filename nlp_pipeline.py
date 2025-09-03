import spacy
nlp = spacy.load("xx_ent_wiki_sm")

def analyze_text(text):
    doc = nlp(text)
    return {
        "tokens": [token.text for token in doc],
        "lemmas": [token.lemma_ for token in doc],
        "pos": [token.pos_ for token in doc],
        "entities": [(ent.text, ent.label_) for ent in doc.ents]
    }