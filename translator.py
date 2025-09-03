from transformers import MarianMTModel, MarianTokenizer

def translate_to_english(text, src_lang):
    model_name = f'Helsinki-NLP/opus-mt-{src_lang}-en'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translated = model.generate(**tokens)
    return tokenizer.decode(translated[0], skip_special_tokens=True)