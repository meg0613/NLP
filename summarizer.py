from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

def summarize(text, src_lang="fr_XX"):
    tokenizer.src_lang = src_lang
    encoded = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    generated = model.generate(**encoded, forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"])
    return tokenizer.decode(generated[0], skip_special_tokens=True)