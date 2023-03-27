

from transformers import pipeline

pipe_en2es = pipeline("translation", model="Helsinki-NLP/opus-mt-en-es")

def en2es(text):
    return pipe_en2es(text)[0]["translation_text"]

pipe_ner = pipeline("ner")

def ner(text):
    output = pipe_ner(text)
    return {"text": text, "entities": output} 