from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import json
tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/roberta-large-ner-english")
model = AutoModelForTokenClassification.from_pretrained("Jean-Baptiste/roberta-large-ner-english")
nlp = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy="simple")