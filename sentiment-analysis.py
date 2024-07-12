#pip install -q transformers
# list of models
# https://huggingface.co/models?other=sentiment-analysis
from transformers import pipeline
sentiment_pipeline = pipeline("sentiment-analysis",model='distilbert/distilbert-base-uncased-finetuned-sst-2-english')
data = ["I love you", "I hate you"]
out = sentiment_pipeline(data)
print(out)
# [{'label': 'POSITIVE', 'score': 0.9998656511306763}, {'label': 'NEGATIVE', 'score': 0.9991129040718079}]
