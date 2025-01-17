from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn",max_length=90,min_length=50)
out = summarizer("Paris is the capital and most populous city of France, with an estimated population of 2,175,601 residents as of 2018, in an area of more than 105 square kilometres (41 square miles). The City of Paris is the centre and seat of government of the region and province of Île-de-France, or Paris Region, which has an estimated population of 12,174,880, or about 18 percent of the population of France as of 2017.")
print(out)
## [{ "summary_text": " Paris is the capital and most populous city of France..." }]
