from rank_bm25 import BM25Okapi
from sklearn.feature_extraction import _stop_words
import string

#####

# Note:
# I'm not doing it in this simple example, but you should also:
# - remove punctuation
# - convert all text to lower case
# - remove leading and trailing spaces

# 'document.' will not be matched to 'document'

# This is an example function:
def bm25_tokenizer(text):
    tokenized_doc = []
    for token in text.lower().split():
        token = token.strip(string.punctuation)

        if len(token) > 0 and token not in _stop_words.ENGLISH_STOP_WORDS:
            tokenized_doc.append(token)
    return tokenized_doc

#####



# Sample text
documents = [
    "This is the first document",
    "This document is the second IS999333 document.",
    "And this is the third one.",
    "Is this the first document",
]

# Preprocess documents (split into tokens)
tokenized_documents = [doc.split(" ") for doc in documents]

# Initialize BM25 model
bm25 = BM25Okapi(tokenized_documents)

# Query - Search for a serial number
query = 'IS999333'

# Tokenize the query
tokenized_query = query.split(" ")

# Perform the keyword search
scores = bm25.get_scores(tokenized_query)

# Rank documents based on scores
ranked_documents = sorted(zip(scores, documents), reverse=True)

# Display ranked documents
for score, document in ranked_documents:
    print(f"Score: {score:.2f}, Document: {document}")
