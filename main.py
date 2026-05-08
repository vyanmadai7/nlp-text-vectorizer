import pandas as pd
import nltk
import string
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer

data = pd.DataFrame({
    "text":["Hello !!! NLP @ is *awesome*???"]
})

def clean_text(text):
    tokens = word_tokenize(text)
    clean_tokens = []
    for word in tokens:
        if word not in string.punctuation:
            clean_tokens.append(word)

    return " ".join(clean_tokens)

data["clean_text"] = data["text"].apply(clean_text)
print("Cleaned Data: ")
print(data)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data["clean_text"])

print("\nFeature name:")
print(vectorizer.get_feature_names_out())

print("\nVectorized Array:")
print(X.toarray())
