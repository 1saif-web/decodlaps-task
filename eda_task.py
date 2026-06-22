import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

stop_words = set(stopwords.words('english'))
negations = {'not', 'no', 'never', 'neither', 'nor', 'but', 'against'}
custom_stopwords = stop_words - negations  

def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def clean_text_pipeline(text):
    text = str(text).lower()
    tokens = word_tokenize(text)
    cleaned_tokens = [word for word in tokens if word.isalnum() and word not in custom_stopwords]
    lemmatizer = WordNetLemmatizer()
    final_tokens = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in cleaned_tokens]
    return " ".join(final_tokens)

if __name__ == "__main__":
    data = {
        'review_text': [
            "This product is absolutely amazing and perfect!",
            "I am not happy with this purchase, it is terrible.",
            "Worst service ever, highly do not recommend.",
            "Very good quality, arrived fast and works fine.",
            "Not bad at all, actually it is quite useful."
        ],
        'sentiment': [1, 0, 0, 1, 1]
    }
    df = pd.DataFrame(data)
    df['cleaned_text'] = df['review_text'].apply(clean_text_pipeline)
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(df['cleaned_text'])
    y = df['sentiment']
    classifier = MultinomialNB()
    classifier.fit(X, y)
    print("\n[✓] NLP Engine Model Trained Successfully!")