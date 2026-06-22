import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# 1. تحميل حزم NLTK الأساسية
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# 2. بناء قائمة الكلمات الدلالية المستثنى منها أدوات النفي (قاعدة هندسية صارمة)
stop_words = set(stopwords.words('english'))
negations = {'not', 'no', 'never', 'neither', 'nor', 'but', 'against'}
custom_stopwords = stop_words - negations  # الحفاظ على النفي لعدم ضرب المشاعر

def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

# 3. خط معالجة النصوص المتكامل (Text Pre-Processing Pipeline)
def clean_text_pipeline(text):
    text = str(text).lower()  # Lowercasing
    tokens = word_tokenize(text)  # Tokenization
    cleaned_tokens = [word for word in tokens if word.isalnum() and word not in custom_stopwords]
    lemmatizer = WordNetLemmatizer()
    final_tokens = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in cleaned_tokens] # POS Lemmatization
    return " ".join(final_tokens)

if __name__ == "__main__":
    # بيانات اختبارية لمراجعات منتجات (فيها نفي ومشاعر)
    data = {
        'review_text': [
            "This product is absolutely amazing and perfect!",
            "I am not happy with this purchase, it is terrible.",
            "Worst service ever, highly do not recommend.",
            "Very good quality, arrived fast and works fine.",
            "Not bad at all, actually it is quite useful."
        ],
        'sentiment': [1, 0, 0, 1, 1]  # 1 = موجب، 0 = سالب
    }
    
    df = pd.DataFrame(data)
    print("\n--- جاري تشغيل خط المعالجة المسبقة وتنظيف النصوص ---")
    
    df['cleaned_text'] = df['review_text'].apply(clean_text_pipeline)
    print(df[['review_text', 'cleaned_text']])
    
    # 4. تحويل النصوص إلى أرقام ومصفوفات رياضية عبر TF-IDF
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(df['cleaned_text'])
    y = df['sentiment']
    
    # 5. تدريب الموديل للتنبؤ بالقطبية الإيجابية والسلبية
    classifier = MultinomialNB()
    classifier.fit(X, y)
    
    print("\n[✓] مبروك يا هندسة! تم معالجة النصوص وتدريب الموديل بنجاح!")