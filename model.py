import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import numpy as np
import re
import nltk
from sklearn.datasets import load_files
nltk.download('stopwords')
import pickle
from nltk.corpus import stopwords
nltk.download('wordnet')

cat_data = load_files(r"database_ml/")
X, y = cat_data.data, cat_data.target

# for i in cat_data.target_names:
#     print(i)
    
documents = []

from nltk.stem import WordNetLemmatizer

stemmer = WordNetLemmatizer()

for sen in range(0, len(X)):
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(X[sen]))
    
    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
    
    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 
    
    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)
    
    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)
    
    # Converting to Lowercase
    document = document.lower()
    
    # Lemmatization
    document = document.split()

    document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)
    
    documents.append(document)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state = 0)

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5,stop_words='english')

fitted_vectorizer = tfidf.fit(X_train)
tfidf_vectorizer_vectors = fitted_vectorizer.transform(X_train)

from sklearn.svm import LinearSVC
classifier = LinearSVC().fit(tfidf_vectorizer_vectors, y_train)

y_pred =classifier.predict(fitted_vectorizer.transform(X_test))

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# print(accuracy_score(y_test, y_pred))


pickle.dump(classifier, open('model.pkl', 'wb'))



# import urllib.request
# from inscriptis import get_text

# url = input("Enter URL: ")
# html = urllib.request.urlopen(url).read().decode('utf-8')

# text = get_text(html)

# extracted_data=text.split()
# refined_data=[]
# SYMBOLS = '{}()[].,:;+-*/&|<>=~0123456789' 
# for i in extracted_data:
# 	if i not in SYMBOLS:
# 		refined_data.append(i)

# # print("\n","$"*50,"HEYAAA we got arround: ",len(refined_data)," of keywords! Here are they: ","$"*50,"\n")
# predict_this=" ".join(refined_data)
# category_predicted=model.predict(fitted_vectorizer.transform([predict_this]))
# print("-"*100)
# print("Predicted Category for giver URL is: ",cat_data.target_names[int(category_predicted)])