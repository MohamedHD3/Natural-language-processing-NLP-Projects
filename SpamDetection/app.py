# conda activate base
# pip install -U streamlit
# pip install -U plotly

# you can run your app with: streamlit run app.py

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import string
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


def count_punctuation(text):
  sum = 0
  for char in text:
    if char in string.punctuation:
      sum += 1
  return sum


def count_words(text):
  lis = str(text).split()
  return len(lis)


stop_words = set(stopwords.words('english'))
def count_stopwords(text):
  words = nltk.word_tokenize(text)
  sum = 0
  for word in words:
    if word.lower() in stop_words:
      sum += 1
  return sum


def remove_email(text):
  email_pattern = r"^\S+@\S+\.\S+$"
  return re.sub(email_pattern, ' ', text)


def remove_url(text):
  url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
  return re.sub(url_pattern, ' ', text)


ps = PorterStemmer() # Source

def transform_text(text):
  text = remove_email(text)
  text = remove_url(text)
  sen = text.lower()
  sen = re.sub('[^a-z]', ' ', sen)
  sen = str(sen)
  sen = sen.split()
  sen = [word for word in sen if word not in stopwords.words('english')]
  sen = [ps.stem(word) for word in sen]
  return ' '.join(sen)


model = pickle.load(open('model_svm.pkl', 'rb'))

st.title('SMS Spam Classifier NLP')
st.image('i2.jpg')

input_sms = st.text_area('Enter the message')

st.sidebar.image('info2.jpg')
st.sidebar.title('Message Info.')

punctuation_count = count_punctuation(input_sms)
char_count = len(input_sms)
word_count = count_words(input_sms)
sentences_count = len(nltk.sent_tokenize(input_sms))
stopwords_count = count_stopwords(input_sms)

if st.button('Predict'):
  
  st.sidebar.write(f'Count of Punctuations: {punctuation_count}')
  st.sidebar.write(f'Count of Characters: {char_count}')
  st.sidebar.write(f'Count of Words: {word_count}')
  st.sidebar.write(f'Count of Sentences: {sentences_count}')
  st.sidebar.write(f'Count of Stopwords: {stopwords_count}')

  transform_sms = transform_text(input_sms)
  result = model.predict([transform_sms])
  if result == 1:
    st.header('SPAM 🙁')
  else:
    st.header('NOT SPAM 🙂')
    st.balloons()

  