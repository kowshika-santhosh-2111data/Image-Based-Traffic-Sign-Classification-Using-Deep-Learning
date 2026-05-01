import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import re
from sklearn.preprocessing import StandardScaler



#-------------------------------Image Data Preprocessing------------------------------------
def preprocess_image_data(x):
    # Normalize pixel values
    x = x / 255.0
    return x

#-------------------------------Tabular Data Preprocessing------------------------------------
def preprocess_tabular_data(df,target_column):
    # Handle missing values (example: fill with mean)
    df = df.fillna(df.mean(numeric_only=True))
    
    # Separate features and target variable
    x = df.drop(target_column, axis=1)
    y = df[target_column]
    
    # Standardize features
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    
    return x_scaled, y, scaler

#-------------------------------Text Data Preprocessing------------------------------------
def clean_text(text):   
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

#-------------------------------tokenization-------------------------------------
def tokenize_text(text):
    # Simple whitespace tokenization
    tokens = text.split()
    return tokens

#-------------------------------stop word removal-------------------------------------
def remove_stop_words(tokens, stop_words):
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return filtered_tokens

#-------------------------------tf idf vectorization-------------------------------------
from sklearn.feature_extraction.text import TfidfVectorizer
def tfidf_vectorize(corpus):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    return tfidf_matrix, vectorizer.get_feature_names_out()

